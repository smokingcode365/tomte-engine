"""Current-message-only activation evaluation; no production changes or LLM calls.

Run from tomte-engine. Uses real configured embeddings and rollback-only local
PostgreSQL fixtures. Policies/MRU are offline. Redis is read only. Reports/cache
remain in ignored log/. Prior evaluation scripts are imported, never modified.
"""
from bisect import bisect_left, bisect_right
from collections import Counter
from copy import deepcopy
import csv
import hashlib
import html
import json
import logging
import math
import os
from pathlib import Path
import statistics
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import load_dotenv
import httpx
import psycopg
import redis

from evaluate_memory_retrieval import entity, cosine, state, RollbackEvaluation
from evaluate_session_entities import fixtures, lexical_hits, SessionState
from tomte.engine.core.config import settings
from tomte.engine.llm.embedder import get_entity_summary
from tomte.engine.providers.openai_provider import OpenAILLMProvider
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.utils.security import SecurityGuard

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "log" / "entity_activation_evaluation.json"
CACHE = ROOT / "log" / "memory_retrieval_embeddings.json"
USAGE = ROOT / "log" / "entity_activation_api_usage.json"
FLOORS = (.20, .25, .30, .35, .40, .45)


def dataset():
    bank, conversations = fixtures()
    additions = [
        ("kids_math", "Kids Math Website", "activities_and_skills", "software_project", ["Number Garden"],
         [("purpose", "children's arithmetic practice"), ("language", "JavaScript"), ("framework", "React"), ("project_status", "active")]),
        ("guitar", "Acoustic Guitar", "activities_and_skills", "hobby", ["Evening Strings"],
         [("practice", "folk songs"), ("instrument", "acoustic guitar"), ("practice_time", "evenings"), ("skill_level", "beginner")]),
        ("basketball", "Falcons Basketball Club", "activities_and_skills", "sport", ["Falcons"],
         [("sport", "basketball"), ("member", "Terry"), ("practice_day", "Thursday"), ("location", "Oak gym")]),
        ("swimming", "Dolphins Swim Club", "activities_and_skills", "sport", ["Dolphins"],
         [("sport", "swimming"), ("member", "Bryan"), ("practice_day", "Saturday"), ("location", "Maple pool")]),
        ("tokyo", "Tokyo Holiday", "events_and_logistics", "vacation", ["Japan Getaway"],
         [("destination", "Tokyo"), ("country", "Japan"), ("travel_status", "completed"), ("travel_month", "April")]),
        ("degree", "Computer Science Degree", "activities_and_skills", "education", ["University Diploma"],
         [("qualification", "bachelor's degree"), ("subject", "computer science"), ("institution", "Westlake University"), ("graduation_year", "2018")]),
        ("certificate", "Cloud Certification", "activities_and_skills", "education", ["Cloud Badge"],
         [("qualification", "professional certificate"), ("subject", "cloud architecture"), ("issuer", "AWS"), ("completion_year", "2025")]),
    ]
    for spec in additions:
        bank[spec[0]] = entity(*spec)
    families = {
        "vehicles": ["tesla", "honda", "toyota"], "people": ["terry", "bryan", "alex"],
        "pets": ["dog", "cat"], "jobs": ["current_employer", "previous_employer", "job_role"],
        "homes": ["current_home", "previous_home"], "skills": ["python", "java"],
        "projects": ["tomte", "kids_math"], "hobbies": ["main_aquarium", "second_aquarium", "guitar"],
        "sports": ["basketball", "swimming"], "travel": ["vacation", "tokyo"], "education": ["degree", "certificate"],
    }
    family = {key: name for name, ids in families.items() for key in ids}
    # Explicit ambiguity sets are stronger than assuming one winner for a generic noun.
    types = {
        "vehicles": ("My car needs attention.", families["vehicles"]),
        "people": ("My son has practice.", ["terry", "bryan"]),
        "pets": ("My pet needs care.", families["pets"]),
        "jobs": ("My employer is changing.", ["current_employer", "previous_employer"]),
        "homes": ("My home needs repairs.", families["homes"]),
        "skills": ("My programming language is giving me trouble.", families["skills"]),
        "projects": ("My software project is broken.", families["projects"]),
        "hobbies": ("My aquarium needs cleaning.", ["main_aquarium", "second_aquarium"]),
        "sports": ("The sports club has practice.", families["sports"]),
        "travel": ("I want to discuss my holiday.", families["travel"]),
        "education": ("I want to discuss my qualifications.", families["education"]),
    }
    facts = {
        "tesla": ("The vehicle I bought in 2024 needs service.", "The one I charge at home is making noise."),
        "honda": ("The vehicle I bought in 2020 needs service.", "My gasoline commuter vehicle needs maintenance."),
        "toyota": ("The white vehicle I sold needs paperwork.", "The vehicle I used for mountain camping needs paperwork."),
        "terry": ("The child who plays basketball has practice.", "The child attending Oak Secondary needs help."),
        "bryan": ("My ten-year-old son needs help.", "The child attending Maple Elementary needs help."),
        "alex": ("My coworker who specializes in databases needs help.", "My cycling colleague at Northstar Labs needs help."),
        "dog": ("My four-year-old beagle is scratching.", "The beagle I keep as a pet needs attention."),
        "cat": ("My two-year-old tabby needs a checkup.", "The tabby I keep as a pet needs attention."),
        "current_employer": ("The robotics employer I joined in 2024 changed policy.", "My hybrid robotics workplace in Vancouver changed policy."),
        "previous_employer": ("The employer I left in 2023 sent paperwork.", "My former banking software workplace sent paperwork."),
        "job_role": ("My position managing eight engineers is stressful.", "My leadership role involving hiring is stressful."),
        "current_home": ("My two-bedroom home needs repairs.", "The place where I live near the harbor needs repairs."),
        "previous_home": ("My former one-bedroom home needs paperwork.", "The downtown Toronto residence I left needs paperwork."),
        "vacation": ("The holiday planned for November needs preparation.", "My upcoming Portuguese museum trip needs preparation."),
        "tokyo": ("The Japan holiday completed in April needs a photo album.", "The holiday where I visited Japan needs a photo album."),
        "python": ("The language I use with pandas is giving me trouble.", "My intermediate data-analysis scripting skill needs practice."),
        "java": ("The language I use with Spring Boot is giving me trouble.", "My enterprise backend programming skill needs practice."),
        "tomte": ("The memory project using PostgreSQL and Redis is broken.", "My AI assistant's entity-memory system needs debugging."),
        "kids_math": ("The React website for children's arithmetic is broken.", "My website helping children practice addition needs debugging."),
        "main_aquarium": ("The 200-liter aquarium has algae.", "The freshwater tank where my angelfish live needs cleaning."),
        "second_aquarium": ("The 60-liter aquarium needs cleaning.", "The freshwater shrimp tank in my study needs cleaning."),
        "guitar": ("The instrument I use for evening folk songs needs tuning.", "My beginner acoustic-string practice needs improvement."),
        "basketball": ("Terry's Thursday basketball club practice was moved.", "The basketball club practicing at Oak gym changed its schedule."),
        "swimming": ("Bryan's Saturday swim club practice was moved.", "The swimming club practicing at Maple pool changed its schedule."),
        "degree": ("My bachelor's qualification from 2018 needs verification.", "My Westlake University computing qualification needs verification."),
        "certificate": ("My AWS certificate completed in 2025 needs renewal.", "My professional cloud-architecture qualification needs renewal."),
    }
    cases = []
    def add(category, text, gold, excluded=(), group=None):
        cases.append(dict(id=f"case-{len(cases)+1:03}", category=category, text=text, gold=list(gold),
            excluded=list(excluded), group=group, ambiguous=len(gold)>1 and category in ("type", "ambiguous")))
    for key, e in bank.items():
        group = family[key]
        add("name", f"I want to discuss {e['entity_name']}.", [key], group=group)
        add("name", f"What do you remember about {e['entity_name']}?", [key], group=group)
        add("alias", f"I need help with {e['aliases'][0]}.", [key], group=group)
        text, gold = types[group]
        if key == "alex": text, gold = "My coworker needs help.", [key]
        if key == "job_role": text, gold = "My job role is stressful.", [key]
        if key == "guitar": text, gold = "My musical instrument needs tuning.", [key]
        add("type", text, gold, group=group)
        fact, indirect = facts[key]
        # Explicit positive mentions of another stored identity are legitimate
        # activation targets too, even when the main noun is a club/colleague.
        fact_gold = [key]+(["terry"] if key == "basketball" else ["bryan"] if key == "swimming" else [])
        indirect_gold = [key]+(["current_employer"] if key == "alex" else [])
        add("fact", fact, fact_gold, group=group)
        add("indirect", indirect, indirect_gold, group=group)
        add("confusable", f"I need help with {e['entity_name']}, specifically.", [key], group=group)
    multis = [
        ["tesla", "honda"], ["terry", "bryan"], ["dog", "cat"], ["python", "java"],
        ["current_employer", "previous_employer"], ["current_home", "previous_home"],
        ["tomte", "kids_math"], ["basketball", "swimming"],
        ["tesla", "terry", "dog"], ["python", "tomte", "kids_math"],
        ["main_aquarium", "second_aquarium", "cat"], ["degree", "certificate", "job_role"],
        ["tesla", "honda", "toyota"], ["vacation", "tokyo", "current_home"],
        ["tesla", "terry", "dog", "python", "main_aquarium"],
        ["honda", "bryan", "cat", "java", "kids_math"],
        ["current_employer", "job_role", "degree", "certificate", "tomte"],
        ["vacation", "tokyo", "guitar", "basketball", "swimming"],
    ]
    for ids in multis:
        add("multi", "I need to discuss "+", ".join(bank[k]["entity_name"] for k in ids)+".", ids)
    unrelated = ["What is the weather tomorrow?", "Explain quantum mechanics.", "What time is sunset?",
        "What is twelve times eleven?", "Tell me a joke.", "How far is the moon from Earth?",
        "Translate hello into Spanish.", "What is the capital of France?", "Who invented the telephone?",
        "Explain the water cycle.", "What is the square root of 144?", "How do volcanoes form?",
        "Define a prime number.", "Explain photosynthesis.", "What is the speed of light?",
        "Give me a riddle.", "Convert ten inches to centimeters.", "What causes an eclipse?",
        "Explain the difference between a comet and an asteroid.", "What is the boiling point of water?"]
    for text in unrelated: add("unrelated", text, [])
    negated = [
        ("Not the Tesla, I mean the Honda.", ["honda"], ["tesla"]),
        ("Not my Honda; the Tesla needs service.", ["tesla"], ["honda"]),
        ("I'm not talking about Terry.", [], ["terry"]),
        ("Not Terry; Bryan needs help.", ["bryan"], ["terry"]),
        ("Not Bryan; Terry needs help.", ["terry"], ["bryan"]),
        ("Not my dog; my cat needs care.", ["cat"], ["dog"]),
        ("Not my cat; my dog needs care.", ["dog"], ["cat"]),
        ("Ignore my old house; I mean my current home.", ["current_home"], ["previous_home"]),
        ("Ignore Cedar Systems; I mean Northstar Labs.", ["current_employer"], ["previous_employer"]),
        ("Not Java; Python is giving me trouble.", ["python"], ["java"]),
        ("Not Python; Java is giving me trouble.", ["java"], ["python"]),
        ("Not Tomte; the Kids Math Website is broken.", ["kids_math"], ["tomte"]),
        ("Not the Main Aquarium; the Second Aquarium needs cleaning.", ["second_aquarium"], ["main_aquarium"]),
        ("Not Lisbon Holiday; I mean Tokyo Holiday.", ["tokyo"], ["vacation"]),
        ("I'm not discussing my Cloud Certification.", [], ["certificate"]),
    ]
    for text, gold, excluded in negated: add("negated", text, gold, excluded)
    for text, gold in [
        ("My car has a problem.", families["vehicles"]), ("My son needs help.", ["terry", "bryan"]),
        ("The project is failing.", families["projects"]), ("My pet needs care.", families["pets"]),
        ("My home needs attention.", families["homes"]), ("My employer sent paperwork.", ["current_employer", "previous_employer"]),
        ("My programming language is hard to use.", families["skills"]),
        ("My aquarium has a problem.", ["main_aquarium", "second_aquarium"]),
        ("My holiday needs planning.", families["travel"]), ("My qualification needs verification.", families["education"]),
        ("The sports club needs attention.", families["sports"]), ("Which blue car do I own?", ["tesla", "honda"]),
    ]: add("ambiguous", text, gold)
    # Every third case within a category calibrates; remaining cases are held out.
    # Same catalog is shared: this tests phrasing, not unseen-entity generalization.
    counters = Counter()
    for case in cases:
        case["split"] = "calibration" if counters[case["category"]] % 3 == 0 else "holdout"
        counters[case["category"]] += 1
    return bank, family, cases, conversations


def rules():
    result = [dict(name="weighted_top5", family="weighted", cap=5)]
    result += [dict(name=f"cosine_top{k}", family="topk", cap=k) for k in (1, 3, 5)]
    for floor in FLOORS:
        result.append(dict(name=f"floor_{floor:.2f}_cap10", family="floor", floor=floor, cap=10))
        result += [dict(name=f"floor_{floor:.2f}_top{k}", family="floor_topk", floor=floor, cap=k) for k in (1, 3, 5)]
        result += [dict(name=f"lexical_{floor:.2f}_cap{k}", family="lexical", floor=floor, cap=k) for k in (5, 8, 10)]
    result += [dict(name=f"margin_{floor:.2f}_{gap:.2f}", family="margin", floor=floor, margin=gap, cap=5)
               for floor in (.25, .30, .35, .40) for gap in (.02, .05)]
    return result


def activate(scores, factor, matches, rule):
    values = {k: v*factor[k] if rule["family"] == "weighted" else v for k, v in scores.items()}
    order = sorted(scores, key=lambda k: (-values[k], k))
    if rule["family"] in ("weighted", "topk"):
        return order[:rule["cap"]]
    passing = [k for k in order if values[k] >= rule["floor"]]
    if rule["family"] == "margin":
        # Observable fixed-rank background; not guaranteed irrelevant, never gold-labelled.
        passing = [k for k in passing if values[k]-values[order[5]] >= rule["margin"]]
    if rule["family"] == "lexical":
        passing = list(set(passing) | matches)
        passing.sort(key=lambda k: (k not in matches, -values[k], k))
    return passing[:rule["cap"]]


def metrics(records):
    true = sum(len(set(r["activated"]) & set(r["gold"])) for r in records)
    false = sum(len(set(r["activated"])-set(r["gold"])) for r in records)
    missed = sum(len(set(r["gold"])-set(r["activated"])) for r in records)
    precision = true/(true+false) if true+false else 1.0
    recall = true/(true+missed) if true+missed else None
    n = len(records)
    definite = [r for r in records if not r["ambiguous"]]
    ambiguous = [r for r in records if r["ambiguous"]]
    required = sum(len(r["gold"]) for r in definite)
    required_missed = sum(len(set(r["gold"])-set(r["activated"])) for r in definite)
    return dict(cases=n, true_positives=true, false_positives=false, false_negatives=missed,
        micro_precision=precision, micro_recall=recall,
        f1=2*precision*recall/(precision+recall) if recall is not None and precision+recall else 0,
        exact_set_accuracy=sum(set(r["gold"])==set(r["activated"]) for r in records)/n,
        false_per_turn=false/n, missed_per_turn=missed/n,
        turns_with_false=sum(bool(set(r["activated"])-set(r["gold"])) for r in records)/n,
        turns_missing=sum(bool(set(r["gold"])-set(r["activated"])) for r in records)/n,
        activated_per_turn=(true+false)/n,
        definite_entity_recall=(required-required_missed)/required if required else None,
        required_missed_per_turn=required_missed/n,
        ambiguous_any_valid_hit=sum(bool(set(r["gold"]) & set(r["activated"])) for r in ambiguous)/len(ambiguous) if ambiguous else None,
        ambiguous_clean_nonempty_set=sum(bool(r["activated"]) and set(r["activated"])<=set(r["gold"])
                                        for r in ambiguous)/len(ambiguous) if ambiguous else None)


def distribution(values):
    xs = sorted(values)
    def p(q):
        position = q*(len(xs)-1)
        lower, upper = math.floor(position), math.ceil(position)
        return xs[lower]+(xs[upper]-xs[lower])*(position-lower)
    return dict(count=len(xs), mean=statistics.mean(xs), median=p(.5), p10=p(.1), p25=p(.25),
                p75=p(.75), p90=p(.9), min=xs[0], max=xs[-1])


def score_rule(rule, cases, bank, scores, factors):
    records = []
    for case in cases:
        selected = activate(scores[case["text"]], factors, lexical_hits(case["text"], bank), rule)
        records.append(dict(**case, activated=selected))
    return dict(rule=rule, overall=metrics(records),
        calibration=metrics([r for r in records if r["split"] == "calibration"]),
        holdout=metrics([r for r in records if r["split"] == "holdout"]),
        categories={category: metrics([r for r in records if r["category"] == category])
                    for category in sorted({r["category"] for r in records})}, records=records)


def session_simulation(rule, conversations, bank, scores, factors):
    state = SessionState(10)
    records = []
    for conversation in conversations:
        key = ("tenant", "user", conversation["name"])
        # Pollution excludes every identity genuinely relevant anywhere in this conversation.
        ever_valid = {identity for t in conversation["turns"] for identity in t["gold"]}
        for index, turn in enumerate(conversation["turns"]):
            if turn["role"] == "assistant":
                before = list(state.states.get(key, []))
                assert state.touch(key, index*60) == before
                continue
            selected = activate(scores[turn["text"]], factors, lexical_hits(turn["text"], bank), rule)
            retained = state.touch(key, index*60, selected)
            returned = retained[:5]
            records.append(dict(conversation=conversation["name"], text=turn["text"], gold=turn["gold"],
                activated=selected, retained=retained, returned=returned,
                pollution=len(set(retained)-ever_valid), irrelevant_returned=len(set(returned)-set(turn["gold"]))))
    positive = [r for r in records if r["gold"]]
    return dict(records=records, mean_pollution=statistics.mean(r["pollution"] for r in records),
        max_pollution=max(r["pollution"] for r in records),
        all_required_available=statistics.mean(set(r["gold"]) <= set(r["returned"]) for r in positive),
        irrelevant_returned=statistics.mean(r["irrelevant_returned"] for r in records))


def main():
    if Path.cwd().resolve() != ROOT: raise RuntimeError("Run from tomte-engine")
    load_dotenv(ROOT / ".env", override=False)
    for name in ("OPENAI_API_KEY", "POSTGRES_URL", "REDIS_URL"):
        if not os.environ.get(name): raise RuntimeError(f"Missing {name}")
    if any(urlsplit(os.environ[n]).hostname not in ("localhost", "127.0.0.1", "::1") for n in ("POSTGRES_URL", "REDIS_URL")):
        raise RuntimeError("Only local infrastructure is allowed")
    assert settings.openai.vector_dimension == 1536 and not settings.enable_pii_scrubbing
    for name in ("httpx", "httpcore", "openai"): logging.getLogger(name).setLevel(logging.CRITICAL)
    bank, family, cases, conversations = dataset()
    guard = SecurityGuard()
    texts = [c["text"] for c in cases] + [t["text"] for c in conversations for t in c["turns"] if t["role"] == "user"]
    for text in texts:
        assert text == guard.sanitize_input(text) and guard.check_for_injection(text)
    summaries = {k: get_entity_summary(e) for k, e in bank.items()}
    OUTPUT.parent.mkdir(exist_ok=True)
    cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    usage = json.loads(USAGE.read_text()) if USAGE.exists() else []
    reused, send = {}, httpx.AsyncClient.send
    async def embeddings_only(client, request, *args, **kwargs):
        if not request.url.path.endswith("/embeddings"): raise RuntimeError("Non-embedding request blocked")
        body = json.loads(request.content)
        response = await send(client, request, *args, **kwargs)
        await response.aread()
        usage.append(dict(model=body.get("model"), texts=len(body["input"]) if isinstance(body["input"], list) else 1,
                          status=response.status_code, usage=response.json().get("usage", {})))
        USAGE.write_text(json.dumps(usage, indent=2), encoding="utf-8")
        return response
    provider = OpenAILLMProvider(os.environ["OPENAI_API_KEY"])
    def cache_key(role, text):
        return hashlib.sha256(json.dumps([settings.openai.embed_model, 1536, role, text]).encode()).hexdigest()
    def embed(role, inputs):
        unique = list(dict.fromkeys(inputs))
        pending = [t for t in unique if cache_key(role,t) not in cache]
        reused[role] = len(unique)-len(pending)
        for start in range(0,len(pending),100):
            chunk = pending[start:start+100]
            vectors = provider.embed_batch(chunk) if role == "document" else provider.embedder.embed_query_sync(chunk).embeddings
            for text, vector in zip(chunk,vectors):
                assert len(vector) == 1536
                cache[cache_key(role,text)] = list(vector)
            CACHE.write_text(json.dumps(cache),encoding="utf-8")
        return {t:cache[cache_key(role,t)] for t in unique}
    httpx.AsyncClient.send = embeddings_only
    try:
        docs = embed("document",summaries.values())
        queries = embed("query",texts)
    finally: httpx.AsyncClient.send = send
    scores = {text:{k:cosine(vector,docs[summary]) for k,summary in summaries.items()} for text,vector in queries.items()}
    print(f"Ready: {len(bank)} entities, {len(cases)} activation cases, {len(queries)} unique messages",flush=True)
    db = PostgresProvider(os.environ["POSTGRES_URL"])
    observer = psycopg.connect(os.environ["POSTGRES_URL"],autocommit=True,options="-c default_transaction_read_only=on")
    redis_client = redis.Redis.from_url(os.environ["REDIS_URL"])
    before = state(observer,redis_client)
    tenant = "activation-evaluation-"+uuid4().hex
    factors, pg_checks, max_difference = {},0,0.0
    try:
        try:
            with db.transaction():
                updates=[]
                for t,u in [(tenant,"equal"),(tenant,"weighted"),(tenant,"other"),(tenant+"-other","equal")]:
                    for k,payload in bank.items():
                        updates.append(dict(mode="INSERT",tenant_id=t,user_id=u,
                            **{name:payload[name] for name in ("entity_id","entity_name","entity_domain","entity_type")},
                            payload=deepcopy(payload),embedding=docs[summaries[k]]))
                assert db.commit_graph_updates(updates)
                db.conn.execute("UPDATE tomte_memory_engine SET last_mentioned_at=NOW()-interval '24 hours' "
                    "WHERE tenant_id=%s AND user_id='weighted' AND entity_id='tesla'",(tenant,))
                db.conn.execute("UPDATE tomte_memory_engine SET mention_count=5 WHERE tenant_id=%s AND user_id='weighted' AND entity_id='honda'",(tenant,))
                db.conn.execute("UPDATE tomte_memory_engine SET mention_count=8 WHERE tenant_id=%s AND user_id='weighted' AND entity_id='python'",(tenant,))
                for user in ("equal","weighted"):
                    factors[user]={r["entity_id"]:float(r["factor"]) for r in db.conn.execute(
                        "SELECT entity_id, mention_count*exp(-0.0001*extract(epoch FROM (now()-last_mentioned_at))) factor "
                        "FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s",(tenant,user)).fetchall()}
                    for text,vector in queries.items():
                        found=db.semantic_retrieval(tenant,user,vector)
                        assert all(r["tenant_id"]==tenant and r["user_id"]==user for r in found)
                        expected=sorted(bank,key=lambda k:(-scores[text][k]*factors[user][k],k))[:5]
                        assert [r["entity_id"] for r in found]==expected
                        rows=db.conn.execute("SELECT entity_id, 1-(embedding <=> %s::vector) cosine FROM tomte_memory_engine "
                            "WHERE tenant_id=%s AND user_id=%s",(vector,tenant,user)).fetchall()
                        max_difference=max(max_difference,max(abs(scores[text][r["entity_id"]]-float(r["cosine"])) for r in rows))
                        pg_checks+=1
                for t,u in [(tenant,"other"),(tenant+"-other","equal")]:
                    found=db.semantic_retrieval(t,u,queries[cases[0]["text"]])
                    assert len(found)==5 and all(r["tenant_id"]==t and r["user_id"]==u for r in found)
                    pg_checks+=1
                assert db.semantic_retrieval(tenant,"absent",queries[cases[0]["text"]])==[]
                pg_checks+=1
                raise RollbackEvaluation()
        except RollbackEvaluation: pass
        after=state(observer,redis_client)
        assert before==after,"Persistent state changed"
    finally:
        db.close();observer.close();redis_client.close()
    comparisons=[score_rule(rule,cases,bank,scores,factors["equal"]) for rule in rules()]
    for item in comparisons:
        m=item["overall"]
        item["pareto"]=not any(other["overall"]["micro_precision"]>=m["micro_precision"] and
            other["overall"]["micro_recall"]>=m["micro_recall"] and
            (other["overall"]["micro_precision"]>m["micro_precision"] or other["overall"]["micro_recall"]>m["micro_recall"])
            for other in comparisons)
        item["session"]=session_simulation(item["rule"],conversations,bank,scores,factors["equal"])
    complexity={"topk":0,"floor":1,"floor_topk":1,"lexical":2,"margin":3,"weighted":4}
    # Recall-first calibration loss, fixed before inspecting holdout metrics.
    selected=min(comparisons,key=lambda c:(3*c["calibration"]["missed_per_turn"]+c["calibration"]["false_per_turn"],
        complexity[c["rule"]["family"]],c["rule"]["cap"]))
    relevant=[scores[c["text"]][k] for c in cases for k in c["gold"]]
    irrelevant=[v for c in cases for k,v in scores[c["text"]].items() if k not in c["gold"]]
    rd,idist=distribution(relevant),distribution(irrelevant)
    sorted_wrong=sorted(irrelevant)
    auc=sum(bisect_left(sorted_wrong,v)+.5*(bisect_right(sorted_wrong,v)-bisect_left(sorted_wrong,v))
            for v in relevant)/(len(relevant)*len(irrelevant))
    weighted=score_rule(rules()[0],cases,bank,scores,factors["weighted"])
    raw_top5=next(c for c in comparisons if c["rule"]["name"]=="cosine_top5")
    lost=[dict(case_id=c["id"],text=c["text"],lost=sorted((set(c["gold"])&set(a["activated"]))-set(b["activated"])))
        for c,a,b in zip(cases,raw_top5["records"],weighted["records"])
        if (set(c["gold"])&set(a["activated"]))-set(b["activated"])]
    links=[{"python","tomte","kids_math"},{"terry","basketball"},{"bryan","swimming"},
           {"alex","current_employer","job_role"}]
    for item in comparisons:
        low,high=0,0
        for r in item["records"]:
            for k in set(r["activated"])-set(r["gold"]):
                related=any(family[k]==family[g] or any({k,g}<=link for link in links) for g in r["gold"])
                if related:low+=1
                else:high+=1
        item["false_severity"]=dict(low_impact_related=low,higher_impact=high)
    report=dict(entity_count=len(bank),case_count=len(cases),category_counts=dict(Counter(c["category"] for c in cases)),
        split_counts=dict(Counter(c["split"] for c in cases)),families={g:[k for k in bank if family[k]==g] for g in sorted(set(family.values()))},
        entities=bank,cases=cases,summaries=summaries,scores=scores,comparisons=comparisons,
        selected_on_calibration=selected["rule"],selection_loss="3 * missed/turn + false/turn; ties prefer simpler rule and lower cap",
        split_limitations="Stratified case split shares identities, duplicate generic texts and authored templates; not an independent external benchmark",
        cosine_distributions=dict(relevant=rd,irrelevant=idist,pooled_auc=auc,
            overlap_range=[max(rd["min"],idist["min"]),min(rd["max"],idist["max"])],
            relevant_below_irrelevant_p90=statistics.mean(v<=idist["p90"] for v in relevant),
            irrelevant_above_relevant_p10=statistics.mean(v>=rd["p10"] for v in irrelevant)),
        weighted_stress=weighted,weighting_factors=factors,weighting_lost_cases=lost,
        margin_definition="score minus rank-6 raw score; observable background proxy, no oracle labels",
        lexical_definition="casefold + exact normalized token/phrase match of full name or stored alias; full scoped catalog searched; no negation parser",
        session_pollution_definition="retained identities NEVER relevant anywhere in that conversation; valid earlier topics are not pollution",
        model=settings.openai.embed_model,dimension=1536,api_usage=usage,requests=len(usage),
        embedded_texts=sum(r["texts"] for r in usage),input_tokens=sum(r["usage"].get("prompt_tokens",0) for r in usage),
        reused=reused,pg_checks=pg_checks,pg_max_cosine_difference=max_difference,synthetic_rows=len(bank)*4,
        before=before,after=after,cleanup="all synthetic PostgreSQL writes rolled back; Redis never written")
    OUTPUT.write_text(json.dumps(report,indent=2),encoding="utf-8")
    fields=["policy","precision","recall","f1","exact_set","false_per_turn","missed_per_turn","false_turns","missed_turns","activated_per_turn","pareto"]
    csv_path=ROOT/"log"/"entity_activation_comparison.csv"
    table_rows=[]
    with csv_path.open("w",encoding="utf-8",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        for c in comparisons:
            m=c["overall"]
            row=dict(policy=c["rule"]["name"],precision=m["micro_precision"],recall=m["micro_recall"],f1=m["f1"],
                exact_set=m["exact_set_accuracy"],false_per_turn=m["false_per_turn"],missed_per_turn=m["missed_per_turn"],
                false_turns=m["turns_with_false"],missed_turns=m["turns_missing"],activated_per_turn=m["activated_per_turn"],pareto=c["pareto"])
            writer.writerow(row);table_rows.append(row)
    # Self-contained comparison table for the repository evaluation artifact.
    header="".join(f"<th>{html.escape(k)}</th>" for k in fields)
    rows="".join("<tr>"+"".join(f"<td>{html.escape(format(row[k],'.3f') if isinstance(row[k],float) else str(row[k]))}</td>" for k in fields)+"</tr>" for row in table_rows)
    (ROOT/"log"/"entity_activation_comparison.html").write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Entity activation evaluation</title>'
        '<style>body{font:15px system-ui;margin:28px;color:#222}table{border-collapse:collapse}td,th{padding:7px 12px;border-bottom:1px solid #ddd;text-align:right}td:first-child,th:first-child{text-align:left}</style>'
        f'<h1>Entity activation evaluation</h1><p>{len(cases)} synthetic cases, {len(bank)} identities. '
        'Pareto = no measured policy with both greater/equal precision and recall and one strictly greater. '
        'Thresholds are evaluation settings only. Full category/holdout/session results are in the JSON report.</p>'
        f'<table><thead><tr>{header}</tr></thead><tbody>{rows}</tbody></table></html>',encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("entity_count","case_count","category_counts","split_counts","requests",
        "embedded_texts","input_tokens","pg_checks","pg_max_cosine_difference","selected_on_calibration","cleanup")},indent=2))


if __name__=="__main__":
    try:main()
    except Exception as exc:
        import traceback
        print(f"Evaluation failed: {type(exc).__name__}; exception contents withheld")
        print([(Path(f.filename).name,f.lineno,f.name) for f in traceback.extract_tb(exc.__traceback__)])
        raise SystemExit(1)
