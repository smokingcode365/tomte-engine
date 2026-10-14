"""Embedding-only memory retrieval evaluation; run from the repository root.

Uses configured real embeddings and local bootstrapped PostgreSQL. No model/chat
calls, daemon, dictionary, or Redis writes. Synthetic persistence is rolled back.
Results and reusable embedding cache are written only to ignored log/.
The dataset is deliberately synthetic, not a benchmark of general user traffic.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
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

from tomte.engine.core.config import settings
from tomte.engine.core.schema import Entity, Fact
from tomte.engine.llm.embedder import get_entity_summary
from tomte.engine.providers.openai_provider import OpenAILLMProvider
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.utils.security import SecurityGuard

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "log" / "memory_retrieval_evaluation.json"
CACHE = ROOT / "log" / "memory_retrieval_embeddings.json"
USAGE = ROOT / "log" / "memory_retrieval_api_usage.json"
TABLES = ["tomte_user_registry", "tomte_memory_engine", "tomte_keyword_audit_log",
          "tomte_semantic_anchors", "tomte_keyword_registry", "tomte_processed_consolidations"]
STAMP = "2026-01-01T00:00:00Z"


def entity(key, name, domain, kind, aliases, pairs):
    facts = [Fact(attribute_name=k, value=v, confidence=1.0,
                  source_chat_id="synthetic-retrieval", last_mentioned_at=STAMP)
             for k, v in pairs]
    return Entity(entity_id=key, entity_name=name, entity_domain=domain,
                  entity_type=kind, aliases=aliases, facts=facts).model_dump(mode="json")


def dataset():
    assets, activities, relationships, identity, logistics = (
        "assets_and_possessions", "activities_and_skills", "relationships",
        "identity_and_demographics", "events_and_logistics")
    specs = [
        ("tesla", "Tesla Model Y", assets, "vehicle", ["Little Dog"],
         [("paint_color", "blue"), ("ownership_status", "owned"), ("purchase_year", "2024"),
          ("charging_location", "home"), ("fuel_type", "electric")],
         "vehicle", "What color is my car?", "Which vehicle do I charge at home?", "Which car did I buy in 2024?"),
        ("honda", "Honda Civic", assets, "vehicle", ["Daily Runner"],
         [("paint_color", "blue"), ("ownership_status", "owned"), ("purchase_year", "2020"),
          ("fuel_type", "gasoline"), ("primary_use", "commuting")],
         "vehicle", "Which car did I buy in 2020?", "Which car uses gasoline for my commute?", "Which car have I owned since 2020?"),
        ("toyota", "Toyota RAV4", assets, "vehicle", ["Mountain Wagon"],
         [("paint_color", "white"), ("ownership_status", "sold"), ("purchase_year", "2016"),
          ("primary_use", "mountain camping"), ("sale_year", "2025")],
         "vehicle", "Which car was white?", "Which vehicle did I use for mountain camping?", "Which car did I sell?"),
        ("current_employer", "Northstar Labs", activities, "employer", ["Northstar"],
         [("employment_status", "current"), ("start_year", "2024"), ("industry", "robotics"),
          ("location", "Vancouver"), ("work_arrangement", "hybrid")],
         "employer", "Which employer did I join in 2024?", "Where do I work on robotics?", "Who is my current employer?"),
        ("previous_employer", "Cedar Systems", activities, "employer", ["Cedar"],
         [("employment_status", "previous"), ("start_year", "2020"), ("end_year", "2023"),
          ("industry", "banking software"), ("location", "Toronto")],
         "employer", "Which employer did I leave in 2023?", "Where did I build banking software?", "Who was my previous employer?"),
        ("job_role", "Engineering Manager", activities, "job_role", ["Team Lead Role"],
         [("role_status", "current"), ("team_size", "eight engineers"), ("responsibility", "hiring"),
          ("start_year", "2024"), ("specialty", "robotics")],
         "job_role", "Which role involves hiring eight engineers?", "What is my leadership responsibility at work?", "What is my current job role?"),
        ("terry", "Terry", relationships, "person", ["T-Man"],
         [("relationship", "son"), ("age", "15"), ("hobby", "basketball"),
          ("school", "Oak Secondary"), ("activity_status", "plays on school team")],
         "sons", "Which son plays basketball?", "Who has practice on the basketball court?", "Which child is on the school team?"),
        ("bryan", "Bryan", relationships, "person", ["Bri"],
         [("relationship", "son"), ("age", "10"), ("hobby", "swimming"),
          ("school", "Maple Elementary"), ("activity_status", "takes pool lessons")],
         "sons", "Which son is ten years old?", "Who takes swimming lessons at the pool?", "Which child currently takes pool lessons?"),
        ("alex", "Alex", relationships, "person", ["Office Buddy"],
         [("relationship", "coworker"), ("specialty", "databases"), ("hobby", "cycling"),
          ("workplace", "Northstar Labs"), ("employment_status", "current coworker")],
         "coworker", "Which coworker specializes in databases?", "Who helps me with database work?", "Who is my current coworker?"),
        ("current_home", "Vancouver Apartment", identity, "home", ["Harbor Place"],
         [("residence_status", "current"), ("move_year", "2024"), ("city", "Vancouver"),
          ("rooms", "two bedrooms"), ("location", "near harbor")],
         "home", "Which home has two bedrooms?", "Where do I live near the harbor?", "Where is my current home?"),
        ("previous_home", "Toronto Condo", identity, "home", ["Old Nest"],
         [("residence_status", "previous"), ("move_out_year", "2023"), ("city", "Toronto"),
          ("rooms", "one bedroom"), ("location", "downtown")],
         "home", "Which home had one bedroom?", "Where did I live downtown in Toronto?", "Where was my previous home?"),
        ("vacation", "Lisbon Holiday", logistics, "vacation", ["Portugal Getaway"],
         [("destination", "Lisbon"), ("country", "Portugal"), ("travel_status", "planned"),
          ("travel_month", "November"), ("activity", "visit museums")],
         "vacation", "Where am I traveling in November?", "Where is my planned Portuguese museum trip?", "What vacation is planned?"),
        ("python", "Python", activities, "programming_skill", ["Favorite Language"],
         [("proficiency", "intermediate"), ("use_case", "data analysis"), ("learning_status", "learning"),
          ("library", "pandas"), ("practice", "weekend scripts")],
         "skill", "Which language do I use for pandas?", "What am I learning for data analysis scripts?", "Which scripting language am I learning?"),
        ("java", "Java", activities, "programming_skill", ["Backend Language"],
         [("proficiency", "advanced"), ("use_case", "enterprise backend"), ("learning_status", "experienced"),
          ("framework", "Spring Boot"), ("practice", "work services")],
         "skill", "Which language do I use for Spring Boot?", "What do I use to build enterprise backend services?", "Which backend language am I experienced in?"),
        ("tomte", "Tomte", activities, "software_project", ["Memory Engine"],
         [("purpose", "AI memory system"), ("storage", "PostgreSQL and Redis"), ("project_status", "active"),
          ("language", "Python"), ("feature", "entity retrieval")],
         "project", "Which project uses PostgreSQL and Redis?", "What project remembers entities for an AI assistant?", "Which memory project is active?"),
        ("main_aquarium", "Main Aquarium", assets, "aquarium", ["Living Room Tank"],
         [("volume", "200 liters"), ("location", "living room"), ("fish", "angelfish"),
          ("water_type", "freshwater"), ("condition", "algae present")],
         "aquarium", "Which aquarium holds 200 liters?", "Where do my angelfish live?", "Which aquarium has algae?"),
        ("second_aquarium", "Second Aquarium", assets, "aquarium", ["Study Tank"],
         [("volume", "60 liters"), ("location", "study"), ("fish", "shrimp"),
          ("water_type", "freshwater"), ("condition", "clean")],
         "aquarium", "Which aquarium holds 60 liters?", "Where do I keep the shrimp?", "Which aquarium is clean?"),
    ]
    bank = {s[0]: entity(*s[:6]) for s in specs}
    groups = {"vehicle": ["tesla", "honda", "toyota"], "employer": ["current_employer", "previous_employer"],
              "sons": ["terry", "bryan"], "home": ["current_home", "previous_home"],
              "skill": ["python", "java"], "aquarium": ["main_aquarium", "second_aquarium"]}
    scenarios = {"baseline": bank}
    cases = []

    def add(scenario, style, text, relevant, **labels):
        cases.append(dict(scenario=scenario, style=style, text=text, relevant=relevant, **labels))

    for key, name, domain, kind, aliases, facts, group, factq, indirectq, stateq in specs:
        for style, text, relevant in [
            ("name", f"What do you remember about {name}?", [key]),
            ("alias", f"What do you remember about {aliases[0]}?", [key]),
            ("type", f"What do you remember about my {group.replace('_', ' ')}?", groups.get(group, [key])),
            ("fact", factq, groups["vehicle"] if key == "tesla" else [key]),
            ("indirect", indirectq, [key]), ("state", stateq, [key]),
        ]:
            add("baseline", style, text, relevant, family=group)

    more = [("charging_location", "home"), ("nickname", "Little Dog"), ("mileage", "24000 kilometers"),
            ("tires", "all season"), ("insurance", "comprehensive"), ("software_version", "2026.4"),
            ("service_date", "March 2026"), ("parking_location", "underground garage"),
            ("wheel_size", "19 inches"), ("battery_health", "96 percent"),
            ("registration_expiry", "January 2027"), ("roof_rack", "installed"),
            ("repair_history", "replaced windshield"), ("interior_color", "black"),
            ("seat_material", "synthetic leather"), ("charging_schedule", "overnight"),
            ("odometer_units", "kilometers"), ("warranty", "active"), ("sound_system", "premium"),
            ("navigation", "built in"), ("floor_mats", "rubber"), ("dash_camera", "installed"),
            ("cargo_liner", "waterproof"), ("child_seat", "rear passenger seat"),
            ("emergency_kit", "trunk"), ("maintenance_interval", "annual"),
            ("favorite_playlist", "classical music"), ("bumper_sticker", "basketball team"),
            ("dashboard_book", "Python programming guide"), ("picnic_blanket", "blue plaid"),
            ("travel_souvenir", "Lisbon keychain"), ("air_freshener", "pine"),
            ("sunshade", "silver"), ("cleaning_schedule", "Saturdays"),
            ("phone_mount", "magnetic"), ("cup_holder", "two cups"), ("tool_kit", "compact")]
    forty = bank["tesla"]["facts"][:3] + [Fact(attribute_name=k, value=v, confidence=1,
        source_chat_id="synthetic-retrieval", last_mentioned_at=STAMP).model_dump(mode="json") for k, v in more]
    assert len(forty) == 40
    size_queries = ["What color is my car?", "What color is my Tesla?", "Which car did I buy in 2024?",
                    "Which vehicle do I charge at home?", "Tell me about Little Dog.", "Tell me about my Tesla Model Y."]
    for size in (0, 1, 3, 5, 10, 20, 40):
        scenario = f"size_{size}"
        scenarios[scenario] = deepcopy(bank)
        scenarios[scenario]["tesla"]["aliases"] = []  # fixed for every size; nickname becomes a fact at size 5
        scenarios[scenario]["tesla"]["facts"] = deepcopy(forty[:size])
        for text in size_queries:
            add(scenario, "size", text, groups["vehicle"] if text == "What color is my car?" else ["tesla"], size=size,
                target="tesla", family="vehicle", diagnostic_single_target=True)
    for position in ("beginning", "middle", "end"):
        scenario = f"order_{position}"
        scenarios[scenario] = deepcopy(scenarios["size_40"])
        facts = deepcopy(forty[1:])
        facts.insert({"beginning": 0, "middle": 20, "end": 39}[position], deepcopy(forty[0]))
        scenarios[scenario]["tesla"]["facts"] = facts
        add(scenario, "order", "What color is my car?", groups["vehicle"], position=position,
            target="tesla", diagnostic_single_target=True)
    topics = [
        ("tesla", ["What color is my Tesla?", "Terry has basketball tonight.", "I'm learning Python.",
                   "My main aquarium has algae.", "We may travel to Lisbon next month."],
         ["tesla", "terry", "python", "main_aquarium", "vacation"]),
        ("python", ["Which Python library do I use?", "I charge my Tesla at home.", "Terry has basketball tonight.",
                    "My main aquarium has algae.", "We may travel to Lisbon next month."],
         ["python", "tesla", "terry", "main_aquarium", "vacation"]),
        ("terry", ["Which school does Terry attend?", "I charge my Tesla at home.", "I'm learning Python.",
                   "My main aquarium has algae.", "We may travel to Lisbon next month."],
         ["terry", "tesla", "python", "main_aquarium", "vacation"]),
    ]
    for target, sentences, ids in topics:
        for n in (1, 2, 3, 5):
            add("baseline", "topics", " ".join(sentences[:n]), ids[:n], target=target, topics=n)
    for label, text in [
        ("short", "My Tesla?"), ("medium", "What do you remember about my Tesla?"),
        ("long", "I would like to revisit what I previously told you about my Tesla Model Y. "
         "Please bring back the details associated with that vehicle so I can review them carefully. "
         "I am referring to the same Tesla, and I am not asking about a different vehicle or a new topic.")]:
        add("baseline", "length", text, ["tesla"], length=label)
    confusable = [
        ("vehicle", "Which car did I buy in 2024?", ["tesla"]),
        ("vehicle", "Which blue car do I own?", ["tesla", "honda"]),
        ("vehicle", "What do you remember about my older car?", ["honda", "toyota"]),
        ("vehicle", "Which car did I sell?", ["toyota"]),
        ("employer", "Who is my current employer?", ["current_employer"]),
        ("employer", "What do you remember about my employer?", ["current_employer", "previous_employer"]),
        ("sons", "What do you remember about my son?", ["terry", "bryan"]),
        ("sons", "What do you remember about Bri?", ["bryan"]),
        ("home", "What do you remember about my home?", ["current_home", "previous_home"]),
        ("home", "What do you remember about Old Nest?", ["previous_home"]),
        ("home", "Where did I live before Vancouver?", ["previous_home"]),
        ("vehicle", "What do you remember about Little Dog?", ["tesla"]),
    ]
    for family, text, ids in confusable:
        add("baseline", "confusable", text, ids, family=family)
    scenarios["history"] = deepcopy(bank)
    historical = scenarios["history"]["tesla"]
    historical["facts"][1]["value"] = "sold"
    for key, value in [("paint_color", "red"), ("ownership_status", "owned")]:
        fact = Fact(attribute_name=key, value=value, status="past", confidence=1,
                    source_chat_id="synthetic-past", last_mentioned_at=STAMP).model_dump(mode="json")
        historical["facts"].append(fact)
    historical["facts"].append(Fact(attribute_name="temporary_condition", value="muddy today",
        importance=1, confidence=1, source_chat_id="synthetic", last_mentioned_at=STAMP).model_dump(mode="json"))
    for text in ("What car did I sell?", "What color is my old Tesla?", "What do you remember about the Tesla?"):
        add("history", "history", text, ["tesla", "toyota"] if text == "What car did I sell?" else ["tesla"],
            target="tesla")
    for scenario in ("weight_mentions", "weight_age"):
        scenarios[scenario] = deepcopy(bank)
        for text in ("What do you remember about Tesla Model Y?", "Which vehicle do I charge at home?",
                     "Tell me about Little Dog."):
            add(scenario, "weight", text, ["tesla"])
    return scenarios, cases


def cosine(a, b):
    return sum(x*y for x, y in zip(a, b)) / math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))


def state(observer, cache):
    tables = {}
    for table in TABLES:
        # Hash on server; no private payloads leave PostgreSQL.
        count, digest = observer.execute(f"SELECT count(*), md5(coalesce(string_agg(h, '' ORDER BY h), '')) "
            f"FROM (SELECT md5(row_to_json(t)::text) AS h FROM {table} t) s").fetchone()
        tables[table] = dict(count=count, digest=digest)
    keys = sorted(cache.scan_iter())
    return dict(postgres=tables, redis=dict(count=cache.dbsize(),
        key_digest=hashlib.sha256(b'\0'.join(keys)).hexdigest()))


def metrics(rows):
    if not rows:
        return {}
    single = [r for r in rows if len(r["relevant"]) == 1]
    ambiguous = [r for r in rows if len(r["relevant"]) > 1]
    gold = [r["cosines"][k] for r in rows for k in r["relevant"]]
    wrong = [max(v for k, v in r["cosines"].items() if k not in r["relevant"]) for r in rows]
    return dict(cases=len(rows), unambiguous=len(single), ambiguous=len(ambiguous),
        top1_accuracy=statistics.mean(r["ranking"][0] in r["relevant"] for r in single) if single else None,
        top3_recall=statistics.mean(len(set(r["ranking"][:3]) & set(r["relevant"]))/len(r["relevant"]) for r in rows),
        top5_recall=statistics.mean(len(set(r["ranking"][:5]) & set(r["relevant"]))/len(r["relevant"]) for r in rows),
        mean_correct_cosine=statistics.mean(gold), min_correct_cosine=min(gold),
        mean_best_wrong_cosine=statistics.mean(wrong),
        mean_margin=statistics.mean(max(r["cosines"][k] for k in r["relevant"])-w for r, w in zip(rows, wrong)),
        diagnostic_cosine_075_recall=statistics.mean(v >= .75 for v in gold),
        ambiguous_all_in_top5=statistics.mean(set(r["relevant"]) <= set(r["ranking"][:5]) for r in ambiguous) if ambiguous else None)


class RollbackEvaluation(Exception):
    pass


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run from tomte-engine")
    load_dotenv(ROOT / ".env", override=False)
    for name in ("OPENAI_API_KEY", "POSTGRES_URL", "REDIS_URL"):
        if not os.environ.get(name):
            raise RuntimeError(f"Missing {name}")
    for name in ("POSTGRES_URL", "REDIS_URL"):
        if urlsplit(os.environ[name]).hostname not in ("localhost", "127.0.0.1", "::1"):
            raise RuntimeError("Only local infrastructure is allowed")
    if settings.enable_pii_scrubbing:
        raise RuntimeError("Evaluate configured PII handling separately before this synthetic run")
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    OUTPUT.parent.mkdir(exist_ok=True)
    scenarios, cases = dataset()
    summaries = {s: {k: get_entity_summary(e) for k, e in bank.items()} for s, bank in scenarios.items()}
    guard = SecurityGuard()
    for case in cases:
        case["query"] = guard.sanitize_input(case["text"])
        assert guard.check_for_injection(case["query"])
        assert case["text"] == case["query"]
    cache_data = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    usage = json.loads(USAGE.read_text()) if USAGE.exists() else []
    original_send = httpx.AsyncClient.send

    async def embeddings_only(client, request, *args, **kwargs):
        if not request.url.path.endswith("/embeddings"):
            raise RuntimeError("Non-embedding API request blocked")
        body = json.loads(request.content)
        response = await original_send(client, request, *args, **kwargs)
        await response.aread()
        usage.append(dict(model=body.get("model"), texts=len(body["input"]) if isinstance(body["input"], list) else 1,
                          status=response.status_code, usage=response.json().get("usage", {})))
        USAGE.write_text(json.dumps(usage, indent=2), encoding="utf-8")
        return response

    provider = OpenAILLMProvider(os.environ["OPENAI_API_KEY"])

    def cache_key(role, text):
        return hashlib.sha256(json.dumps([settings.openai.embed_model, settings.openai.vector_dimension,
                                          role, text]).encode()).hexdigest()

    def embed(role, texts):
        unique = list(dict.fromkeys(texts))
        pending = [t for t in unique if cache_key(role, t) not in cache_data]
        for start in range(0, len(pending), 100):
            chunk = pending[start:start+100]
            # Documents use normal consolidation path. Queries batch the same
            # embed_query_sync method used by provider.embed_text, without prefixes.
            vectors = provider.embed_batch(chunk) if role == "document" else provider.embedder.embed_query_sync(chunk).embeddings
            for text, vector in zip(chunk, vectors):
                assert len(vector) == settings.openai.vector_dimension == 1536
                cache_data[cache_key(role, text)] = list(vector)
            CACHE.write_text(json.dumps(cache_data), encoding="utf-8")
        return {t: cache_data[cache_key(role, t)] for t in unique}

    httpx.AsyncClient.send = embeddings_only
    try:
        docs = embed("document", [t for bank in summaries.values() for t in bank.values()])
        queries = embed("query", [c["query"] for c in cases])
        parity, parity_cosines = [], []
        for style in ("name", "alias", "type", "fact", "indirect", "state"):
            text = next(c["query"] for c in cases if c["style"] == style)
            single_key = cache_key("single-query", text)
            if single_key not in cache_data:
                cache_data[single_key] = provider.embed_text(text)
                CACHE.write_text(json.dumps(cache_data), encoding="utf-8")
            actual = cache_data[single_key]
            parity.append(max(abs(a-b) for a, b in zip(actual, queries[text])))
            parity_cosines.append(cosine(actual, queries[text]))
    finally:
        httpx.AsyncClient.send = original_send
    print(f"Embedded/cached {len(docs)} distinct summaries and {len(queries)} distinct queries", flush=True)

    db = PostgresProvider(os.environ["POSTGRES_URL"])
    observer = psycopg.connect(os.environ["POSTGRES_URL"], autocommit=True,
                              options="-c default_transaction_read_only=on")
    redis_client = redis.Redis.from_url(os.environ["REDIS_URL"])
    before = state(observer, redis_client)
    tenant = "memory-retrieval-evaluation-" + uuid4().hex
    results, max_difference, isolation_checks = [], 0.0, 0
    try:
        try:
            with db.transaction():
                updates = []
                for scenario, bank in scenarios.items():
                    db.upsert_user_activity(tenant, scenario)
                    for key, payload in bank.items():
                        updates.append(dict(mode="INSERT", tenant_id=tenant, user_id=scenario,
                            **{k: payload[k] for k in ("entity_id", "entity_name", "entity_domain", "entity_type")},
                            payload=deepcopy(payload), embedding=docs[summaries[scenario][key]]))
                # Exact duplicates under independent tenant and user namespaces.
                for other_tenant, other_user in [(tenant+"-other", "baseline"), (tenant, "other-user")]:
                    payload = deepcopy(scenarios["baseline"]["tesla"])
                    updates.append(dict(mode="INSERT", tenant_id=other_tenant, user_id=other_user,
                        **{k: payload[k] for k in ("entity_id", "entity_name", "entity_domain", "entity_type")},
                        payload=payload, embedding=docs[summaries["baseline"]["tesla"]]))
                assert db.commit_graph_updates(updates)
                # All baseline rows get identical NOW() and mention_count=1.
                db.conn.execute("UPDATE tomte_memory_engine SET mention_count=5 WHERE tenant_id=%s "
                    "AND user_id='weight_mentions' AND entity_id='honda'", (tenant,))
                db.conn.execute("UPDATE tomte_memory_engine SET last_mentioned_at=NOW()-interval '24 hours' "
                    "WHERE tenant_id=%s AND user_id='weight_age' AND entity_id='tesla'", (tenant,))
                for case in cases:
                    scenario, vector = case["scenario"], queries[case["query"]]
                    found = db.semantic_retrieval(tenant, scenario, vector)
                    assert all(r["tenant_id"] == tenant and r["user_id"] == scenario for r in found)
                    isolation_checks += 1
                    sql = db.conn.execute("SELECT entity_id, 1-(embedding <=> %s::vector) AS cosine, "
                        "mention_count*exp(-0.0001*extract(epoch FROM (now()-last_mentioned_at))) AS factor "
                        "FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s", (vector, tenant, scenario)).fetchall()
                    scores = {r["entity_id"]: float(r["cosine"]) for r in sql}
                    weighted = {r["entity_id"]: float(r["cosine"])*float(r["factor"]) for r in sql}
                    ranking = sorted(scores, key=lambda k: weighted[k], reverse=True)
                    assert [r["entity_id"] for r in found] == ranking[:settings.semantic_retrieval_limit]
                    for key, summary in summaries[scenario].items():
                        max_difference = max(max_difference, abs(cosine(vector, docs[summary])-scores[key]))
                    target = case.get("target")
                    results.append(dict(**case, cosines=scores, weighted_scores=weighted, ranking=ranking,
                        top5=[r["entity_id"] for r in found],
                        target_rank=ranking.index(target)+1 if target else None))
                for t, u, expected in [(tenant+"-other", "baseline", 1), (tenant, "other-user", 1),
                                        (tenant+"-absent", "baseline", 0), (tenant, "absent-user", 0)]:
                    rows = db.semantic_retrieval(t, u, queries[cases[0]["query"]])
                    assert len(rows) == expected
                    assert all(r["tenant_id"] == t and r["user_id"] == u for r in rows)
                    isolation_checks += 1
                raise RollbackEvaluation()
        except RollbackEvaluation:
            pass
        after = state(observer, redis_client)
        assert before == after, "Before/after database or Redis state differs"
    finally:
        db.close()
        observer.close()
        redis_client.close()
    grouped = defaultdict(list)
    for row in results:
        grouped[f"style:{row['style']}"] .append(row)
        for label in ("size", "topics", "family", "scenario"):
            if label in row:
                grouped[f"{label}:{row[label]}"].append(row)
    report = dict(model=settings.openai.embed_model, dimension=settings.openai.vector_dimension,
        semantic_limit=settings.semantic_retrieval_limit, query_independent_limit=settings.query_entities_limit,
        production_threshold=None, diagnostic_threshold=.75,
        scenarios=len(scenarios), entities_per_scenario=len(scenarios["baseline"]),
        synthetic_rows=len(scenarios)*len(scenarios["baseline"])+2, cases=len(results),
        overall=metrics(results), baseline=metrics([r for r in results if r["scenario"] == "baseline" and
            r["style"] in ("name", "alias", "type", "fact", "indirect", "state")]),
        groups={k: metrics(v) for k, v in grouped.items()}, results=results,
        summaries=summaries, entities=scenarios,
        summary_lengths=dict(unique=len(docs), mean_chars=statistics.mean(map(len, docs)), max_chars=max(map(len, docs)),
                             estimated_tokens_method="characters / 4 (rough estimate)",
                             max_estimated_tokens=max(map(len, docs))/4),
        query_lengths=dict(unique=len(queries), max_chars=max(map(len, queries)), truncated=0),
        api_usage=usage, total_requests=len(usage), total_texts=sum(r["texts"] for r in usage),
        recorded_input_tokens=sum(r["usage"].get("prompt_tokens", 0) for r in usage),
        requests_without_token_record=sum("prompt_tokens" not in r["usage"] for r in usage),
        query_parity_max_difference=max(parity), query_parity_min_cosine=min(parity_cosines),
        pgvector_cosine_max_difference=max_difference,
        isolation_checks=isolation_checks, before=before, after=after, cleanup="whole synthetic transaction rolled back")
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("cases", "baseline", "total_requests", "total_texts", "total_input_tokens",
                    "pgvector_cosine_max_difference", "isolation_checks", "cleanup") if k in report}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Never print SDK exception bodies, connection strings, or credentials.
        print(f"Evaluation failed: {type(exc).__name__}; no exception content displayed")
        import traceback
        print([(Path(f.filename).name, f.lineno, f.name) for f in traceback.extract_tb(exc.__traceback__)])
        raise SystemExit(1)
