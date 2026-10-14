"""Evaluation-only session entity policies; no production/Redis state changes.

Run from tomte-engine. Uses the prior memory benchmark fixtures/cache, real
configured embeddings, and rollback-only local PostgreSQL. No chat model calls.
All MRU state contains IDs only and is simulated in process. Results are ignored.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import re
import statistics
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import load_dotenv
import httpx
import psycopg
import redis

from evaluate_memory_retrieval import dataset, entity, cosine, state, RollbackEvaluation
from tomte.engine.core.config import settings
from tomte.engine.llm.embedder import get_entity_summary
from tomte.engine.providers.openai_provider import OpenAILLMProvider
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.utils.security import SecurityGuard
from tomte.engine.utils.keys import generate_common_key, validate_id

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "log" / "session_entity_evaluation.json"
CACHE = ROOT / "log" / "memory_retrieval_embeddings.json"
USAGE = ROOT / "log" / "session_entity_api_usage.json"


def fixtures():
    bank = deepcopy(dataset()[0]["baseline"])
    for key, alias in [("tesla", "Tesla"), ("honda", "Honda"), ("toyota", "Toyota")]:
        bank[key]["aliases"].append(alias)
    bank["dog"] = entity("dog", "Milo", "relationships", "pet_dog", ["my dog"],
        [("species", "dog"), ("breed", "beagle"), ("condition", "scratching"), ("age", "four years")])
    bank["cat"] = entity("cat", "Miso", "relationships", "pet_cat", ["my cat"],
        [("species", "cat"), ("breed", "tabby"), ("condition", "healthy"), ("age", "two years")])
    conversations = []

    def user(text, gold, kind="ordinary"):
        return dict(role="user", text=text, gold=gold, kind=kind)

    def assistant(text):
        return dict(role="assistant", text=text, gold=[], kind="assistant")

    def add(name, turns, split="holdout"):
        conversations.append(dict(name=name, turns=turns, split=split))

    add("explicit", [user("My Tesla needs service.", ["tesla"], "discovery")], "calibration")
    add("short_followup", [user("My Tesla needs service.", ["tesla"], "discovery"),
        assistant("What symptoms have you noticed?"), user("It makes noise when I accelerate.", ["tesla"], "continuity")], "calibration")
    for target, first, followups in [
        ("tesla", "My Tesla needs tire maintenance.", ["What type should I use?", "Is the premium option better?",
         "What about winter?", "How often should I change them?", "Is there a safer choice?", "What should I do next?"]),
        ("honda", "My Honda needs an oil change.", ["What type should I use?", "Is synthetic better?",
         "What about winter?", "How often should I change it?", "Is the cheaper option okay?", "How do I check it?"]),
        ("terry", "Terry joined a basketball team.", ["He likes it.", "How should he practice?",
         "What about weekends?", "Is that enough?", "Should he do it more often?", "What would help him improve?"]),
        ("python", "I'm learning Python.", ["Where should I start?", "Is that difficult?",
         "How much practice is enough?", "What about weekends?", "How can I improve?", "What should I do next?"]),
    ]:
        turns = [user(first, [target], "discovery")]
        for text in followups:
            turns.extend([assistant("There are several options; consider your needs."), user(text, [target], "continuity")])
        add("long_"+target, turns)
    add("topic_switch_return", [user("My Tesla needs service.", ["tesla"], "discovery"),
        assistant("Let's discuss the service."), user("My dog has been scratching.", ["dog"], "switch"),
        assistant("How long has this been happening?"), user("Terry has basketball tonight.", ["terry"], "switch"),
        user("Back to the car, what maintenance does the Tesla need?", ["tesla"], "reactivation")])
    add("implicit_return", [user("My Tesla needs service.", ["tesla"], "discovery"),
        user("My dog has been scratching.", ["dog"], "switch"),
        user("Terry has basketball tonight.", ["terry"], "switch"),
        user("Back to the car, what should I do?", ["tesla"], "reactivation")])
    add("multiple", [user("My Tesla and Honda both need tires.", ["tesla", "honda"], "discovery"),
        assistant("Both vehicles may need attention."), user("How often should I replace them?", ["tesla", "honda"], "ambiguous")], "calibration")
    add("older_vehicle", [user("I have a Tesla and a Honda.", ["tesla", "honda"], "discovery"),
        assistant("You own both vehicles."), user("The older one needs service.", ["honda"], "ambiguous")])
    add("pronoun", [user("Terry joined a basketball team.", ["terry"], "discovery"),
        assistant("That sounds exciting."), user("He likes it.", ["terry"], "continuity")], "calibration")
    add("hard_switch", [user("My Tesla needs service.", ["tesla"], "discovery"),
        user("What about winter?", ["tesla"], "continuity"), assistant("Consider the seasonal conditions."),
        user("Anyway, my main aquarium has algae.", ["main_aquarium"], "switch"),
        user("How should I clean it?", ["main_aquarium"], "continuity")])
    add("multi_topic", [user("My Tesla needs service, Terry has basketball tonight, and my dog has been scratching.",
        ["tesla", "terry", "dog"], "discovery")], "calibration")
    add("five_topics", [user("My Tesla needs service, Terry has basketball tonight, I'm learning Python, "
        "my dog has been scratching, and my main aquarium has algae.",
        ["tesla", "terry", "python", "dog", "main_aquarium"], "discovery")])
    negatives = ["What's the weather tomorrow?", "What is twelve times eleven?", "Tell me a joke.",
        "How far is the moon from Earth?", "Translate hello into Spanish.", "What is the capital of France?"]
    for n, text in enumerate(negatives):
        add("negative_"+str(n), [user(text, [], "negative")], "calibration" if n % 2 == 0 else "holdout")
    add("unrelated_after_topic", [user("My Tesla needs service.", ["tesla"], "discovery"),
        assistant("We can discuss maintenance."), user("What's the weather tomorrow?", [], "negative"),
        assistant("I don't have a weather forecast."), user("Tell me a joke.", [], "negative")])
    add("assistant_bias", [user("My dog has been scratching.", ["dog"], "discovery"),
        assistant("Your Tesla Model Y and Little Dog are worth discussing instead."),
        user("Please continue about that scratching.", ["dog"], "continuity")])
    add("contrast", [user("Not my Tesla; I mean my Honda needs service.", ["honda"], "discovery")])
    add("alias_collision", [user("Little Dog needs service.", ["tesla"], "discovery"),
        user("My dog has been scratching.", ["dog"], "switch")])
    add("long_topic_walk", [user(f"Let's discuss {payload['entity_name']}.", [key], "discovery")
        for key, payload in bank.items()] + [user("Back to my Tesla, what service is needed?", ["tesla"], "reactivation")])
    return bank, conversations


def normalize(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def lexical_hits(text, bank):
    haystack = " "+normalize(text)+" "
    return {key for key, e in bank.items() if any(" "+normalize(name)+" " in haystack
        for name in [e["entity_name"], *e["aliases"]] if normalize(name))}


def representations(conversations):
    guard = SecurityGuard()
    turns = []
    for conversation in conversations:
        history = []
        for index, turn in enumerate(conversation["turns"]):
            text = guard.sanitize_input(turn["text"])
            assert text == turn["text"] and guard.check_for_injection(text)
            if turn["role"] == "user":
                context = {}
                for n in (2, 4):
                    context[str(n)] = "\n".join(f"{t['role'].upper()}: {t['text']}" for t in history[-n:])
                    context[str(n)] += ("\n" if context[str(n)] else "") + "USER: "+text
                turns.append(dict(**turn, conversation=conversation["name"], split=conversation["split"],
                                  index=index, context=context))
            history.append(turn)
    return turns


def policies():
    result = [dict(name=f"top{k}", family="topk", k=k) for k in (1, 3, 5)]
    result += [dict(name=f"floor{f:.2f}", family="floor", floor=f, k=5) for f in (.20, .25, .30, .35, .40)]
    result += [dict(name=f"margin{f:.2f}_{gap:.2f}", family="margin", floor=f, margin=gap, k=5)
               for f in (.25, .30, .35) for gap in (.02, .05)]
    result += [dict(name=f"lexical{f:.2f}", family="lexical", floor=f, k=5) for f in (.25, .30, .35, .40)]
    result += [dict(name=f"weighted{k}", family="weighted", k=k) for k in (3, 5)]
    return result


def activate(scores, factors, lexical, policy):
    family = policy["family"]
    values = {k: v*factors[k] if family == "weighted" else v for k, v in scores.items()}
    order = sorted(values, key=lambda k: (-values[k], k))
    if family in ("topk", "weighted"):
        return order[:policy["k"]]
    # Rank 6 is a deployable background proxy, NOT an oracle-labelled irrelevant row.
    background = values[order[5]]
    accepted = [k for k in order[:5] if values[k] >= policy["floor"] and
                (family != "margin" or values[k]-background >= policy["margin"])]
    if family == "lexical":
        accepted = list(set(accepted) | lexical)
        return sorted(accepted, key=lambda k: (k not in lexical, -values[k], k))[:5]
    return accepted


class SessionState:
    def __init__(self, cap):
        self.cap, self.states, self.last_activity = cap, {}, {}

    def touch(self, key, now, activated=()):
        if key in self.last_activity and now-self.last_activity[key] >= 86400:
            self.states.pop(key, None)
        ids = self.states.setdefault(key, [])
        # Reverse promotion preserves highest-confidence-first in the resulting MRU.
        for identity in reversed(activated):
            if identity in ids:
                ids.remove(identity)
            ids.insert(0, identity)
        del ids[self.cap:]
        self.last_activity[key] = now
        return list(ids)


def simulate(conversations, turns, bank, scores, factors, representation, policy, retention, returned):
    lookup = {(t["conversation"], t["index"]): t for t in turns}
    records, assistant_checks = [], 0
    session = SessionState(retention)
    for conversation in conversations:
        key = ("synthetic-tenant", "synthetic-user", conversation["name"])
        ever_relevant, discovered = set(), set()
        for index, turn in enumerate(conversation["turns"]):
            if turn["role"] == "assistant":
                before = list(session.states.get(key, []))
                assert session.touch(key, index*60) == before
                assistant_checks += 1
                continue
            t = lookup[(conversation["name"], index)]
            selected = activate(scores[(t["conversation"], index)][representation], factors,
                                lexical_hits(t["text"], bank), policy)
            gold = set(t["gold"])
            current = session.touch(key, index*60, selected)
            output = current[:returned]
            ever_relevant |= gold
            was_discovered = bool(gold) and gold <= discovered
            discovered |= set(selected) & gold
            records.append(dict(conversation=conversation["name"], index=index, split=t["split"], kind=t["kind"],
                text=t["text"], gold=sorted(gold), activated=selected, retained=current, returned=output,
                previously_discovered=was_discovered,
                pollution=len(set(current)-ever_relevant),
                irrelevant_retained=len(set(current)-gold),
                # Persona is logically returned separately; never enters these IDs/caps.
                persona_returned=True))
    return records, assistant_checks


def metrics(rows):
    positives = [r for r in rows if r["gold"]]
    negatives = [r for r in rows if not r["gold"]]
    activated = sum(len(r["activated"]) for r in rows)
    true = sum(len(set(r["activated"]) & set(r["gold"])) for r in rows)
    relevant = sum(len(r["gold"]) for r in rows)
    continuity = [r for r in rows if r["kind"] == "continuity" and r["previously_discovered"]]
    long = [r for r in positives if r["conversation"].startswith("long_") and r["kind"] == "continuity"]
    switches = [r for r in positives if r["kind"] == "switch"]
    reactivate = [r for r in positives if r["kind"] == "reactivation"]
    discovered = [r for r in positives if r["kind"] == "discovery"]
    average = lambda values: statistics.mean(values) if values else None
    front = lambda group: average([bool(r["returned"]) and r["returned"][0] in r["gold"] for r in group])
    return dict(user_turns=len(rows), activation_recall=true/relevant if relevant else None,
        activation_precision=true/activated if activated else 1.0,
        false_activation_turn_rate=average([bool(set(r["activated"])-set(r["gold"])) for r in rows]),
        negative_turn_activation_rate=average([bool(r["activated"]) for r in negatives]),
        discovery_recall=average([len(set(r["activated"]) & set(r["gold"]))/len(r["gold"]) for r in discovered]),
        conditional_continuity=average([set(r["gold"]) <= set(r["returned"]) for r in continuity]),
        long_continuity=average([set(r["gold"]) <= set(r["returned"]) for r in long]),
        switch_front=front(switches), reactivation_front=front(reactivate),
        return_all_needed=average([set(r["gold"]) <= set(r["returned"]) for r in positives]),
        return_entity_recall=average([len(set(r["returned"]) & set(r["gold"]))/len(r["gold"]) for r in positives]),
        irrelevant_returned=average([len(set(r["returned"])-set(r["gold"])) for r in rows]),
        irrelevant_retained=average([r["irrelevant_retained"] for r in rows]),
        pollution=average([r["pollution"] for r in rows]),
        max_pollution=max((r["pollution"] for r in rows), default=0))


def isolation_checks():
    state = SessionState(10)
    keys = [("tenant", "user", "one"), ("tenant", "user", "two"),
            ("tenant", "other", "one"), ("other", "user", "one")]
    for index, key in enumerate(keys):
        state.touch(key, 0, [f"identity-{index}"])
    assert len({generate_common_key(t, s, u) for t, u, s in keys}) == 4
    for index, key in enumerate(keys):
        assert state.states[key] == [f"identity-{index}"]
        assert all(validate_id(part) for part in key)
    key = keys[0]
    assert state.touch(key, 86399) == ["identity-0"]  # assistant-like refresh, no promotion
    assert state.touch(key, 86401) == ["identity-0"]
    assert state.touch(key, 172801) == []
    # Existing key helpers rely on colon-safe IDs; future dedicated prefix still needed.
    return dict(namespace_checks=4, sliding_ttl_checks=3, state="offline only; no Redis key created")


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run from tomte-engine")
    load_dotenv(ROOT / ".env", override=False)
    for name in ("OPENAI_API_KEY", "POSTGRES_URL", "REDIS_URL"):
        if not os.environ.get(name):
            raise RuntimeError(f"Missing {name}")
    if any(urlsplit(os.environ[n]).hostname not in ("localhost", "127.0.0.1", "::1") for n in ("POSTGRES_URL", "REDIS_URL")):
        raise RuntimeError("Only local infrastructure is allowed")
    assert settings.openai.vector_dimension == 1536 and settings.isolate_chat_sessions
    assert not settings.enable_pii_scrubbing
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    bank, conversations = fixtures()
    turns = representations(conversations)
    summaries = {k: get_entity_summary(e) for k, e in bank.items()}
    vector_cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    usage, reused, original_send = [], defaultdict(int), httpx.AsyncClient.send

    async def embeddings_only(client, request, *args, **kwargs):
        if not request.url.path.endswith("/embeddings"):
            raise RuntimeError("Non-embedding request blocked")
        body = json.loads(request.content)
        response = await original_send(client, request, *args, **kwargs)
        await response.aread()
        usage.append(dict(model=body.get("model"), texts=len(body["input"]) if isinstance(body["input"], list) else 1,
                          status=response.status_code, usage=response.json().get("usage", {})))
        USAGE.write_text(json.dumps(usage, indent=2), encoding="utf-8")
        return response

    provider = OpenAILLMProvider(os.environ["OPENAI_API_KEY"])
    def key(role, text):
        return hashlib.sha256(json.dumps([settings.openai.embed_model, settings.openai.vector_dimension, role, text]).encode()).hexdigest()

    def embed(role, texts):
        unique = list(dict.fromkeys(texts))
        pending = [t for t in unique if key(role, t) not in vector_cache]
        reused[role] = len(unique)-len(pending)
        for start in range(0, len(pending), 100):
            chunk = pending[start:start+100]
            vectors = provider.embed_batch(chunk) if role == "document" else provider.embedder.embed_query_sync(chunk).embeddings
            for text, vector in zip(chunk, vectors):
                assert len(vector) == 1536
                vector_cache[key(role, text)] = list(vector)
            CACHE.write_text(json.dumps(vector_cache), encoding="utf-8")
        return {t: vector_cache[key(role, t)] for t in unique}

    OUTPUT.parent.mkdir(exist_ok=True)
    httpx.AsyncClient.send = embeddings_only
    try:
        docs = embed("document", summaries.values())
        queries = embed("query", [text for t in turns for text in (t["text"], t["context"]["2"], t["context"]["4"])])
    finally:
        httpx.AsyncClient.send = original_send
    print(f"Embeddings ready: {len(bank)} entities, {len(turns)} user turns, {len(queries)} query representations", flush=True)
    scores = {}
    for t in turns:
        raw = {k: cosine(queries[t["text"]], docs[s]) for k, s in summaries.items()}
        c2 = {k: cosine(queries[t["context"]["2"]], docs[s]) for k, s in summaries.items()}
        c4 = {k: cosine(queries[t["context"]["4"]], docs[s]) for k, s in summaries.items()}
        scores[(t["conversation"], t["index"])] = dict(current=raw, context2=c2, context4=c4,
            **{f"dual{w}": {k: max(raw[k], w*c4[k]) for k in bank} for w in (.5, .7, .9)})

    db = PostgresProvider(os.environ["POSTGRES_URL"])
    observer = psycopg.connect(os.environ["POSTGRES_URL"], autocommit=True, options="-c default_transaction_read_only=on")
    redis_client = redis.Redis.from_url(os.environ["REDIS_URL"])
    before = state(observer, redis_client)
    tenant = "session-entity-evaluation-"+uuid4().hex
    factors, checks, max_difference = {}, 0, 0.0
    try:
        try:
            with db.transaction():
                updates = []
                for t, u in [(tenant, "equal"), (tenant, "weighted"), (tenant, "other-user"), (tenant+"-other", "equal")]:
                    for identity, payload in bank.items():
                        updates.append(dict(mode="INSERT", tenant_id=t, user_id=u,
                            **{k: payload[k] for k in ("entity_id", "entity_name", "entity_domain", "entity_type")},
                            payload=deepcopy(payload), embedding=docs[summaries[identity]]))
                assert db.commit_graph_updates(updates)
                db.conn.execute("UPDATE tomte_memory_engine SET last_mentioned_at=NOW()-interval '24 hours' "
                    "WHERE tenant_id=%s AND user_id='weighted' AND entity_id='tesla'", (tenant,))
                db.conn.execute("UPDATE tomte_memory_engine SET mention_count=5 WHERE tenant_id=%s "
                    "AND user_id='weighted' AND entity_id='honda'", (tenant,))
                db.conn.execute("UPDATE tomte_memory_engine SET mention_count=8 WHERE tenant_id=%s "
                    "AND user_id='weighted' AND entity_id='python'", (tenant,))
                for user in ("equal", "weighted"):
                    factors[user] = {r["entity_id"]: float(r["factor"]) for r in db.conn.execute(
                        "SELECT entity_id, mention_count*exp(-0.0001*extract(epoch FROM (now()-last_mentioned_at))) factor "
                        "FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s", (tenant, user)).fetchall()}
                    # Every distinct representation checked through production API.
                    for text, vector in queries.items():
                        rows = db.semantic_retrieval(tenant, user, vector)
                        assert all(r["tenant_id"] == tenant and r["user_id"] == user for r in rows)
                        raw = {k: cosine(vector, docs[s]) for k, s in summaries.items()}
                        expected = sorted(bank, key=lambda k: (-raw[k]*factors[user][k], k))[:settings.semantic_retrieval_limit]
                        assert [r["entity_id"] for r in rows] == expected
                        pg = db.conn.execute("SELECT entity_id, 1-(embedding <=> %s::vector) cosine FROM tomte_memory_engine "
                            "WHERE tenant_id=%s AND user_id=%s", (vector, tenant, user)).fetchall()
                        max_difference = max(max_difference, max(abs(raw[r["entity_id"]]-float(r["cosine"])) for r in pg))
                        checks += 1
                for t, u in [(tenant, "other-user"), (tenant+"-other", "equal")]:
                    rows = db.semantic_retrieval(t, u, queries[turns[0]["text"]])
                    assert len(rows) == 5 and all(r["tenant_id"] == t and r["user_id"] == u for r in rows)
                    checks += 1
                assert db.semantic_retrieval(tenant, "absent", queries[turns[0]["text"]]) == []
                checks += 1
                raise RollbackEvaluation()
        except RollbackEvaluation:
            pass
        after = state(observer, redis_client)
        assert before == after, "Persistent state changed"
    finally:
        db.close()
        observer.close()
        redis_client.close()

    comparisons, traces = [], {}
    for representation in ("current", "context2", "context4", "dual0.5", "dual0.7", "dual0.9"):
        for policy in policies():
            for weighting in (("equal", "weighted") if policy["family"] == "weighted" else ("equal",)):
                for retention in (10, 20):
                    for returned in (3, 5, 8):
                        records, assistant_checks = simulate(conversations, turns, bank, scores, factors[weighting],
                            representation, policy, retention, returned)
                        comparison = dict(representation=representation, policy=policy, weighting=weighting,
                            retention=retention, returned=returned, overall=metrics(records),
                            calibration=metrics([r for r in records if r["split"] == "calibration"]),
                            holdout=metrics([r for r in records if r["split"] == "holdout"]), assistant_checks=assistant_checks)
                        comparisons.append(comparison)
                        if policy["name"] in ("top5", "floor0.35", "lexical0.35", "weighted5") and retention == 10:
                            traces[f"{representation}/{policy['name']}/{weighting}/return{returned}"] = records
    # Calibrate on explicit/follow-up/negative/multi cases; holdout is reported,
    # never used to change floors or combination weights.
    candidates = [c for c in comparisons if c["retention"] == 10 and c["returned"] == 5 and c["weighting"] == "equal"]
    def utility(c):
        m = c["calibration"]
        return 2*m["return_all_needed"]+m["activation_precision"]-m["false_activation_turn_rate"]-.1*m["irrelevant_returned"]
    selected = max(candidates, key=utility)
    selected_records, _ = simulate(conversations, turns, bank, scores, factors[selected["weighting"]],
        selected["representation"], selected["policy"], selected["retention"], selected["returned"])
    traces["calibration_selected"] = selected_records
    report = dict(entities=bank, conversations=conversations, user_turns=turns,
        entity_count=len(bank), conversation_count=len(conversations), user_turn_count=len(turns),
        assistant_turn_count=sum(t["role"] == "assistant" for c in conversations for t in c["turns"]),
        model=settings.openai.embed_model, dimension=settings.openai.vector_dimension,
        history_limit=settings.history_chat_number, isolate_chat_sessions=settings.isolate_chat_sessions,
        context_window_definition="last 2/4 individual user-or-assistant messages, plus current user; role-labelled",
        dual_definition="max(current cosine, w * context4 cosine)",
        margin_definition="candidate minus rank-6 score; proxy not guaranteed irrelevant; no gold-label oracle",
        mru_order="lexical priority where enabled, then descending cosine, tie entity ID; reverse push keeps strongest first",
        metrics_definition=dict(activation="selected/promoted this turn, including already retained IDs",
            false_activation="fraction of user turns promoting any ID not relevant this turn",
            pollution="retained IDs never relevant yet in this conversation",
            irrelevant_returned="returned IDs irrelevant this turn, including formerly relevant old topics",
            continuity="conditional on prior discovery plus unconditional long-follow-up score",
            return_all_needed="all gold IDs present; excludes turns with no gold IDs"),
        policy_comparisons=comparisons, traces=traces, calibration_selected=selected,
        score_records=[dict(conversation=t["conversation"], index=t["index"],
            **scores[(t["conversation"],t["index"])]) for t in turns],
        weighting_factors=factors, pg_checks=checks, pg_max_cosine_difference=max_difference,
        synthetic_rows=len(bank)*4, before=before, after=after, isolation=isolation_checks(),
        embedding_usage=usage, requests=len(usage), texts_embedded=sum(r["texts"] for r in usage),
        input_tokens=sum(r["usage"].get("prompt_tokens",0) for r in usage),
        cache_reused=dict(reused), summaries=summaries, cleanup="all synthetic PostgreSQL rows rolled back; Redis never written")
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("entity_count", "conversation_count", "user_turn_count", "assistant_turn_count",
        "requests", "texts_embedded", "input_tokens", "pg_checks", "pg_max_cosine_difference", "cleanup", "calibration_selected")}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        print(f"Evaluation failed: {type(exc).__name__}; exception content withheld")
        print([(Path(f.filename).name, f.lineno, f.name) for f in traceback.extract_tb(exc.__traceback__)])
        raise SystemExit(1)
