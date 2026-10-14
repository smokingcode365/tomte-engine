"""Replay existing synthetic evaluations through real session SQL/Redis/workflow.

Uses cached real embeddings only (no model/API requests), worker disabled, and
rollback-only PostgreSQL fixtures. Deletes only its unique Redis namespace.
Run from tomte-engine after the existing activation evaluation has populated log/.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
import logging
import os
import re
from pathlib import Path
import statistics
import threading
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import load_dotenv
import psycopg
import redis

from evaluate_entity_activation import dataset, activate, metrics
from evaluate_memory_retrieval import cosine, state, RollbackEvaluation
from evaluate_session_entities import lexical_hits, SessionState
from tomte.engine import Tomte
from tomte.engine.core.config import settings
from tomte.engine.core.enums import ChatSenderRole, MemoryDomain
from tomte.engine.llm.embedder import get_entity_summary
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.providers.redis_provider import RedisCacheProvider
from tomte.engine.utils.keys import generate_session_entity_key, get_section_key_active_sessions
from tomte.engine.workflows.chat_hook import ChatHookWorkflow

ROOT = Path(__file__).resolve().parents[1]


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run from tomte-engine")
    load_dotenv(ROOT / ".env", override=False)
    for name in ("POSTGRES_URL", "REDIS_URL"):
        if not os.environ.get(name) or urlsplit(os.environ[name]).hostname not in ("localhost", "127.0.0.1", "::1"):
            raise RuntimeError("Local PostgreSQL/Redis required")
    assert not settings.enable_pii_scrubbing
    assert settings.openai.embed_model == "openai:text-embedding-3-small"
    assert settings.openai.vector_dimension == 1536
    logging.getLogger("tomte.engine").setLevel(logging.WARNING)
    vectors = json.loads((ROOT / "log/memory_retrieval_embeddings.json").read_text())
    def cached(role, text):
        key = hashlib.sha256(json.dumps([settings.openai.embed_model, 1536, role, text]).encode()).hexdigest()
        if key not in vectors:
            raise RuntimeError("Missing cached synthetic embedding; no API fallback")
        return vectors[key]
    bank, _, cases, conversations = dataset()
    eligible_bank = {eid: entity for eid, entity in bank.items()
                     if entity["entity_domain"] != MemoryDomain.IDENTITY.value}
    docs = {eid: cached("document", get_entity_summary(entity)) for eid, entity in bank.items()}
    texts = {case["text"] for case in cases} | {turn["text"] for c in conversations for turn in c["turns"] if turn["role"] == "user"}
    scores = {text: {eid: cosine(cached("query", text), docs[eid]) for eid in bank} for text in texts}
    eligible_scores = {text: {eid: value for eid, value in values.items() if eid in eligible_bank}
                       for text, values in scores.items()}
    rule = dict(name="production", family="lexical", floor=settings.session_entity_cosine_floor,
                cap=settings.session_entity_activation_cap)
    db = PostgresProvider(os.environ["POSTGRES_URL"])
    cache = RedisCacheProvider(os.environ["REDIS_URL"])
    raw_cache = redis.Redis.from_url(os.environ["REDIS_URL"])
    observer = psycopg.connect(os.environ["POSTGRES_URL"], autocommit=True,
                              options="-c default_transaction_read_only=on")
    before = state(observer, raw_cache)
    tenant = "session-implementation-" + uuid4().hex
    calls = Counter()
    class CachedEmbeddingProvider:
        def embed_text(self, text):
            calls["cached_embedding"] += 1
            return cached("query", text)
    original_start = threading.Thread.start
    def forbidden_start(*args, **kwargs):
        raise AssertionError("Background thread forbidden")
    threading.Thread.start = forbidden_start
    activations = []
    original_promote = cache.promote_session_entities
    def recorded_promote(t, u, s, ids, retention, ttl):
        activations.append(list(ids))
        return original_promote(t, u, s, ids, retention, ttl)
    cache.promote_session_entities = recorded_promote
    for name in ("upsert_user_activity", "get_important_entities", "get_recent_entities", "get_frequent_entities",
                 "semantic_retrieval", "find_session_entity_mentions", "discover_session_entities", "get_entities_by_ids"):
        original = getattr(db, name)
        def counted(*args, _name=name, _operation=original, **kwargs):
            calls[_name] += 1
            return _operation(*args, **kwargs)
        setattr(db, name, counted)
    rows, turns, differences = [], [], []
    query_counts = []
    assistant_checks = 0
    try:
        Tomte.init(db, cache, CachedEmbeddingProvider(), start_daemon_worker=False)
        workflow = ChatHookWorkflow()
        try:
            with db.transaction():
                updates = [dict(mode="INSERT", tenant_id=tenant, user_id="u", **{field: entity[field]
                    for field in ("entity_id", "entity_name", "entity_domain", "entity_type")},
                    payload=deepcopy(entity), embedding=docs[eid]) for eid, entity in bank.items()]
                assert db.commit_graph_updates(updates)
                for case in cases:
                    prior_calls = calls.copy()
                    workflow.on_chat(tenant, case["id"], case["text"], ChatSenderRole.USER, "u")
                    if len(query_counts) < 2:
                        query_counts.append(dict(calls - prior_calls))
                    actual = activations[-1]
                    expected = activate(eligible_scores[case["text"]], {eid: 1 for eid in eligible_bank}, lexical_hits(case["text"], eligible_bank), rule)
                    if actual != expected:
                        differences.append(dict(case=case["id"], actual=actual, expected=expected))
                    rows.append(dict(case, context_gold=case["gold"],
                                     gold=[eid for eid in case["gold"] if eid in eligible_bank],
                                     ambiguous=case["ambiguous"] and sum(eid in eligible_bank for eid in case["gold"]) > 1,
                                     activated=actual))
                simulator = SessionState(settings.session_entity_retention_cap)
                for c in conversations:
                    session = "conversation-" + c["name"]
                    scope = (tenant, "u", session)
                    ever_valid = {eid for turn in c["turns"] for eid in turn["gold"]}
                    for index, turn in enumerate(c["turns"]):
                        old = cache.get_session_entity_ids(*scope, 10)
                        result = workflow.on_chat(tenant, session, turn["text"], ChatSenderRole(turn["role"]), "u")
                        actual = cache.get_session_entity_ids(*scope, 10)
                        if turn["role"] == "assistant":
                            assert actual == old
                            assistant_checks += 1
                            continue
                        selected = activate(eligible_scores[turn["text"]], {eid: 1 for eid in eligible_bank}, lexical_hits(turn["text"], eligible_bank), rule)
                        expected = simulator.touch(scope, index * 60, selected)
                        if actual != expected:
                            differences.append(dict(conversation=c["name"], turn=index, actual=actual, expected=expected))
                        fetched = db.get_entities_by_ids(tenant, "u", cache.get_session_entity_ids(*scope, 5))
                        returned = [entity["entity_id"] for entity in fetched]
                        assert all("[[ENTITY: " + eid + " |" in result["context_text"] for eid in returned)
                        ttl = cache.client.ttl(generate_session_entity_key(*scope))
                        assert not actual or 86380 <= ttl <= 86400
                        turns.append(dict(conversation=c["name"], kind=turn["kind"], text=turn["text"], gold=[eid for eid in turn["gold"] if eid in eligible_bank],
                            context_gold=turn["gold"], context_returned=re.findall(r"\[\[ENTITY: ([^| ]+) \|", result["context_text"]),
                            activated=activations[-1], retained=actual, returned=returned,
                            pollution=len(set(actual) - ever_valid)))
                for scope in [(tenant, "u", "absent"), (tenant, "other", "case-001"), (tenant + "-other", "u", "case-001")]:
                    assert cache.get_session_entity_ids(*scope, 10) == []
                    if scope[0] != tenant or scope[1] != "u":
                        assert db.get_entities_by_ids(scope[0], scope[1], ["tesla"]) == []
                raise RollbackEvaluation()
        except RollbackEvaluation:
            pass
    finally:
        keys = list(cache.client.scan_iter(match="*" + tenant + "*"))
        if keys:
            cache.client.delete(*keys)
        active = get_section_key_active_sessions()
        refs = [ref for ref, _ in cache.client.zscan_iter(active) if ref.startswith(tenant + ":")]
        if refs:
            cache.client.zrem(active, *refs)
        after = state(observer, raw_cache)
        observer.close()
        raw_cache.close()
        cache.client.close()
        db.close()
        threading.Thread.start = original_start
    positive = [turn for turn in turns if turn["gold"]]
    context_positive = [turn for turn in turns if turn["context_gold"]]
    report = dict(persona_excluded_from_session_caps=True, eligible_entities=len(eligible_bank),
        all_required_caller_context_availability=statistics.mean(set(t["context_gold"]) <= set(t["context_returned"]) for t in context_positive),
        activation_cases=len(rows), session_user_turns=len(turns), assistant_nonpromotion_checks=assistant_checks,
        simulator_differences=differences, activation_metrics=metrics(rows),
        all_required_top5_availability=statistics.mean(set(turn["gold"]) <= set(turn["returned"]) for turn in positive),
        mean_pollution=statistics.mean(turn["pollution"] for turn in turns), max_pollution=max(turn["pollution"] for turn in turns),
        multi_entity_availability=[dict(conversation=t["conversation"], gold=t["gold"], returned=t["returned"]) for t in turns if len(t["gold"]) > 1],
        cross_scope_checks=3, representative_user_turn_calls=query_counts, calls=dict(calls), external_api_calls=0, before=before, after=after,
        cleanup_verified=before == after, activations=rows, session_turns=turns)
    (ROOT / "log/session_implementation_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in ("activations", "session_turns", "before", "after")}, indent=2))
    assert before == after, "Synthetic cleanup did not restore baseline state"
    assert not differences, "Production session ordering differs from simulator"


if __name__ == "__main__":
    main()
