"""Synthetic representation evaluation only; never changes production dictionary policy.

Run from the checkout with local PostgreSQL/Redis available and configured .env:
    uv run python scripts/evaluate_dictionary_representations.py

Makes one embedding batch, no chat/model calls. All temporary anchors are rolled
back. Summary and terminology queries use manually supplied gold information:
they are compatibility upper bounds, not implemented pre-extraction mechanisms.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np
import redis
from dotenv import load_dotenv

from tomte.engine.core.config import settings
from tomte.engine.dictionary.logic import calculate_topic_centroids
from tomte.engine.dictionary.service import DictionaryService
from tomte.engine.llm.embedder import get_entity_summary
from tomte.engine.providers.openai_provider import OpenAILLMProvider
from tomte.engine.providers.postgres_provider import PostgresProvider

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    dict(concept="vehicle_color", original="I painted my Tesla red.", followup="My Tesla is still red.",
         name="Tesla", domain="assets_and_possessions", entity_type="vehicle", attribute="paint_color", old="red", new="red"),
    dict(concept="ownership", original="I own a Tesla.", followup="I sold my Tesla.",
         name="Tesla", domain="assets_and_possessions", entity_type="vehicle", attribute="ownership_status", old="owned", new="sold"),
    dict(concept="employment", original="I work as a Software Engineer.", followup="My position is Software Engineer.",
         name="User career", domain="activities_and_skills", entity_type="employment", attribute="job_title", old="Software Engineer", new="Software Engineer"),
    dict(concept="home_address", original="I live at 123 Example Street.", followup="My address is still 123 Example Street.",
         name="User home", domain="assets_and_possessions", entity_type="residence", attribute="current_home_address", old="123 Example Street", new="123 Example Street"),
    dict(concept="programming_language", original="I use Python.", followup="I also code in Python.",
         name="User programming", domain="activities_and_skills", entity_type="skill", attribute="programming_language", old="Python", new="Python"),
]
NEGATIVE = dict(concept="interior_color", original="My Tesla has a beige interior.", followup="The inside of my Tesla is beige.",
                name="Tesla", domain="assets_and_possessions", entity_type="vehicle", attribute="interior_color", old="beige", new="beige")


def summary(case, value):
    return get_entity_summary(dict(entity_name=case["name"], entity_domain=case["domain"], entity_type=case["entity_type"],
        aliases=[], facts=[dict(attribute_name=case["attribute"], value=value, importance=3, assertion_state="ACTIVE")]))


def terms(case):
    # Gold labels deliberately supplied for this upper-bound evaluation.
    return f"ENTITY TYPE: {case['entity_type']} | ATTRIBUTE NAME: {case['attribute']}"


def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def dictionary_state(db):
    # Compare full row digests without reporting existing dictionary content.
    result = {}
    with db.conn.cursor() as cur:
        for table in ("tomte_semantic_anchors", "tomte_keyword_registry", "tomte_keyword_audit_log"):
            cur.execute("SELECT * FROM " + table)
            rows = sorted(json.dumps(row, sort_keys=True, default=str) for row in cur.fetchall())
            result[table] = {"count": len(rows), "digest": hashlib.sha256("\n".join(rows).encode()).hexdigest()}
    return result


class RollbackEvaluation(Exception):
    """Leave temporary anchors and all original dictionary rows exactly as found."""


def main():
    if Path.cwd().resolve() != ROOT:
        raise SystemExit("Run this evaluation from the tomte-engine repository root.")
    load_dotenv(ROOT / ".env", override=False)
    required = ("OPENAI_API_KEY", "POSTGRES_URL", "REDIS_URL")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit("Missing configuration names: " + ", ".join(missing))
    if any(urlsplit(os.environ[name]).hostname not in ("127.0.0.1", "localhost", "::1") for name in required[1:]):
        raise SystemExit("This evaluation requires local PostgreSQL and Redis endpoints.")
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    db = PostgresProvider(os.environ["POSTGRES_URL"])
    cache = redis.Redis.from_url(os.environ["REDIS_URL"])
    report = {"threshold": settings.llm_threshold_low, "oracle_query_warning":
              "Summary/summary and terminology-channel queries use gold information unavailable before extraction; "
              "centroid diagnostics include the original message and are not held-out retrieval scores."}
    try:
        before = dictionary_state(db)
        redis_before = cache.dbsize()
        # Existing rows would contaminate bootstrap's unscoped word lists. Never clear them.
        if before["tomte_semantic_anchors"]["count"] or before["tomte_keyword_registry"]["count"]:
            raise RuntimeError("Evaluation requires an empty dictionary; existing rows were not changed.")
        payloads = []
        offsets = []
        for case in [*CASES, NEGATIVE]:
            offsets.append(len(payloads))
            payloads.extend([case["original"], case["followup"], summary(case, case["old"]),
                             summary(case, case["new"]), terms(case)])
        provider = OpenAILLMProvider(os.environ["OPENAI_API_KEY"])
        # Existing provider's Embedder preserves configured document semantics and exposes usage.
        response = provider.embedder.embed_documents_sync(payloads)
        vectors = [list(v) for v in response.embeddings]
        if len(vectors) != len(payloads) or any(len(v) != settings.openai.vector_dimension for v in vectors):
            raise RuntimeError("Unexpected embedding count or dimensions.")
        report["embedding_usage"] = {"batch_calls": 1, "texts": len(payloads), "dimension": settings.openai.vector_dimension,
                                     "input_tokens": response.usage.input_tokens}
        items = []
        for case, index in zip([*CASES, NEGATIVE], offsets):
            a, q, s, t, term = vectors[index:index+5]
            items.append(dict(case=case, chat=a, query=q, summary=s, summary_query=t, terms=term))
        report["comparisons"] = []
        for item in items[:5]:
            centroid = calculate_topic_centroids([item["chat"], item["query"]])[0]
            report["comparisons"].append({"concept": item["case"]["concept"],
                "chat_chat": cosine(item["chat"], item["query"]), "summary_chat_current": cosine(item["summary"], item["query"]),
                "summary_summary_oracle": cosine(item["summary"], item["summary_query"]),
                "chat_centroid_nonheldout": cosine(item["chat"], centroid), "summary_centroid": cosine(item["summary"], centroid)})
        strategies = {"current_summary_chat": [("summary", "query")], "A_chat_chat": [("chat", "query")],
                      "B_summary_summary_oracle": [("summary", "summary_query")],
                      "C_dual_chat_terms_oracle": [("chat", "query"), ("terms", "terms")]}
        service = object.__new__(DictionaryService)
        service.db = db
        report["strategies"] = {}
        for name, channels in strategies.items():
            scores = [[max(cosine(anchor[write], query[read]) for write, read in channels)
                       for anchor in items[:5]] for query in items]
            same = [scores[i][i] for i in range(5)]
            cross = [score for i, row in enumerate(scores) for j, score in enumerate(row) if i != j]
            margins = [scores[i][i] - max(s for j, s in enumerate(scores[i]) if i != j) for i in range(5)]
            metrics = {"same_concept_retrieved": sum(s > report["threshold"] for s in same), "same_concept_total": 5,
                "same_concept_rate": sum(s > report["threshold"] for s in same)/5,
                "same_mean": float(np.mean(same)), "same_min": min(same),
                "off_concept_retrieved": sum(s > report["threshold"] for s in cross), "off_concept_total": len(cross),
                "off_concept_rate": sum(s > report["threshold"] for s in cross)/len(cross),
                "off_concept_mean": float(np.mean(cross)), "off_concept_max": max(cross),
                "mean_best_other_margin": float(np.mean(margins)), "minimum_best_other_margin": min(margins),
                "deliberate_comparisons": {"vehicle_color_vs_job_title": scores[2][0],
                    "home_address_vs_programming_language": scores[4][3], "ownership_vs_interior_color": scores[5][1]},
                "score_matrix": scores, "bootstrap": []}
            try:
                with db.transaction():
                    owned = {}
                    for j, item in enumerate(items[:5]):
                        for write, _ in channels:
                            anchor = db.create_dictionary_anchor(item[write])
                            owned[str(anchor)] = j
                            db.upsert_dictionary_keyword(anchor, item["case"]["entity_type"], "ENTITY")
                            db.upsert_dictionary_keyword(anchor, item["case"]["attribute"], "ATTRIBUTE")
                    for i, query in enumerate(items):
                        found = set()
                        words = {"ENTITY": set(), "ATTRIBUTE": set()}
                        for _, read in channels:
                            rows = db.get_semantic_keywords(query[read], report["threshold"], id_limit=20)
                            found.update(owned[str(row["dictionary_id"])] for row in rows)
                            bootstrap = service.bootstrap([query[read]])
                            for kind, values in bootstrap.items():
                                words[kind.value].update(values)
                        expected = {j for j, s in enumerate(scores[i]) if s > report["threshold"]}
                        if expected != found:
                            raise RuntimeError("Real SQL retrieval disagreed with vector simulation.")
                        expected_attributes = {CASES[j]["attribute"] for j in expected}
                        expected_entities = {CASES[j]["entity_type"] for j in expected}
                        if words["ATTRIBUTE"] != expected_attributes or words["ENTITY"] != expected_entities:
                            raise RuntimeError("Real bootstrap keywords disagreed with SQL retrieval.")
                        metrics["bootstrap"].append({"query": query["case"]["concept"],
                            "matched_concepts": [CASES[j]["concept"] for j in sorted(found)],
                            "words": {kind: sorted(values) for kind, values in words.items()}})
                    raise RollbackEvaluation()
            except RollbackEvaluation:
                pass
            report["strategies"][name] = metrics
        after = dictionary_state(db)
        redis_after = cache.dbsize()
        report["cleanup"] = {"before": before, "after": after, "dictionary_exactly_restored": before == after,
                             "redis_before": redis_before, "redis_after": redis_after}
        if before != after or redis_before != redis_after:
            raise RuntimeError("Original local state was not restored exactly.")
        output = ROOT / "log" / "dictionary_representation_evaluation.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"embedding_usage": report["embedding_usage"], "metrics": {
            name: {k: v for k, v in metrics.items() if k not in ("score_matrix", "bootstrap")}
            for name, metrics in report["strategies"].items()}, "cleanup": report["cleanup"]}, indent=2))
    finally:
        db.close()
        cache.close()


if __name__ == "__main__":
    # Never render connection exceptions or full configuration values.
    try:
        main()
    except Exception as exc:
        raise SystemExit("Representation evaluation failed (" + type(exc).__name__ + "). No credentials were printed.")
