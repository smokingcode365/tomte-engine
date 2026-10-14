"""Small live prototype evaluation, with prescribed raw extraction (not E2E).

Run from tomte-engine using its Python environment. Calls configured embeddings
and terminology model, never extraction/identity/mutation/consolidation/daemon.
Requires an empty local dictionary to isolate offered vocabulary. Writes and
removes only its own temporary dictionary anchors. Results remain in ignored log/.
"""
from collections import Counter
from copy import deepcopy
import json
import logging
import os
from pathlib import Path
from time import perf_counter
from urllib.parse import urlsplit

from dotenv import load_dotenv
import httpx
import redis

from evaluate_dictionary_representations import dictionary_state
from tomte.engine.core.config import settings
from tomte.engine.core.extraction import ExtractionMemorySchema, resolve_extraction
from tomte.engine.dictionary.alignment import align_terms, query_text
from tomte.engine.providers import openai_provider
from tomte.engine.providers.postgres_provider import PostgresProvider

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "log" / "terminology_alignment_evaluation.json"
# case, raw type, attribute, value, source, offered term, keyword type, domain
CASES = [
    ("A", "car", "exterior_vehicle_colour", "dark blue", "I painted the exterior of my Tesla dark blue.", "paint_color", "ATTRIBUTE", "assets_and_possessions"),
    ("B", "career", "position", "Engineering Manager", "My current job title is Engineering Manager.", "job_title", "ATTRIBUTE", "activities_and_skills"),
    ("C", "automobile", None, None, "I own an ordinary automobile, a Tesla.", "car", "ENTITY", "assets_and_possessions"),
    ("D", "car", None, None, "I own a car, a Tesla.", "vehicle", "ENTITY", "assets_and_possessions"),
    ("E", "car", "interior_color", "black", "The seats and interior of my Tesla are black.", "paint_color", "ATTRIBUTE", "assets_and_possessions"),
    ("F", "car", "sale_status", "completed", "The paperwork for the sale was completed.", "ownership_status", "ATTRIBUTE", "assets_and_possessions"),
    ("G", "person", "current_location", "Vancouver", "I am temporarily visiting Vancouver this week.", "current_home_address", "ATTRIBUTE", "identity_and_demographics"),
    ("H", "career", "professional_field", "software engineering", "My professional field is software engineering, not a job title.", "job_title", "ATTRIBUTE", "activities_and_skills"),
]


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError("Run from tomte-engine")
    load_dotenv(ROOT / ".env", override=False)
    for name in ("OPENAI_API_KEY", "POSTGRES_URL", "REDIS_URL"):
        if not os.environ.get(name):
            raise RuntimeError("Required configuration is missing")
    if any(urlsplit(os.environ[n]).hostname not in ("localhost", "127.0.0.1", "::1")
           for n in ("POSTGRES_URL", "REDIS_URL")):
        raise RuntimeError("Only local infrastructure is allowed")
    # No SDK request headers or exception bodies can enter logs.
    for name in ("openai", "httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    db = PostgresProvider(os.environ["POSTGRES_URL"])
    cache = redis.Redis.from_url(os.environ["REDIS_URL"])
    llm = openai_provider.OpenAILLMProvider(os.environ["OPENAI_API_KEY"])
    before = dictionary_state(db)
    redis_before = cache.dbsize()
    if any(v["count"] for v in before.values()):
        db.close()
        cache.close()
        raise RuntimeError("Isolated evaluation requires an empty dictionary; no data was removed")
    api = Counter()
    observed_models = set()
    results, usage, diagnostics, created = [], [], [], []
    send = httpx.AsyncClient.send
    run = openai_provider._run_structured_agent
    debug = openai_provider._debug_metadata
    async def count_send(client, request, *args, **kwargs):
        path = request.url.path
        if path.endswith("/embeddings"):
            api["embeddings"] += 1
        elif path.endswith("/responses"):
            api["responses"] += 1
            observed_models.add(json.loads(request.content)["model"])
        else:
            raise RuntimeError("Unexpected external route in terminology evaluation")
        response = await send(client, request, *args, **kwargs)
        if path.endswith("/embeddings"):
            api["embedding_input_tokens"] += response.json().get("usage", {}).get("total_tokens", 0)
        return response
    def record_run(*args, **kwargs):
        result = run(*args, **kwargs)
        total = result.usage()
        offered = json.loads(args[3])["entities"]
        targets = {(e["entity_handle"], None): e["candidates"] for e in offered}
        targets.update({(e["entity_handle"], f["fact_handle"]): f["candidates"]
                        for e in offered for f in e["facts"]})
        output = result.output.model_dump(mode="json")
        unsupported = sum(d["decision"] == "REUSE" and not any(
            c["candidate_handle"] == d["candidate_handle"]
            for c in targets.get((d["entity_handle"], d.get("fact_handle")), []))
            for group in output.values() for d in group)
        record = {k: getattr(total, k, None) for k in ("requests", "input_tokens", "output_tokens")}
        record.update(structured_result=output, offered_candidates=offered,
                      unsupported_candidate_count=unsupported)
        usage.append(record)
        return result
    report = {"model": None, "embedding_model": settings.openai.embed_model,
              "vector_dimension": settings.openai.vector_dimension, "cases": results}
    httpx.AsyncClient.send = count_send
    openai_provider._run_structured_agent = record_run
    openai_provider._debug_metadata = lambda metadata: diagnostics.append(deepcopy(metadata))
    try:
        seed_texts = [query_text(c[7], c[5] if c[6] == "ENTITY" else c[1],
                                None if c[6] == "ENTITY" else c[5]) for c in CASES]
        seed_vectors = llm.embed_batch(seed_texts)
        for repeat in range(1, 4):
            for case, seed_vector in zip(CASES, seed_vectors):
                label, entity_type, attribute, value, source, candidate, kind, domain = case
                with db.transaction():
                    anchor = db.create_dictionary_anchor(seed_vector)
                    created.append(anchor)
                    db.upsert_dictionary_keyword(anchor, candidate, kind)
                batch = [{"item_id": f"synthetic-{label}-{repeat}", "timestamp": 1791547200 + repeat,
                          "content": source, "role": "user"}]
                wire = ExtractionMemorySchema.model_validate({domain: [{
                    "entity_id": "syn_01", "entity_name": "Tesla" if domain == "assets_and_possessions" else "Synthetic person career",
                    "entity_domain": domain, "entity_type": entity_type,
                    "facts": [] if attribute is None else [{"attribute_name": attribute,
                        "value": value, "confidence": 1., "source_chat_id": batch[0]["item_id"]}]}]})
                raw = [e.model_dump(mode="json") for e in getattr(resolve_extraction(wire, batch), domain)]
                usage_start, diagnostic_start = len(usage), len(diagnostics)
                counts_before = api.copy()
                started = perf_counter()
                result = align_terms(raw, batch, llm, db)
                elapsed = perf_counter() - started
                field = result.entities[0]["entity_type"] if attribute is None else result.entities[0]["facts"][0]["attribute_name"]
                evidence = result.evidence[0]
                target = next((e for e in evidence if e["keyword_type"] == kind), None)
                reused = bool(target and target["dictionary_id"])
                expected = label in "ABC"
                restored = deepcopy(result.entities)
                restored[0]["entity_type"] = raw[0]["entity_type"]
                if attribute is not None:
                    restored[0]["facts"][0]["attribute_name"] = attribute
                assert restored == raw, "Non-vocabulary fields changed"
                records = diagnostics[diagnostic_start:]
                results.append({"case": label, "repeat": repeat, "raw_type": entity_type,
                    "raw_attribute": attribute, "value": value, "source": source, "candidate": candidate,
                    "result_term": field, "reused": reused, "safe_reuse_case": expected,
                    "false_reuse": reused and not expected,
                    "value_changed": False, "raw_fallback": result.metrics["raw_fallback"],
                    "offered_candidate_used": bool(target and target["dictionary_id"] == str(anchor)),
                    "metrics": result.metrics, "latency_seconds": elapsed,
                    "api_requests": dict(api - counts_before), "usage": usage[usage_start:],
                    "structured_retry_count": sum(r.get("event") == "structured_output_retry" for r in records)})
                print(f"Case {label} run {repeat}: {'REUSE' if reused else 'KEEP_RAW'}; fallback={result.metrics['raw_fallback']}", flush=True)
                with db.transaction():
                    with db.conn.cursor() as cursor:
                        cursor.execute("DELETE FROM tomte_semantic_anchors WHERE dictionary_id=%s", (anchor,))
        safe = [r for r in results if r["safe_reuse_case"]]
        keep = [r for r in results if not r["safe_reuse_case"]]
        reused = [r for r in results if r["reused"]]
        report["summary"] = {"runs": len(results), "safe_reuses": sum(r["reused"] for r in safe),
            "safe_reuse_opportunities": len(safe),
            "safe_reuse_precision": sum(not r["false_reuse"] for r in reused) / len(reused) if reused else None,
            "keep_raw_correct": sum(not r["reused"] for r in keep), "keep_raw_cases": len(keep),
            "false_reuse_count": sum(r["false_reuse"] for r in results), "value_change_count": 0,
            "unsupported_candidate_count": sum(u["unsupported_candidate_count"] for u in usage),
            "structured_retries": sum(r["structured_retry_count"] for r in results),
            "raw_fallback_count": sum(r["raw_fallback"] for r in results),
            "api_requests": dict(api), "terminology_input_tokens": sum(u["input_tokens"] or 0 for u in usage),
            "terminology_output_tokens": sum(u["output_tokens"] or 0 for u in usage)}
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        raise RuntimeError("Live terminology evaluation failed; see sanitized report") from None
    finally:
        report["observed_models"] = sorted(observed_models)
        report["model"] = next(iter(observed_models)) if len(observed_models) == 1 else None
        httpx.AsyncClient.send = send
        openai_provider._run_structured_agent = run
        openai_provider._debug_metadata = debug
        try:
            with db.transaction():
                with db.conn.cursor() as cursor:
                    for anchor in created:
                        cursor.execute("DELETE FROM tomte_semantic_anchors WHERE dictionary_id=%s", (anchor,))
            report["dictionary_restored"] = dictionary_state(db) == before
            report["redis_unchanged"] = cache.dbsize() == redis_before
        finally:
            OUTPUT.parent.mkdir(exist_ok=True)
            OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
            cache.close()
            db.close()
    assert report["dictionary_restored"] and report["redis_unchanged"]
    print(json.dumps(report["summary"], sort_keys=True))


if __name__ == "__main__":
    main()
