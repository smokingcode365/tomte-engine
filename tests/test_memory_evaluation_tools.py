"""Offline compatibility checks for evaluation fixtures and independent scoring."""
from copy import deepcopy
import importlib
from pathlib import Path

import pytest

from tomte.engine.core.memory import Entity
from test_postgres_transactions import local_db


@pytest.fixture
def evaluation(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("memory_eval_support")


def test_evaluation_entrypoints_import_without_retired_runtime_helpers(evaluation):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    for path in sorted(scripts.glob("evaluate_*.py")):
        importlib.import_module(path.stem)
    importlib.import_module("compare_terminology_models")


def test_all_retrieval_fixture_variants_have_nonempty_strict_memory(evaluation):
    retrieval = importlib.import_module("evaluate_memory_retrieval")
    scenarios, queries = retrieval.dataset()
    assert queries and "size_0" not in scenarios
    for bank in scenarios.values():
        for payload in bank.values():
            assert Entity.model_validate(payload).facts
    history = scenarios["history"]["tesla"]["facts"]
    assert sum(len(f["history_values"]) for f in history) == 2
    assert len({f["fact_id"] for f in history}) == len(history)
    for module in ("evaluate_entity_activation", "evaluate_retrieval_stack"):
        source = importlib.import_module(module)
        bank = (source.dataset() if hasattr(source, "dataset") else source.build_dataset())[0]
        for payload in bank.values():
            Entity.model_validate(payload)


def test_cross_batch_candidate_fixtures_use_canonical_uuid_and_structured_items(evaluation):
    cases, _ = importlib.import_module("evaluate_cross_batch_context").datasets()
    for case in cases:
        for row in case["candidates"]:
            assert row["id"] == row["entity_id"] == Entity.model_validate(row["payload"]).entity_id


def test_evaluation_sql_oracle_includes_history_and_preserves_scope(evaluation, local_db):
    bank = {"bob": evaluation.entity("bob", "Bob", "relationships", "person", [], [("employment", "Cedar Labs")])}
    payload = bank["bob"]
    old = deepcopy(payload["facts"][0]["value"])
    old["detail"]["text"] = "Pine Labs"
    payload["facts"][0]["history_values"] = [{"value": old, "_archived_at": "2026-10-09T00:00:00Z"}]
    vectors = {text: [float(i == (1 if " history " in text else 0)) for i in range(1536)] for text in evaluation.search_texts(payload)}
    first, second = local_db.users[:2]
    writes = [evaluation.fixture_write(local_db.tenant, user, payload, vectors) for user in (first, second)]
    assert writes[0]["entity_id"] != writes[1]["entity_id"]
    assert local_db.db.commit_graph_updates(writes)
    query = [float(i == 1) for i in range(1536)]
    rows = evaluation.database_scores(local_db.db, local_db.tenant, first, query)
    assert len(rows) == 1 and rows[0]["entity_id"] == writes[0]["entity_id"]
    assert float(rows[0]["cosine"]) == pytest.approx(evaluation.similarity(query, payload, vectors)) == 1
    assert evaluation.label_map(local_db.tenant, first, bank)[rows[0]["entity_id"]] == "bob"
    assert evaluation.database_scores(local_db.db, local_db.tenant + "-absent", first, query) == []


def test_real_world_scorer_uses_current_items_and_embedded_history(evaluation):
    harness = importlib.import_module("evaluate_real_world_conversations")
    payload = evaluation.entity("car", "Tesla", "assets_and_possessions", "vehicle", [], [("paint_color", "blue")])
    payload["facts"][0]["history_values"] = [{"_archived_at": "2026-10-09T00:00:00Z", "value": evaluation.fact("paint_color", "red")["value"]}]
    blue = {"entity": "tesla", "attribute_contains": ["color"], "values": ["blue"], "location": "current"}
    red = {**blue, "values": ["red"], "location": "history"}
    truth = {"entities": {"tesla": {"names": ["Tesla"]}}, "expected_final_facts": [blue, red],
             "forbidden_active": [{**red, "location": "current"}], "expected_aliases": {}}
    score = harness.evaluate_state([{"entity_name": "Tesla", "payload": payload}], truth, {})
    assert score["checklist_recall"] == 1 and score["forbidden_active"] == []
    assert score["unsupported_provenance"] is None  # Absence of stored sources is not evidence of correctness.
