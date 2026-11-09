"""Per-object discovery and bounded candidate delivery; no semantic word rules."""
import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import UUID
import pytest

from memory_fixtures import simple_entity, simple_fact, prepare_write
from test_memory_pipeline import LLM, MemoryDB, chat
from test_postgres_transactions import local_db
from tomte.engine.core.extraction import ExtractionMemorySchema
from tomte.engine.core.resolution import ResolutionResponse
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


@pytest.mark.parametrize("filter_enabled,limit", [(False, 10), (True, 20)])
def test_each_extracted_entity_search_uses_its_own_plain_text_vector(monkeypatch, filter_enabled, limit):
    class Model(LLM):
        def __init__(self):
            super().__init__()
            self.embedded = []

        def parse_structured(self, system_prompt, user_text, schema):
            assert schema is ExtractionMemorySchema
            return schema.model_validate({"relationships": [{
                "entity_id": f"tmp_person_{i}", "entity_name": name, "aliases": [alias],
                "entity_type": "person", "entity_domain": "relationships",
                "facts": [{"attribute_name": "employment", "source_chat_id": "m1",
                           "value": [{"field_name": "employer", "text": employer,
                                      "fact_confidence": "high"}]}],
            } for i, (name, alias, employer) in enumerate([
                ("Bob", "Robert", "Cedar Labs"), ("Alice", "Ali", "Harbour Hospital")])]})

        def embed_batch(self, texts):
            self.embedded.append(texts)
            return [[1.0, 0.0], [0.0, 1.0]]

    llm, db = Model(), MemoryDB()
    llm.supports_candidate_filter = filter_enabled
    db.get_resolution_candidates = Mock(return_value=[])
    workflow = ConsolidationWorkflow(db, llm)
    monkeypatch.setattr("tomte.engine.workflows.consolidation.align_terms", lambda es, *args:
                        SimpleNamespace(entities=es, evidence=[{} for _ in es], metrics={}))
    monkeypatch.setattr(workflow, "_reconcile_with_golden_records", lambda *args, **kwargs: [])
    monkeypatch.setattr(workflow, "_resolve_contextual_actions", lambda *args: [])
    monkeypatch.setattr(workflow, "_plan_fact_mutations", lambda *args: [])
    monkeypatch.setattr("tomte.engine.workflows.unresolved.prepare_unresolved_bindings",
                        lambda *args: ([], {}))
    messages = chat()
    messages[0].update(content="Bob works at Cedar Labs. Alice works at Harbour Hospital.", vector=[0.5, 0.5])
    assert workflow.run_consolidation_cycle("tenant", "user", messages)
    assert llm.embedded == [["Bob\nRobert\nCedar Labs", "Alice\nAli\nHarbour Hospital"]]
    calls = db.get_resolution_candidates.call_args_list
    assert len(calls) == 2
    assert [c.args[4] for c in calls] == [[1.0, 0.0], [0.0, 1.0]]
    assert [c.args[2] for c in calls] == [["bob", "robert"], ["alice", "ali"]]
    assert all(c.args[:2] == ("tenant", "user") and c.args[6] == limit for c in calls)


def test_tenth_candidate_reaches_identity_model_and_can_be_selected():
    rows = []
    for i in range(11):
        payload = simple_entity(f"Person {i}", facts=[simple_fact("detail", f"Person {i} details")])
        rows.append({"id": payload["entity_id"], "payload": payload,
                     **{k: payload[k] for k in ("entity_id", "entity_name", "entity_type", "entity_domain")}})
    incoming = {"entity_id": "tmp_person", "entity_name": "My colleague", "entity_type": "person",
                "entity_domain": "relationships", "aliases": [], "facts": []}
    llm = Mock()
    llm.parse_structured.return_value = ResolutionResponse.model_validate({"resolutions": [{
        "extracted_entity_id": "tmp_person", "kind": "MATCH_ONE",
        "targets": [{"db_id": rows[9]["id"]}], "reasoning": "Prescribed test target"}]})
    workflow = ConsolidationWorkflow(MemoryDB(), llm)
    operations = workflow._reconcile_with_golden_records("tenant", "user", [incoming], rows)
    request = json.loads(llm.parse_structured.call_args.kwargs["user_text"])
    assert len(request["CANDIDATES_BY_OBJECT"]["tmp_person"]) == 10
    assert rows[10]["id"] not in request["CANDIDATES_BY_ID"]
    assert [op["entity_id"] for op in operations] == [rows[9]["id"]]
    llm.parse_structured.return_value.resolutions[0].targets[0].db_id = UUID(rows[10]["id"])
    assert workflow._reconcile_with_golden_records("tenant", "user", [incoming], rows) == []


def test_similarity_beats_batch_name_and_ordinary_domain_priority(local_db):
    case = local_db
    query = [1.0, 0.0] + [0.0] * 1534
    weak = [0.1, 1.0] + [0.0] * 1534
    violin = simple_entity("My violin", kind="instrument", domain="assets_and_possessions",
                           facts=[simple_fact("ownership", "I own an Amati violin, which I call my fiddle.")])
    nature = simple_entity("Nature", domain="activities_and_skills", facts=[simple_fact("detail", "Enjoys nature.")])
    assert case.db.commit_graph_updates([
        prepare_write(case.tenant, case.users[0], violin, vector=query),
        prepare_write(case.tenant, case.users[0], nature, vector=weak)])
    rows = case.db.get_resolution_candidates(case.tenant, case.users[0], ["my fiddle"],
        "activities_and_skills", query, "Nature Nature Nature. My fiddle needs a new bridge.", 10)
    assert [r["entity_id"] for r in rows] == [violin["entity_id"], nature["entity_id"]]
    assert rows[0]["similarity"] > rows[1]["similarity"]
    # Exact names still retrieve a weak vector; they never authorize a match.
    exact = case.db.get_resolution_candidates(case.tenant, case.users[0], ["nature"],
        "assets_and_possessions", query, "", 10)
    assert exact[0]["entity_id"] == nature["entity_id"]
    assert case.db.get_resolution_candidates(case.tenant, case.users[1], [], "", query, "", 10) == []


def test_filter_candidate_limit_reaches_real_database_and_remains_scoped(local_db):
    case = local_db
    workflow = ConsolidationWorkflow(case.db, SimpleNamespace(supports_candidate_filter=True))
    limit = workflow._identity_candidate_limit()
    assert limit == 20
    vector = [1.0] + [0.0] * 1535
    args = (case.tenant, case.users[0], [], "relationships", vector, "")
    # New users must pass the same boundary before any candidates exist.
    assert case.db.get_resolution_candidates(*args, limit=limit) == []
    payloads = [simple_entity(f"Person {i}", facts=[simple_fact("detail", f"Person {i}")])
                for i in range(21)]
    foreign = simple_entity("Other user's person", facts=[simple_fact("detail", "Other user")])
    assert case.db.commit_graph_updates([
        *[prepare_write(case.tenant, case.users[0], p, vector=vector) for p in payloads],
        prepare_write(case.tenant, case.users[1], foreign, vector=vector),
    ])
    rows = case.db.get_resolution_candidates(*args, limit=limit)
    assert len(rows) == 20
    assert len({r["entity_id"] for r in rows}) == 20
    assert {r["entity_id"] for r in rows} <= {p["entity_id"] for p in payloads}
    assert all(r["tenant_id"] == case.tenant and r["user_id"] == case.users[0] for r in rows)
    assert len(case.db.get_resolution_candidates(*args)) == 10
    workflow.llm.supports_candidate_filter = False
    assert workflow._identity_candidate_limit() == 10
    assert len(case.db.get_resolution_candidates(*args, limit=workflow._identity_candidate_limit())) == 10
    assert case.db.get_resolution_candidates(case.tenant + "-other", case.users[0], [], "", vector, "", limit) == []
    for invalid in (0, 21, True, 1.5):
        with pytest.raises(ValueError):
            case.db.get_resolution_candidates(*args, limit=invalid)
