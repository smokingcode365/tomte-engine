from types import SimpleNamespace
from unittest.mock import Mock


def test_reconciliation_returns_accumulated_operations(monkeypatch):
    from tomte.engine import Tomte
    from tomte.engine.workflows import consolidation

    llm = Mock()
    from tomte.engine.core.resolution import ResolutionResponse
    llm.parse_structured.return_value = ResolutionResponse.model_validate({"resolutions":[{
        "extracted_entity_id":"tmp_bicycle","kind":"NEW","explicitly_new":True,
        "new_entity_evidence":"I bought a bicycle.","reasoning":"Explicit new bicycle"}]})
    monkeypatch.setattr(Tomte, "_db", Mock())
    monkeypatch.setattr(Tomte, "_llm", llm)
    monkeypatch.setattr(consolidation, "get_dictionary_service", lambda: Mock())
    workflow = consolidation.ConsolidationWorkflow()
    entity = {
        "entity_id": "tmp_bicycle",
        "entity_name": "Bicycle",
        "entity_domain": "assets_and_possessions",
        "entity_type": "bicycle",
        "facts": [],
    }

    operations = workflow._reconcile_with_golden_records(
        "test-tenant", "test-user", [entity], [], current_batch=[{"role":"user","content":"I bought a bicycle."}]
    )

    assert isinstance(operations, list)
    assert len(operations) == 1
    assert operations[0]["mode"] == "INSERT"
    assert operations[0]["entity_id"].startswith("entity_")
    assert operations[0]["reserved_id"]
    assert operations[0]["tenant_id"] == "test-tenant"
    assert operations[0]["user_id"] == "test-user"
    llm.parse_structured.assert_called_once()
