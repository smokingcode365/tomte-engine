from types import SimpleNamespace
from unittest.mock import Mock


def test_reconciliation_returns_accumulated_operations(monkeypatch):
    from tomte.engine import Tomte
    from tomte.engine.workflows import consolidation

    llm = Mock()
    llm.parse_structured.return_value = SimpleNamespace(mappings=[])
    monkeypatch.setattr(Tomte, "_db", Mock())
    monkeypatch.setattr(Tomte, "_llm", llm)
    monkeypatch.setattr(consolidation, "get_dictionary_service", lambda: Mock())
    workflow = consolidation.ConsolidationWorkflow()
    entity = {
        "entity_id": "veh_01",
        "entity_name": "Bicycle",
        "entity_domain": "assets_and_possessions",
        "entity_type": "bicycle",
        "facts": [],
    }

    operations = workflow._reconcile_with_golden_records(
        "test-tenant", "test-user", [entity], []
    )

    assert isinstance(operations, list)
    assert len(operations) == 1
    assert operations[0]["mode"] == "INSERT"
    assert operations[0]["entity_id"] == "veh_01"
    assert operations[0]["tenant_id"] == "test-tenant"
    assert operations[0]["user_id"] == "test-user"
    llm.parse_structured.assert_called_once()
