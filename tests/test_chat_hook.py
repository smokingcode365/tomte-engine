from types import SimpleNamespace


def test_chat_hook_constructs_cognitive_processor(monkeypatch):
    from tomte.engine import Tomte
    from tomte.engine.utils.cognitive import CognitiveProcessor
    from tomte.engine.workflows import chat_hook

    db, cache, llm, security = (SimpleNamespace() for _ in range(4))
    monkeypatch.setattr(Tomte, "_db", db)
    monkeypatch.setattr(Tomte, "_cache", cache)
    monkeypatch.setattr(Tomte, "_llm", llm)
    monkeypatch.setattr(chat_hook, "get_security_guard", lambda: security)

    workflow = chat_hook.ChatHookWorkflow()

    assert workflow.db is db
    assert workflow.cache is cache
    assert workflow.llm is llm
    assert workflow.security is security
    assert isinstance(workflow.cog, CognitiveProcessor)
    assert workflow.cog.merge_entities(
        [{"entity_id": "veh_01", "entity_name": "Old bicycle"}],
        [{"entity_id": "veh_01", "entity_name": "New bicycle"}],
    ) == [{"entity_id": "veh_01", "entity_name": "New bicycle"}]
