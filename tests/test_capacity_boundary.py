from unittest.mock import Mock
import json
import pytest

from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from tomte.engine.core.capacity import CapacityExceeded, check_capacity, check_provider_capacity
from tomte.engine.core.config import settings, TomteSettings


def batch(text):
    return [{'tenant_id':'t','user_id':'u','session_id':'s','item_id':'source',
             'timestamp':1700000000, 'role':'user','content':text,'vector':[1.]}]


def test_large_batch_reaches_extraction_without_splitting_or_truncation(monkeypatch):
    from tomte.engine.core.extraction import ExtractionMemorySchema
    # Exercise the shipped default independently of local configuration overrides.
    monkeypatch.setattr(settings, 'max_model_input_bytes',
                        TomteSettings.model_fields['max_model_input_bytes'].default)
    text = 'Synthetic work update. ' * 40
    items = [{**batch(text)[0], 'item_id': f'source-{i}', 'timestamp': 1700000000 + i * 60,
              'role': 'user' if i % 2 == 0 else 'assistant'} for i in range(200)]
    assert all(len(item['content']) <= settings.max_input_length for item in items)
    db, llm = Mock(), Mock()
    db.is_consolidation_processed.return_value = False
    db.get_resolution_candidates.return_value = []
    db.get_unresolved_memories.return_value = []

    def junk(chat_log):
        check_provider_capacity('is_junk', {'chat_log': chat_log})
        return False

    def extraction(**arguments):
        assert arguments['schema'] is ExtractionMemorySchema
        check_provider_capacity('parse_structured', arguments)
        current = json.loads(arguments['user_text'])['CURRENT_BATCH']
        assert len(current) == len(items)
        assert [(m['item_id'], m['role'], m['content']) for m in current] == [
            (m['item_id'], m['role'], m['content']) for m in items]
        return ExtractionMemorySchema()

    llm.is_junk.side_effect = junk
    llm.parse_structured.side_effect = extraction
    workflow = ConsolidationWorkflow(db=db, llm=llm)
    workflow._commit_refined_operations = Mock()
    assert workflow.run_consolidation_cycle('t', 'u', items)
    llm.parse_structured.assert_called_once()
    assert settings.max_model_input_bytes == 327680


def test_oversized_frozen_input_stops_before_any_model_call():
    db, llm = Mock(), Mock()
    db.is_consolidation_processed.return_value = False
    workflow = ConsolidationWorkflow(db=db,llm=llm)
    workflow._commit_refined_operations = Mock()
    with pytest.raises(CapacityExceeded, match='capacity'):
        workflow.run_consolidation_cycle('t','u',batch('x'*(settings.max_model_input_bytes + 1)))
    assert llm.mock_calls == []
    db.claim_consolidation_batch.assert_not_called()


def test_normal_frozen_input_reaches_the_same_workflow():
    from tomte.engine.core.extraction import ExtractionMemorySchema
    db, llm = Mock(), Mock()
    db.is_consolidation_processed.return_value = False
    db.get_resolution_candidates.return_value = []
    db.get_unresolved_memories.return_value = []
    llm.is_junk.side_effect = AssertionError('Separate junk filtering is disabled')
    llm.parse_structured.return_value = ExtractionMemorySchema()
    workflow = ConsolidationWorkflow(db=db, llm=llm)
    workflow._commit_refined_operations = Mock()
    items = batch('hello')
    assert workflow.run_consolidation_cycle('t','u',items)
    llm.is_junk.assert_not_called()
    llm.parse_structured.assert_called_once()
    assert 'hello' in llm.parse_structured.call_args.kwargs['user_text']
    workflow._commit_refined_operations.assert_called_once()
    assert items == batch('hello')


def test_native_assembled_schema_prompt_is_checked_before_dispatch(monkeypatch):
    from tomte.engine.providers.openai_provider import OpenAILLMProvider
    from tomte.engine.core.assertions import MutationPlanResponse
    import tomte.engine.providers.openai_provider as module
    provider = object.__new__(OpenAILLMProvider)
    agent = Mock()
    monkeypatch.setattr(module, 'Agent', agent)
    with pytest.raises(CapacityExceeded):
        provider.parse_structured('synthetic', '字'*(settings.max_model_input_bytes // 3 + 1), MutationPlanResponse)
    agent.assert_not_called()


def test_capacity_failure_is_permanent_and_next_session_progresses():
    from test_managed_worker import Harness
    from tomte.engine.workers.control import transient
    h = Harness()
    h.cache.add()
    h.cache.queues[('t','u','s')][0]['content'] = 'x'*(settings.max_model_input_bytes + 1)
    h.cache.add(session='later')
    h.behavior = lambda h,t,u,items:check_capacity(items[0]['content'])
    try:
        h.step()
        assert h.cache.terminal[-1][3] == 'skipped'
        assert h.cache.terminal[-1][2]['reason'] == 'capacity_exceeded'
        assert h.cache.terminal[-1][2]['attempts'] == 1
        h.step()
        assert h.calls == [('t','u','s'),('t','u','later')]
        assert h.cache.terminal[-1][3] == 'committed'
        assert not transient(CapacityExceeded('synthetic'))
    finally:
        assert h.worker.stop()


def test_capacity_counts_utf8_bytes_across_parts_at_limit():
    prefix = 'x' * (settings.max_model_input_bytes - len('é'.encode('utf-8')))
    check_capacity(prefix, 'é')
    with pytest.raises(CapacityExceeded):
        check_capacity(prefix, 'é', 'x')
