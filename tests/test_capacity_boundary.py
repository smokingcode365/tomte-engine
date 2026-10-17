from unittest.mock import Mock
import pytest

from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from tomte.engine.core.capacity import CapacityExceeded, check_capacity


def batch(text):
    return [{'tenant_id':'t','user_id':'u','session_id':'s','item_id':'source',
             'timestamp':1700000000, 'role':'user','content':text,'vector':[1.]}]


def test_oversized_frozen_input_stops_before_any_model_call():
    db, llm = Mock(), Mock()
    db.is_consolidation_processed.return_value = False
    workflow = ConsolidationWorkflow(db=db,llm=llm)
    workflow._commit_refined_operations = Mock()
    with pytest.raises(CapacityExceeded, match='capacity'):
        workflow.run_consolidation_cycle('t','u',batch('x'*70000))
    assert llm.mock_calls == []
    db.claim_consolidation_batch.assert_not_called()


def test_normal_frozen_input_reaches_the_same_workflow():
    db, llm = Mock(), Mock()
    db.is_consolidation_processed.return_value = False
    llm.is_junk.return_value = True
    workflow = ConsolidationWorkflow(db=db, llm=llm)
    workflow._commit_refined_operations = Mock()
    items = batch('hello')
    assert workflow.run_consolidation_cycle('t','u',items)
    llm.is_junk.assert_called_once()
    assert 'hello' in llm.is_junk.call_args.args[0]
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
        provider.parse_structured('synthetic', '字'*23000, MutationPlanResponse)
    agent.assert_not_called()


def test_capacity_failure_is_permanent_and_next_session_progresses():
    from test_managed_worker import Harness
    from tomte.engine.workers.control import transient
    h = Harness()
    h.cache.add()
    h.cache.queues[('t','u','s')][0]['content'] = 'x'*70000
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
