"""The answering model owns information requests before and after retrieval."""
import json
from copy import deepcopy
from uuid import uuid4

from test_answer_question import answering, decision, plan
from test_relevant_entities_api import retrieval
from memory_fixtures import simple_fact
from tomte.engine.core.answering import AnswerDecision, RequiredInformation


def test_answer_analyses_question_before_first_retrieval(answering):
    r = answering
    inputs = []

    def model(**kwargs):
        payload = json.loads(kwargs['user_text'])
        inputs.append(payload)
        if len(inputs) == 1:
            assert kwargs['schema'] is RequiredInformation
            assert 'memory' not in payload
            r.model.embed_text.assert_not_called()
            r.db.get_raw_semantic_context.assert_not_called()
            return plan('When did I get Lucky?')
        return decision('The available memories do not establish when you got Lucky.')

    r.model.parse_structured.side_effect = model
    r.client.answer_question('user-a', 'What pet did I get before Lucky?')
    assert len(inputs) == 2
    r.model.embed_text.assert_called_once_with('When did I get Lucky?')


def test_dependent_request_carries_date_and_final_answer_sees_both_results(answering):
    r = answering
    lucky, coco = deepcopy(r.row), deepcopy(r.row)
    lucky.update(entity_name='Lucky', entity_type='dog', entity_domain='relationships',
                 facts=[simple_fact('acquisition', 'Got Lucky in March 2022.')])
    coco.update(entity_id=str(uuid4()), entity_name='Coco', entity_type='cat', entity_domain='relationships',
                facts=[simple_fact('acquisition', 'Got Coco in June 2020.')])
    r.db.get_raw_semantic_context.side_effect = [[lucky], [coco]]
    calls = []

    def model(**kwargs):
        payload = json.loads(kwargs['user_text'])
        calls.append(payload)
        if len(calls) == 1:
            return plan('When did I get Lucky?')
        if len(calls) == 2:
            assert 'Got Lucky in March 2022.' in payload['memory']
            return decision(action='retrieve', missing='Other pets acquired before March 2022 and their acquisition dates')
        if len(calls) == 3:
            assert kwargs['schema'] is RequiredInformation
            assert 'Got Lucky in March 2022.' in payload['retrieval_results'][-1]
            assert payload['suggestions'] == ['Other pets acquired before March 2022 and their acquisition dates']
            return plan('Other pets acquired before March 2022 and their acquisition dates')
        assert 'Got Lucky in March 2022.' in payload['memory']
        assert 'Got Coco in June 2020.' in payload['memory']
        return decision('You got Coco before Lucky.')

    r.model.parse_structured.side_effect = model
    assert r.client.answer_question('user-a', 'What pet did I get before Lucky?') == 'You got Coco before Lucky.'
    assert [c.args[0] for c in r.model.embed_text.call_args_list] == [
        'When did I get Lucky?', 'Other pets acquired before March 2022 and their acquisition dates']
    assert len(calls) == 4
    r.cache.get_pending_search_snapshot.assert_called_once()
    r.model.embed_batch.assert_called_once()


def test_retrieval_returns_only_current_pass_evidence_when_reused(answering):
    from tomte.engine.workflows.retrieving import MemoryRetrievalWorkflow
    from tomte.engine.workers.control import WorkControl
    r = answering
    first, second = deepcopy(r.row), deepcopy(r.row)
    first['facts'] = [simple_fact('first', 'First request evidence only')]
    second.update(entity_id=str(uuid4()), facts=[simple_fact('second', 'Second request evidence only')])
    r.db.get_raw_semantic_context.side_effect = [[first], [second]]
    workflow = MemoryRetrievalWorkflow('tenant-a', 'user-a', r.db, r.model, r.cache, WorkControl())
    assert 'First request evidence only' in workflow.run('First information request')
    result = workflow.run('Second information request')
    assert 'Second request evidence only' in result and 'First request evidence only' not in result
    r.cache.get_pending_search_snapshot.assert_called_once()
    r.model.embed_batch.assert_called_once()
    r.model.parse_structured.assert_not_called()
