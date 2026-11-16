"""Caller-supplied owner context travels with one scoped read, never as identity."""
from copy import deepcopy
import json

import pytest

from test_answer_facts import fact_answering, judgment, response
from test_answer_question import answering, decision, model_inputs, plan
from test_relevant_entities_api import retrieval
from tomte.engine.exceptions import TomteInvalidInputError, TomteSecurityViolationError


def test_named_owner_reaches_planner_filter_and_answer_without_rewriting_question(fact_answering):
    r = fact_answering
    question = "What are the breeds of Audrey's dogs?"
    before = deepcopy(r.packets)
    r.model.parse_structured.side_effect = [plan("Audrey's dog breeds"), judgment(), decision('Four breeds.')]

    assert r.client.answer_question('user-a', question, memory_owner_name='Audrey') == 'Four breeds.'

    inputs = model_inputs(r)
    assert len(inputs) == 3
    assert all(p['memory_owner_name'] == 'Audrey' for p in inputs)
    assert inputs[0]['question'] == inputs[-1]['question'] == question
    assert inputs[1]['information_needed'] == "Audrey's dog breeds"
    assert r.packets == before
    assert r.db.get_answer_facts.call_args.args[:2] == ('tenant-a', 'user-a')


def test_named_owner_stays_available_across_follow_up_searches(fact_answering):
    r = fact_answering
    r.model.parse_structured.side_effect = [plan('first'), judgment(),
        decision('Partial', action='retrieve', missing='more'), plan('second'), judgment(), decision('Complete')]
    assert r.client.answer_question('user-a', 'Question', memory_owner_name='Mina') == 'Complete'
    assert all(p['memory_owner_name'] == 'Mina' for p in model_inputs(r))


def test_standalone_retrieval_returns_owner_context_without_storing_or_rewriting_it(fact_answering):
    r = fact_answering
    owner = '安娜 "Ana"'
    r.model.parse_structured.side_effect = [judgment()]
    result = r.client.retrieve_memories('user-a', 'Dog breeds', memory_owner_name=owner)
    assert model_inputs(r)[0]['memory_owner_name'] == owner
    assert result.startswith('Memory owner: ' + json.dumps(owner, ensure_ascii=False) + '\n')
    assert r.model.parse_structured.call_count == 1
    assert r.db.get_answer_facts.call_args.args[:2] == ('tenant-a', 'user-a')


def test_owner_context_is_not_inferred_or_reused_across_calls(fact_answering):
    r = fact_answering
    r.model.parse_structured.side_effect = [plan('Dog breeds'), judgment(), decision('Known'),
                                           plan('Dog breeds'), judgment(), decision('Uncertain')]
    r.client.answer_question('user-a', 'Dog breeds?', memory_owner_name='Mina')
    r.client.answer_question('user-a', "What are Audrey's dogs' breeds?")
    assert all(p['memory_owner_name'] == 'Mina' for p in model_inputs(r)[:3])
    assert all('memory_owner_name' not in p for p in model_inputs(r)[3:])


@pytest.mark.parametrize('method', ['answer_question', 'retrieve_memories'])
@pytest.mark.parametrize('owner', ['', '   ', 123, {'name': 'Audrey'}, 'bad\x00name', 'x' * 257])
def test_invalid_owner_context_is_rejected_before_services_open(fact_answering, method, owner):
    r = fact_answering
    with pytest.raises(TomteInvalidInputError):
        getattr(r.client, method)('user-a', 'Question', memory_owner_name=owner)
    r.context.assert_not_called()
    r.model.parse_structured.assert_not_called()


def test_owner_display_name_never_overrides_database_scope(fact_answering):
    r = fact_answering
    rows = response(r.packets)
    rows['facts'][0]['user_id'] = 'foreign-user'
    r.db.get_answer_facts.return_value = rows
    with pytest.raises(TomteSecurityViolationError):
        r.client.answer_question('user-a', 'Dog breeds', memory_owner_name='Audrey')
    assert r.model.parse_structured.call_count == 1
