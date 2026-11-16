"""Search text and filtering intent have different responsibilities."""
from unittest.mock import Mock
import json
import pytest
from pydantic import ValidationError

from tomte.engine.workflows.retrieving import MemoryRetrievalWorkflow
from tomte.engine.workflows.search_request import prepare_embedding_query, SearchEmbeddingQuery
from tomte.engine.workers.control import WorkControl


def test_embedding_uses_model_query_but_reader_keeps_exclusion():
    request = 'Outdoor activities the user has done or considered besides hiking'
    positive = 'Outdoor activities the user has done or considered'
    llm, db = Mock(), Mock()
    llm.parse_structured.return_value = {'embedding_text': positive}
    db.supports_answer_fact_search = True
    workflow = MemoryRetrievalWorkflow('tenant', 'user', db, llm, Mock(), WorkControl())
    workflow.pending = Mock(snapshot_at='2026-11-12T00:00:00Z')
    workflow._read_facts = Mock()
    workflow.retrieve_once(request)
    llm.embed_text.assert_called_once_with(positive)
    assert workflow._read_facts.call_args.args[0] == request
    assert json.loads(llm.parse_structured.call_args.kwargs['user_text']) == {'information_needed': request}


def test_preparation_is_reused_for_identical_request_only():
    llm, db = Mock(), Mock()
    llm.parse_structured.side_effect = [{'embedding_text': 'Activities'}, {'embedding_text': 'Games'}]
    db.supports_answer_fact_search = True
    workflow = MemoryRetrievalWorkflow('tenant', 'user', db, llm, Mock(), WorkControl())
    workflow.pending = Mock(snapshot_at='2026-11-12T00:00:00Z')
    workflow._read_facts = Mock()
    for request in ('Activities except hiking', 'Activities except hiking', 'Games except chess'):
        workflow.retrieve_once(request)
    assert llm.parse_structured.call_count == 2
    assert [c.args[0] for c in llm.embed_text.call_args_list] == ['Activities', 'Activities', 'Games']
    assert workflow._read_facts.call_args.args[0] == 'Games except chess'


@pytest.mark.parametrize('output', ['', ' ', 'bad\x00text'])
def test_invalid_rewrite_stops_before_embedding_instead_of_using_excluded_words(output):
    llm, db = Mock(), Mock()
    llm.parse_structured.return_value = {'embedding_text': output}
    workflow = MemoryRetrievalWorkflow('t', 'u', db, llm, Mock(), WorkControl())
    with pytest.raises(ValidationError):
        workflow.retrieve_once('Activities besides hiking')
    llm.embed_text.assert_not_called()


def test_answer_and_direct_llm_retrieval_preserve_full_filter_request(fact_answering, monkeypatch):
    from filter_fixtures import model_input
    from test_answer_facts import judgment
    from test_answer_question import plan, decision
    from tomte.engine.core.answering import RequiredInformation, FactBatchJudgment, AnswerDecision
    monkeypatch.setattr('tomte.engine.workflows.retrieving.prepare_embedding_query', prepare_embedding_query)
    r = fact_answering
    request = 'Outdoor activities besides hiking'
    def model(**kwargs):
        schema = kwargs['schema']
        payload = model_input(kwargs['user_text'])
        if schema is RequiredInformation:
            return plan(request)
        if schema is SearchEmbeddingQuery:
            assert payload == {'information_needed': request}
            return SearchEmbeddingQuery(embedding_text='Outdoor activities')
        if schema is FactBatchJudgment:
            assert payload['information_needed'] == request
            return judgment()
        assert schema is AnswerDecision
        assert payload['question'] == 'Which activities besides hiking?'
        return decision('Supported result.')
    r.model.parse_structured.side_effect = model
    assert r.client.answer_question('user-a', 'Which activities besides hiking?') == 'Supported result.'
    assert r.model.parse_structured.call_count == 4
    r.client.retrieve_memories('user-a', request)
    assert r.model.parse_structured.call_count == 6
    assert [c.args[0] for c in r.model.embed_text.call_args_list] == ['Outdoor activities'] * 2


# Reuse the scoped fake providers; this test restores the actual preparer above.
from test_answer_facts import fact_answering
from test_answer_question import answering
from test_relevant_entities_api import retrieval
