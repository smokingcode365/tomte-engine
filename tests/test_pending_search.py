"""Request-level failure/reuse behavior; deterministic provider doubles only."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from tomte.engine.exceptions import TomteSecurityViolationError, TomteSessionStateUnavailable
from tomte.engine.llm.embedding_identity import openai_embedding_identity
from tomte.engine.workflows.pending_search import PendingMemorySearch
from test_pending_windows import messages
from test_relevant_entities_api import retrieval


def source():
    return dict(sessions=[dict(session_id='s',generation='generation',processing=[],identity=[],
                              queued=messages(1))],anchors=[],complete=True,
                observed_messages=1,snapshot_at=1700000000)


def model():
    return SimpleNamespace(get_embedding_cache_identity=lambda:'synthetic-v1',
        get_vector_dimension=lambda:2,embed_batch=Mock(return_value=[[1.,0.]]))


def test_waiter_takes_over_a_released_claim_then_reuses_request_vector():
    cache=Mock()
    cache.get_pending_search_snapshot.return_value=source()
    cache.claim_pending_window.side_effect=[{'state':'busy'},{'state':'owned'}]
    search=PendingMemorySearch('t','u',Mock(),cache)
    llm=model()
    assert search.search([1.,0.],llm)['pending_search']['complete']
    assert search.search([0.,1.],llm)['pending_search']['complete']
    llm.embed_batch.assert_called_once()
    cache.publish_pending_window.assert_called_once()
    cache.release_pending_window.assert_called_once()


def test_unavailable_pending_snapshot_reports_incomplete():
    cache=Mock()
    cache.get_pending_search_snapshot.side_effect=TomteSessionStateUnavailable('Synthetic failure')
    search=PendingMemorySearch('t','u',Mock(),cache)
    llm=model()
    result=search.search([1.,0.],llm)
    assert result['pending_message_windows']==[]
    assert result['pending_search']['complete'] is False
    assert result['pending_search']['reasons']==['pending_snapshot_unavailable']
    llm.embed_batch.assert_not_called()


def test_unsupported_provider_cache_reuses_vectors_only_within_request():
    cache=Mock(); cache.get_pending_search_snapshot.return_value=source()
    llm=model(); llm.get_embedding_cache_identity=lambda:None
    search=PendingMemorySearch('t','u',Mock(),cache)
    search.search([1.,0.],llm); search.search([1.,0.],llm)
    llm.embed_batch.assert_called_once()
    cache.claim_pending_window.assert_not_called()


def test_cross_user_source_is_rejected_before_marker_lookup():
    snapshot=source()
    snapshot['sessions'][0]['identity']=[dict(item_id='x',tenant_id='t',user_id='other',session_id='s')]
    cache=Mock(); cache.get_pending_search_snapshot.return_value=snapshot
    db=Mock()
    with pytest.raises(TomteSecurityViolationError): PendingMemorySearch('t','u',db,cache)
    db.is_consolidation_processed.assert_not_called()


def test_embedding_identity_normalizes_endpoint_but_separates_deployments():
    a=openai_embedding_identity('openai:model',2,'https://example.test/v1/')
    assert a==openai_embedding_identity('model',2,'https://example.test/v1')
    assert a!=openai_embedding_identity('model',3,'https://example.test/v1')
    assert a!=openai_embedding_identity('model',2,'https://other.test/v1')


def test_pending_cache_outage_still_returns_durable_entities(retrieval,monkeypatch):
    from tomte.engine import Tomte
    from tomte.engine.core.memory import public_entity
    r=retrieval
    cache=Mock()
    cache.get_pending_search_snapshot.side_effect=TomteSessionStateUnavailable('Synthetic outage')
    monkeypatch.setattr(Tomte,'get_cache',lambda:cache)
    r.db.get_unresolved_memories=Mock(return_value=[])
    result=r.client.get_relevant_memories('user-a','My job',include_pending=True)
    assert result['entities']==[public_entity(r.row)]
    assert result['pending_search']['complete'] is False
    assert [call[0] for call in r.model.mock_calls]==['embed_text']


def test_cosine_search_returns_best_window_with_explicit_roles(monkeypatch):
    from tomte.engine.core.config import settings
    snapshot=source(); items=messages(14)
    items[-1]['role']='assistant'
    snapshot['sessions'][0]['queued']=items
    snapshot['observed_messages']=14
    cache=Mock(); cache.get_pending_search_snapshot.return_value=snapshot
    llm=model(); llm.get_embedding_cache_identity=lambda:None
    llm.embed_batch.side_effect=lambda texts:[[0.,1.] if 'message 14' in t else [1.,0.] for t in texts]
    monkeypatch.setattr(settings.pending_search,'return_windows',1)
    search=PendingMemorySearch('t','u',Mock(),cache)
    result=search.search([0.,1.],llm)
    returned=result['pending_message_windows'][0]['messages']
    assert [m['message_id'] for m in returned]==[f'm{i}' for i in range(9,15)]
    assert returned[-1]['role']=='assistant'


def test_get_context_returns_ready_text_with_all_memory_sources(retrieval,monkeypatch):
    from tomte.engine import Tomte
    from test_unresolved_memories import pending_payload
    r=retrieval
    snapshot=source()
    snapshot['sessions'][0]['queued'][0].update(
        tenant_id='tenant-a',user_id='user-a',content='I took Scout to the vet this morning.')
    cache=Mock(spec=['get_pending_search_snapshot'])
    cache.get_pending_search_snapshot.return_value=snapshot
    monkeypatch.setattr(Tomte,'get_cache',lambda:cache)
    payload=pending_payload()
    r.db.get_unresolved_memories=Mock(return_value=[dict(
        id=payload['entity_id'],payload=payload,tenant_id='tenant-a',user_id='user-a')])
    r.model.get_vector_dimension=lambda:2
    r.model.embed_batch=Mock(return_value=[[1.,0.]])

    context=r.client.get_context(user_id='user-a',message='My dog Scout')

    assert isinstance(context,str)
    assert 'Doctor at Harbour Hospital' in context
    assert 'Which car is not established.' in context
    assert 'UNRESOLVED MEMORIES' in context and 'PENDING CONVERSATION' in context
    assert 'I took Scout to the vet this morning.' in context
    assert 'pending_search' in context
    r.model.embed_text.assert_called_once_with('My dog Scout')
    r.model.embed_batch.assert_called_once()
    r.model.parse_structured.assert_not_called()
    assert [call[0] for call in cache.mock_calls]==['get_pending_search_snapshot']


def test_get_context_preserves_retrieval_failure(retrieval,monkeypatch):
    from tomte.engine import Tomte
    r=retrieval
    cache=Mock(spec=['get_pending_search_snapshot'])
    cache.get_pending_search_snapshot.side_effect=NotImplementedError
    monkeypatch.setattr(Tomte,'get_cache',lambda:cache)
    r.model.embed_text.side_effect=RuntimeError('Synthetic query embedding failure')

    with pytest.raises(RuntimeError,match='Synthetic query embedding failure'):
        r.client.get_context('user-a','My dog Scout')
