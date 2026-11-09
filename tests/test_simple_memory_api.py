"""Separated ingestion/read contracts. No network or model calls."""
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import pytest

from tomte.engine import Tomte
from tomte.engine.clients.chat_client import TomteChatClient
from tomte.engine.core.enums import MemoryDomain
from tomte.engine.core.memory import public_entity
from tomte.engine.exceptions import TomteInvalidIdError, TomteInvalidInputError, TomteSecurityViolationError
from memory_fixtures import simple_entity, simple_fact


@pytest.fixture
def api(monkeypatch):
    cache = Mock(spec=['store_message', 'get_pending_message_batches'])
    cache.store_message.return_value = {'message_id':'receipt-id', 'status':'queued'}
    cache.get_pending_message_batches.return_value = []
    db = Mock(spec=['get_profile_entities','get_frequent_entities','is_consolidation_processed'])
    db.get_profile_entities.return_value = []
    db.get_frequent_entities.return_value = []
    db.is_consolidation_processed.return_value = False
    class NoModel:
        def __getattr__(self, name):
            raise AssertionError('This API must not access models: ' + name)
    modes = []
    @contextmanager
    def context(control=None, *, cache_only=False):
        modes.append(cache_only)
        yield (None if cache_only else db), NoModel()
    monkeypatch.setattr(Tomte,'operation_context',context)
    monkeypatch.setattr(Tomte,'get_cache',lambda:cache)
    return SimpleNamespace(client=TomteChatClient('tenant'),cache=cache,db=db,modes=modes)


def test_store_user_message_only_queues_and_returns_receipt(api):
    result = api.client.store_user_message('user','session','I adopted Scout.')
    assert result == {'message_id':'receipt-id','status':'queued'}
    api.cache.store_message.assert_called_once_with('tenant','user','session','I adopted Scout.','user')
    assert api.modes == [True] and not api.db.mock_calls


def test_store_assistant_message_has_same_storage_contract(api):
    result = api.client.store_assistant_message('user','session','What breed is Scout?')
    assert result == api.cache.store_message.return_value
    api.cache.store_message.assert_called_once_with('tenant','user','session','What breed is Scout?','assistant')
    assert api.modes == [True] and not api.db.mock_calls


@pytest.mark.parametrize('method',['store_user_message','store_assistant_message'])
@pytest.mark.parametrize('args',[('', 'session', 'text'),('user','','text'),('user','session',''),('user','session',None)])
def test_storage_rejects_invalid_input_before_providers(api,method,args):
    with pytest.raises((TomteInvalidIdError,TomteInvalidInputError)):
        getattr(api.client,method)(*args)
    assert not api.modes and not api.cache.mock_calls


def test_storage_failure_is_not_acknowledged(api):
    api.cache.store_message.side_effect = RuntimeError('storage unavailable')
    with pytest.raises(RuntimeError,match='storage unavailable'):
        api.client.store_user_message('user','session','A message')


@pytest.mark.parametrize('method,provider',[
    ('get_profile_entities','get_profile_entities'),
    ('get_most_mentioned_entities','get_frequent_entities'),
])
def test_database_only_reads_preserve_rank_and_public_projection(api,method,provider):
    entities=[simple_entity(name,domain=MemoryDomain.IDENTITY,facts=[simple_fact('detail',name)]) for name in ('Second','First')]
    rows=[dict(e,tenant_id='tenant',user_id='user',mention_count=20) for e in entities]
    getattr(api.db,provider).return_value=rows
    before=deepcopy(rows)
    assert getattr(api.client,method)('user') == [public_entity(e) for e in entities]
    assert rows==before and api.modes==[False] and not api.cache.mock_calls
    assert [c[0] for c in api.db.mock_calls]==[provider]


@pytest.mark.parametrize('method,provider',[
    ('get_profile_entities','get_profile_entities'),
    ('get_most_mentioned_entities','get_frequent_entities'),
])
def test_database_only_read_rejects_foreign_scope(api,method,provider):
    getattr(api.db,provider).return_value=[dict(simple_entity(),tenant_id='foreign',user_id='user')]
    with pytest.raises(TomteSecurityViolationError):getattr(api.client,method)('user')


def item(mid,stamp,session='s',user='user'):
    return dict(item_id=mid,tenant_id='tenant',user_id=user,session_id=session,
                role='user',content='message '+mid,timestamp=stamp,vector=None,recent_context=[])


def test_pending_read_includes_processing_excludes_committed_and_selects_latest(api):
    api.cache.get_pending_message_batches.return_value=[
        dict(session_id='s',queued=[item('new',4)],queued_count=1,processing=[item('running',2)]),
        dict(session_id='old',queued=[],queued_count=0,processing=[item('committed',1,'old')]),
        dict(session_id='other',queued=[item('middle',3,'other')],queued_count=1,processing=[]),
    ]
    from tomte.engine.workflows.consolidation import ConsolidationWorkflow
    committed=ConsolidationWorkflow._batch_id('tenant','user',[item('committed',1,'old')])
    api.db.is_consolidation_processed.side_effect=lambda t,u,b:b==committed
    snapshot=deepcopy(api.cache.get_pending_message_batches.return_value)
    result=api.client.get_pending_messages('user',limit=2)
    assert [m['message_id'] for m in result['messages']]==['middle','new']
    assert result['has_more'] is True
    assert all(set(m)=={'message_id','session_id','role','content','received_at','status'} for m in result['messages'])
    assert api.cache.get_pending_message_batches.return_value==snapshot
    assert [c[0] for c in api.cache.mock_calls]==['get_pending_message_batches']
    result=api.client.get_pending_messages('user',limit=10)
    assert [(m['message_id'],m['status']) for m in result['messages']]==[('running','processing'),('middle','queued'),('new','queued')]
    assert result['has_more'] is False


def test_pending_read_cannot_expose_foreign_user_or_unrequested_session(api):
    api.cache.get_pending_message_batches.return_value=[dict(session_id='s',queued=[item('x',1,user='another')],queued_count=1,processing=[])]
    with pytest.raises(TomteSecurityViolationError):api.client.get_pending_messages('user')
    api.cache.get_pending_message_batches.return_value=[dict(session_id='other',queued=[],queued_count=0,processing=[])]
    with pytest.raises(TomteSecurityViolationError):api.client.get_pending_messages('user',session_id='s')


def test_pending_read_reports_hidden_older_rows(api):
    api.cache.get_pending_message_batches.return_value=[dict(session_id='s',queued=[item('x',1)],queued_count=50,processing=[])]
    assert api.client.get_pending_messages('user')['has_more'] is True


@pytest.mark.parametrize('limit',[0,-1,True,1.5])
def test_read_limit_validation_precedes_providers(api,limit):
    for method in ('get_most_mentioned_entities','get_pending_messages'):
        with pytest.raises(TomteInvalidInputError):getattr(api.client,method)('user',limit=limit)
    assert not api.modes and not api.cache.mock_calls and not api.db.mock_calls
