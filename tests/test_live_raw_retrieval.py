"""Deterministic live composition and real scoped cosine/transaction regressions."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import math

import psycopg
from psycopg.pq import TransactionStatus
import pytest

from test_postgres_transactions import local_db
from test_session_entities import local_session, local_session_db, session_workflow, insert_entity
from tomte.engine import Tomte
from tomte.engine.core.enums import ChatSenderRole
from tomte.engine.exceptions import TomteSecurityViolationError
from tomte.engine.workflows.chat_hook import ChatHookWorkflow
from tomte.engine.database.postgre import operations
from tomte.engine.database.postgre.transactions import TransactionScopeError
from tomte.engine.exceptions import TomteSessionStateUnavailable


def vector(cosine):
    return [cosine, math.sqrt(1-cosine*cosine)] + [0.0]*1534


def insert(db,tenant,user,eid,cosine):
    payload=dict(entity_id=eid,entity_name=eid,entity_domain='assets_and_possessions',
                 entity_type='vehicle',aliases=[],facts=[])
    assert db.commit_graph_updates([dict(mode='INSERT',tenant_id=tenant,user_id=user,
        **{k:payload[k] for k in ('entity_id','entity_name','entity_domain','entity_type')},
        payload=payload,embedding=vector(cosine))])


def test_raw_top_five_ungated_scoped_stable_and_statistic_independent(local_db):
    db,t,u=local_db.db,local_db.tenant,local_db.users[0]
    for eid,c in [('old',1.0),('a',.2),('b',.2),('c',.1),('d',0.0),('e',-.1)]:
        insert(db,t,u,eid,c)
    insert(db,t,local_db.users[1],'foreign-user',1.0)
    insert(db,t+'-foreign',u,'foreign-tenant',1.0)
    try:
        with db.transaction():
            db.conn.execute("UPDATE tomte_memory_engine SET mention_count=1,last_mentioned_at=now()-interval '365 days' WHERE tenant_id=%s AND user_id=%s AND entity_id='old'",(t,u))
            db.conn.execute("UPDATE tomte_memory_engine SET mention_count=10000,last_mentioned_at=now() WHERE tenant_id=%s AND user_id=%s AND entity_id<>'old'",(t,u))
        query=[1.0]+[0.0]*1535
        before=db.conn.execute('SELECT entity_id,mention_count,last_mentioned_at FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s ORDER BY entity_id',(t,u)).fetchall()
        rows=db.get_raw_semantic_context(t,u,query)
        assert [r['entity_id'] for r in rows]==['old','a','b','c','d']
        assert rows[1]['raw_cosine']==pytest.approx(.2)
        assert all('payload' not in r and r['tenant_id']==t and r['user_id']==u for r in rows)
        assert db.semantic_retrieval(t,u,query)[0]['entity_id']!='old'
        assert db.get_frequent_entities(t,u)[0]['entity_id']!='old'
        assert db.discover_session_entities(t,u,query,.25,5)[0]['entity_id']=='old'
        assert len(db.discover_session_entities(t,u,query,.25,5))==1
        assert before==db.conn.execute('SELECT entity_id,mention_count,last_mentioned_at FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s ORDER BY entity_id',(t,u)).fetchall()
        with db.transaction():
            db.conn.execute("UPDATE tomte_memory_engine SET mention_count=999999,last_mentioned_at=now() WHERE tenant_id=%s AND user_id=%s AND entity_id='old'",(t,u))
            db.conn.execute("UPDATE tomte_memory_engine SET mention_count=1,last_mentioned_at=now()-interval '500 days' WHERE tenant_id=%s AND user_id=%s AND entity_id<>'old'",(t,u))
        assert [r['entity_id'] for r in db.get_raw_semantic_context(t,u,query)]==['old','a','b','c','d']
        assert db.conn.info.transaction_status==TransactionStatus.IDLE
    finally:
        db.delete_user_memory(t+'-foreign',u)


def test_raw_undefined_cosine_sorts_after_valid_matches_without_gating(local_db):
    db,t,u=local_db.db,local_db.tenant,local_db.users[0]
    insert(db,t,u,'valid',.2);insert(db,t,u,'zero-vector',0.0)
    db.conn.execute("UPDATE tomte_memory_engine SET embedding=%s::vector WHERE tenant_id=%s AND user_id=%s AND entity_id='zero-vector'",([0.0]*1536,t,u))
    rows=db.get_raw_semantic_context(t,u,[1.0]+[0.0]*1535)
    assert [r['entity_id'] for r in rows]==['valid','zero-vector']
    assert rows[0]['raw_cosine']==pytest.approx(.2)
    assert rows[1]['raw_cosine'] is None


def test_raw_normalization_failure_caught_by_caller_still_rolls_back(local_db,monkeypatch):
    db,t,u=local_db.db,local_db.tenant,local_db.users[0]
    monkeypatch.setattr(operations,'get_raw_semantic_context',Mock(return_value=[{'payload':None}]))
    with pytest.raises(TransactionScopeError):
        with db.transaction():
            db.upsert_user_activity(t,u)
            with pytest.raises(ValueError):db.get_raw_semantic_context(t,u,[1.0]+[0.0]*1535)
    assert local_db.visible(u)==0
    assert db.conn.info.transaction_status==TransactionStatus.IDLE


@pytest.mark.parametrize('error',[psycopg.OperationalError,psycopg.InterfaceError,ValueError])
def test_raw_provider_only_translates_transient_errors(local_db,monkeypatch,error):
    db=local_db.db
    monkeypatch.setattr(operations,'get_raw_semantic_context',Mock(side_effect=error('synthetic')))
    with pytest.raises(ValueError if error is ValueError else TomteSessionStateUnavailable):
        db.get_raw_semantic_context(local_db.tenant,local_db.users[0],[1.0]+[0.0]*1535)
    assert db.conn.info.transaction_status==TransactionStatus.IDLE


@pytest.fixture
def live_workflow(monkeypatch):
    def e(eid,name):
        return dict(entity_id=eid,entity_name=name,tenant_id='t',user_id='u',aliases=[],facts=[])
    recent=e('r','Prior conversation aquarium')
    session=e('s','Fresh session Tesla')
    below=e('below','Below activation floor')
    below['raw_cosine']=.2
    db=SimpleNamespace(upsert_user_activity=Mock(),get_recent_entities=Mock(return_value=[recent]),
        get_important_entities=Mock(return_value=[e('p','Persona')]),
        get_raw_semantic_context=Mock(return_value=[e('s','Stale Tesla'),below]),
        get_frequent_entities=Mock(side_effect=AssertionError('Frequent live call')),
        semantic_retrieval=Mock(side_effect=AssertionError('Weighted live call')),
        find_session_entity_mentions=Mock(return_value=[]),discover_session_entities=Mock(return_value=[]),
        get_entities_by_ids=Mock(return_value=[session]))
    cache=SimpleNamespace(get_persona_entities=Mock(return_value=[e('p','Persona')]),
        get_chat_history=Mock(return_value=[]),append_chat_history=Mock(),enqueue_for_consolidation=Mock(),
        promote_session_entities=Mock(),get_session_entity_ids=Mock(return_value=['s']))
    monkeypatch.setattr(Tomte,'_db',db);monkeypatch.setattr(Tomte,'_cache',cache)
    monkeypatch.setattr(Tomte,'_llm',SimpleNamespace(embed_text=Mock(return_value=[1.0])))
    monkeypatch.setattr('tomte.engine.workflows.chat_hook.get_security_guard',lambda:SimpleNamespace(
        sanitize_input=lambda t:t,check_for_injection=lambda t:True))
    return ChatHookWorkflow(),db,cache


def test_live_composition_omits_frequent_and_weighted_and_keeps_fresh_session(live_workflow):
    workflow,db,cache=live_workflow
    text=workflow.on_chat('t','s','My aquarium needs care.',ChatSenderRole.USER,'u')['context_text']
    db.get_frequent_entities.assert_not_called();db.semantic_retrieval.assert_not_called()
    db.get_raw_semantic_context.assert_called_once_with('t','u',[1.0],limit=5)
    db.get_recent_entities.assert_called_once_with('t','u')
    assert 'Below activation floor' in text and 'Persona' in text and 'Prior conversation aquarium' in text
    assert 'Fresh session Tesla' in text and 'Stale Tesla' not in text
    assert text.count('[[ENTITY: s |')==1
    assert [text.index('[[ENTITY: '+eid+' |') for eid in ('s','below','p','r')]==sorted(text.index('[[ENTITY: '+eid+' |') for eid in ('s','below','p','r'))
    db.discover_session_entities.assert_called_once_with('t','u',[1.0],.25,5)
    cache.promote_session_entities.assert_called_once_with('t','u','s',[],10,86400)


def test_memory_groups_keep_query_rank_and_complete_fresh_overlap(live_workflow):
    workflow, db, cache = live_workflow
    tesla = dict(id='tesla-db', entity_id='veh_01', entity_name='Tesla Model Y',
                 tenant_id='t', user_id='u', entity_type='vehicle', aliases=['Little Dog'],
                 facts=[dict(fact_id='current', attribute_name='color', value='red', assertion_state='ACTIVE'),
                        dict(fact_id='old', attribute_name='color', value='white', assertion_state='HISTORICAL')],
                 metadata={'preserved': True})
    stale = {**tesla, 'entity_name': 'Stale Tesla', 'facts': []}
    db.get_recent_entities.return_value = [stale]
    second = dict(id='second-db', entity_id='veh_02', entity_name='Honda', tenant_id='t', user_id='u', facts=[])
    lexical = dict(id='lexical-db', entity_id='veh_03', entity_name='Bicycle', tenant_id='t', user_id='u', facts=[])
    continuity = dict(id='continuity-db', entity_id='act_01', entity_name='Running', tenant_id='t', user_id='u', facts=[])
    db.get_raw_semantic_context.return_value = [stale, second]
    db.find_session_entity_mentions.return_value = [second, lexical]
    db.get_entities_by_ids.return_value = [tesla, second, lexical, continuity]
    cache.get_session_entity_ids.return_value = ['veh_01', 'veh_02', 'veh_03', 'act_01']
    result = workflow.on_chat('t', 's', 'Tesla Model Y', ChatSenderRole.USER, 'u')
    relevant, background = result['query_relevant_memories'], result['background_memories']
    assert [e['id'] for e in relevant] == ['tesla-db', 'second-db', 'lexical-db']
    assert relevant[0] == tesla
    assert [e['entity_id'] for e in background] == ['p', 'act_01']
    assert len(relevant + background) == 5
    text = result['context_text']
    assert text.startswith('QUERY-RELEVANT MEMORIES\n')
    assert text.index('BACKGROUND/CONTINUITY') < text.index('[[ENTITY: p |')
    assert text.count('[[ENTITY: veh_01 |') == 1
    assert 'color: red' in text and 'color: white' not in text
    assert {'context_text', 'chat_history', 'clean_message', 'user_id'} <= result.keys()
    workflow.llm.embed_text.assert_called_once_with('Tesla Model Y')
    db.get_raw_semantic_context.assert_called_once()
    db.find_session_entity_mentions.assert_called_once()
    db.get_entities_by_ids.assert_called_once()
    cache.append_chat_history.assert_called_once()
    cache.enqueue_for_consolidation.assert_called_once()


def test_empty_query_groups_keep_background_without_claiming_abstention(live_workflow):
    workflow, db, _ = live_workflow
    db.get_raw_semantic_context.return_value = []
    result = workflow.on_chat('t', 's', 'hello', ChatSenderRole.USER, 'u')
    assert result['query_relevant_memories'] == []
    assert [e['entity_id'] for e in result['background_memories']] == ['p', 'r', 's']
    assert result['context_text'].startswith('QUERY-RELEVANT MEMORIES\n\n\nBACKGROUND/CONTINUITY')


def test_memory_group_dedup_uses_db_identity_not_just_logical_id(live_workflow):
    workflow, _, _ = live_workflow
    row = dict(id='first', entity_id='same', tenant_id='t', user_id='u', facts=[])
    other = {**row, 'id': 'second'}
    query, background = workflow._group_memories([other], [], [row], [])
    assert query == [row] and background == [other]


@pytest.mark.parametrize('role,user', [(ChatSenderRole.USER, None),
                                     (ChatSenderRole.ASSISTANT, 'u'),
                                     (ChatSenderRole.ASSISTANT, None)])
def test_nonretrieval_success_branches_return_empty_memory_lists(live_workflow, role, user):
    workflow, db, cache = live_workflow
    result = workflow.on_chat('t', 's', 'hello', role, user)
    assert result['query_relevant_memories'] == result['background_memories'] == []
    assert result['context_text'] == ''
    workflow.llm.embed_text.assert_not_called()
    db.get_raw_semantic_context.assert_not_called()
    db.upsert_user_activity.assert_not_called()
    cache.append_chat_history.assert_called_once()
    assert cache.enqueue_for_consolidation.call_count == int(role == ChatSenderRole.ASSISTANT and user is not None)


def test_memory_groups_do_not_hide_cross_tenant_lexical_rows(live_workflow):
    workflow, db, _ = live_workflow
    db.find_session_entity_mentions.return_value = [dict(entity_id='foreign', tenant_id='other', user_id='u')]
    with pytest.raises(TomteSecurityViolationError):
        workflow.on_chat('t', 's', 'hello', ChatSenderRole.USER, 'u')


def test_public_client_keeps_error_only_payload(live_workflow):
    from tomte.engine.clients.chat_client import TomteChatClient
    workflow, _, _ = live_workflow
    workflow.security.check_for_injection = lambda _: False
    client = TomteChatClient('t')
    client._workflow = workflow
    result = client.send_user_message('s', 'blocked', user_id='u')
    assert set(result) == {'error'}


@pytest.mark.parametrize('error',[TomteSessionStateUnavailable,NotImplementedError])
def test_transient_raw_failure_keeps_other_streams(live_workflow,error):
    workflow,db,_=live_workflow
    db.get_raw_semantic_context.side_effect=error()
    text=workflow.on_chat('t','s','hello',ChatSenderRole.USER,'u')['context_text']
    assert all(name in text for name in ('Persona','Prior conversation aquarium','Fresh session Tesla'))


def test_legacy_injected_provider_without_raw_capability_keeps_other_streams(live_workflow):
    workflow,db,_=live_workflow
    del db.get_raw_semantic_context
    text=workflow.on_chat('t','s','hello',ChatSenderRole.USER,'u')['context_text']
    assert all(name in text for name in ('Persona','Prior conversation aquarium','Fresh session Tesla'))
    db.get_frequent_entities.assert_not_called();db.semantic_retrieval.assert_not_called()


@pytest.mark.parametrize('error',[ValueError,TomteSecurityViolationError])
def test_raw_programming_and_security_failures_are_visible(live_workflow,error):
    workflow,db,_=live_workflow
    db.get_raw_semantic_context.side_effect=error('synthetic')
    with pytest.raises(error):workflow.on_chat('t','s','hello',ChatSenderRole.USER,'u')


def test_recent_failure_retains_existing_strict_policy(live_workflow):
    workflow,db,_=live_workflow
    db.get_recent_entities.side_effect=TomteSessionStateUnavailable()
    with pytest.raises(TomteSessionStateUnavailable):workflow.on_chat('t','s','hello',ChatSenderRole.USER,'u')


def test_recent_preserves_prior_session_topic_without_semantic_or_mru_hit(session_workflow):
    workflow,db,cache,t,_=session_workflow
    # Generic query in this fixture uses basis vector 2. Tesla ranks sixth and
    # cannot be activated by cosine or a lexical mention in this new session.
    insert_entity(db,t,'u','tesla','Tesla Model Y',vector=[0.0,1.0]+[0.0]*1534)
    for i in range(5):
        insert_entity(db,t,'u','other-'+str(i),'Unrelated '+str(i),vector=[0.0,0.0,1.0]+[0.0]*1533)
    with db.transaction():
        db.conn.execute("UPDATE tomte_memory_engine SET last_mentioned_at=now()-interval '5 days' WHERE tenant_id=%s AND user_id='u' AND entity_id<>'tesla'",(t,))
    assert cache.get_session_entity_ids(t,'u','new-session',10)==[]
    query=[0.0,0.0,1.0]+[0.0]*1533
    assert 'tesla' not in [e['entity_id'] for e in db.get_raw_semantic_context(t,'u',query)]
    result=workflow.on_chat(t,'new-session','What about that problem from yesterday?',ChatSenderRole.USER,'u')
    assert 'Tesla Model Y' in result['context_text']
    assert 'tesla' not in cache.get_session_entity_ids(t,'u','new-session',10)
