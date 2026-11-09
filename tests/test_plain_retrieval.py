"""Plain-field caller retrieval, distinct-entity union, and index isolation."""
from contextlib import contextmanager
from copy import deepcopy
import math
from unittest.mock import Mock

import pytest
import psycopg

from tomte.engine import Tomte
from tomte.engine.clients.admin_client import TomteAdminClient
from tomte.engine.exceptions import TomteSecurityViolationError
from tomte.engine.llm.embedder import get_search_entries, get_retrieval_entries, changed_retrieval_entries
from tomte.engine.workers.control import StaleSnapshot
from memory_fixtures import simple_entity, simple_fact, prepare_write
from test_postgres_transactions import local_db


def vector(score):
    return [score, math.sqrt(1-score*score)] + [0.] * 1534


def test_plain_fields_preserve_current_history_and_long_text_without_labels():
    raw = 'The user’s dog named Panda is a Lab mix. 不同信息🙂' * 200
    fact = simple_fact('breed', raw)
    fact['history_values'] = [{'value':simple_fact('breed','Previously reported as a Jack Russell mix')['value'],
                               '_archived_at':'2026-10-09T12:00:00Z'}]
    data = simple_entity('Panda', facts=[fact])
    entries = get_retrieval_entries(data)
    current = [e for e in entries if e['history_index'] == -1]
    assert len(current) > 1
    assert ''.join(e['text'] for e in current) == raw
    assert all(len(e['text']) <= 2000 for e in entries)
    assert [e['text'] for e in entries if e['history_index'] == 0] == ['Previously reported as a Jack Russell mix']
    assert all(e['fact_id'] == fact['fact_id'] for e in entries)
    changed = deepcopy(data)
    changed.update(entity_name='Different label', entity_type='pet')
    changed['facts'][0]['value']['detail']['fact_confidence'] = 'low'
    assert get_retrieval_entries(changed) == entries
    assert changed_retrieval_entries(changed, data) == []
    assert get_search_entries(changed) != get_search_entries(data)  # Existing identity input retained.


def seed(case, name, summary, plain, labelled=None, *, user=None, facts=1):
    data = simple_entity(name, facts=[simple_fact('detail', f'{name} detail {i}') for i in range(facts)])
    op = prepare_write(case.tenant, user or case.users[0], data, vector=vector(summary))
    for e in op['search_entries']: e['embedding'] = vector(summary if labelled is None else labelled)
    for e in op['retrieval_entries']: e['embedding'] = vector(plain)
    assert case.db.commit_graph_updates([op])
    return op


def test_union_returns_top_five_entities_from_each_ranking_without_fact_crowding(local_db):
    c = local_db
    summaries = [seed(c, f'summary-{i}', .99-i*.01, .10) for i in range(5)]
    fields = [seed(c, f'field-{i}', .05, .89-i*.01, facts=8 if i==0 else 1) for i in range(5)]
    seed(c, 'outside-both', .01, .02)
    seed(c, 'other-user', 1., 1., user=c.users[1])
    rows = c.db.get_raw_semantic_context(c.tenant, c.users[0], vector(1.))
    assert len(rows) == 10
    assert {r['entity_id'] for r in rows} == {op['entity_id'] for op in summaries + fields}
    assert [r['relevance_score'] for r in rows] == sorted([r['relevance_score'] for r in rows], reverse=True)
    assert len(c.db.get_raw_semantic_context(c.tenant, c.users[0], vector(1.), limit=2)) == 4
    assert c.db.get_raw_semantic_context(c.tenant+'-foreign', c.users[0], vector(1.)) == []
    # Put one entity in both lists. It uses one slot, without trimming the union.
    with c.db.transaction():
        c.db.conn.execute('UPDATE tomte_memory_retrieval SET embedding=%s::vector WHERE entity_uuid=%s',
                          (vector(1.), summaries[0]['entity_id']))
    rows = c.db.get_raw_semantic_context(c.tenant, c.users[0], vector(1.))
    assert len(rows) == len({r['entity_id'] for r in rows}) == 9


def test_plain_ranking_cannot_change_mutation_candidate_vectors(local_db):
    c=local_db; u=c.users[0]
    identity = seed(c, 'Identity target', .1, .01, labelled=.99)
    answer = seed(c, 'Answer target', .2, 1., labelled=.2)
    assert str(c.db.get_resolution_candidates(c.tenant,u,[],'',vector(1.),'',1)[0]['id']) == identity['entity_id']
    assert c.db.get_raw_semantic_context(c.tenant,u,vector(1.),limit=1)[0]['entity_id'] == answer['entity_id']


def test_missing_plain_index_aborts_entire_memory_and_index_write(local_db):
    c=local_db
    data=simple_entity('Unindexed write',facts=[simple_fact('detail','Some content')])
    op=prepare_write(c.tenant,c.users[0],data)
    del op['retrieval_entries']
    # The provider's public commit contract reports validation rejection as False.
    assert c.db.commit_graph_updates([op]) is False
    assert c.db.get_user_graph_skeleton(c.tenant,c.users[0]) == []


def test_index_backfill_is_atomic_scoped_and_preserves_memory_and_identity_index(local_db,monkeypatch):
    c=local_db; u=c.users[0]
    op=seed(c,'Backfill target',.2,.1, labelled=.8)
    before=c.db.get_user_graph_skeleton(c.tenant,u)
    entries=deepcopy(op['retrieval_entries'])
    for e in entries: e['embedding']=vector(1.)
    llm=Mock()
    llm.get_vector_dimension.return_value=1536
    llm.embed_batch.side_effect=lambda texts: [vector(1.) for _ in texts]
    @contextmanager
    def operation(*args): yield c.db,llm
    monkeypatch.setattr(Tomte,'operation_context',operation)
    assert TomteAdminClient(c.tenant).rebuild_retrieval_index(u) == 1
    assert c.db.get_raw_semantic_context(c.tenant,u,vector(1.),1)[0]['field_cosine'] == pytest.approx(1.)
    assert c.db.get_resolution_candidates(c.tenant,u,[],'',vector(1.),'',1)[0]['similarity'] == pytest.approx(.8)
    assert c.db.get_user_graph_skeleton(c.tenant,u) == before
    for tenant,user,payload in [(c.tenant,c.users[1],op['payload']),
                               (c.tenant+'-other',u,op['payload']),
                               (c.tenant,u,{**op['payload'],'entity_name':'stale'})]:
        with pytest.raises(StaleSnapshot):
            c.db.replace_retrieval_index(tenant,user,op['entity_id'],payload,entries)
    broken=deepcopy(entries); broken[0]['embedding']=[1.,0.]
    with pytest.raises(psycopg.DataError):  # SQL dimension rejection must roll back the delete too.
        c.db.replace_retrieval_index(c.tenant,u,op['entity_id'],op['payload'],broken)
    assert c.db.get_raw_semantic_context(c.tenant,u,vector(1.),1)[0]['field_cosine'] == pytest.approx(1.)


def test_admin_rebuild_embeds_only_text_and_checks_scope(monkeypatch):
    data=simple_entity('Named pet',facts=[simple_fact('breed','A Lab mix.')])
    row=dict(id=data['entity_id'],tenant_id='t',user_id='u',payload=data)
    db=Mock(); llm=Mock()
    db.get_user_graph_skeleton.return_value=[row]
    llm.get_vector_dimension.return_value=2
    llm.embed_batch.return_value=[[1.,0.]]
    @contextmanager
    def guard(*args): yield True
    @contextmanager
    def operation(*args): yield db,llm
    db.user_guard.side_effect=guard
    monkeypatch.setattr(Tomte,'operation_context',operation)
    assert TomteAdminClient('t').rebuild_retrieval_index('u') == 1
    llm.embed_batch.assert_called_once_with(['A Lab mix.'])
    db.replace_retrieval_index.assert_called_once()
    row['user_id']='other'
    with pytest.raises(TomteSecurityViolationError,match='namespace mismatch'):
        TomteAdminClient('t').rebuild_retrieval_index('u')
    assert llm.embed_batch.call_count == 1


def test_optional_summary_refresh_commits_with_fields_and_rolls_back_together(local_db, monkeypatch):
    c = local_db; user = c.users[0]
    op = seed(c, 'Summary refresh', .2, .1, labelled=.8)
    before = c.db.get_user_graph_skeleton(c.tenant, user)
    from tomte.engine.llm.embedder import get_entity_summary
    summary = get_entity_summary(op['payload'])
    llm = Mock()
    llm.get_vector_dimension.return_value = 1536
    llm.embed_batch.side_effect = lambda texts: [vector(.9 if t == summary else .7) for t in texts]
    @contextmanager
    def operation(*args): yield c.db, llm
    monkeypatch.setattr(Tomte, 'operation_context', operation)
    assert TomteAdminClient(c.tenant).rebuild_retrieval_index(user, refresh_summary=True) == 1
    assert llm.embed_batch.call_args.args[0][0] == summary
    def scores():
        row = c.db.get_raw_semantic_context(c.tenant, user, vector(1.), 1)[0]
        return row['summary_cosine'], row['field_cosine']
    assert scores() == pytest.approx((.9, .7))
    assert c.db.get_user_graph_skeleton(c.tenant, user) == before
    with c.db.transaction():
        labelled = c.db.conn.execute(
            'SELECT 1-(embedding <=> %s::vector) AS score FROM tomte_memory_search '
            'WHERE tenant_id=%s AND user_id=%s AND entity_uuid=%s',
            (vector(1.), c.tenant, user, op['entity_id'])).fetchone()['score']
    assert labelled == pytest.approx(.8)
    # A summary failure must roll back the preceding plain-index replacement.
    entries = deepcopy(op['retrieval_entries'])
    with pytest.raises(psycopg.DataError):
        c.db.replace_retrieval_index(c.tenant, user, op['entity_id'], op['payload'], entries,
                                    summary_embedding=[1., 0.])
    assert scores() == pytest.approx((.9, .7))
    assert c.db.get_user_graph_skeleton(c.tenant, user) == before
