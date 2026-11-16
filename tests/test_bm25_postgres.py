"""Real PostgreSQL lexical scoring and transactional integration; no model calls."""
from copy import deepcopy
import math
from concurrent.futures import ThreadPoolExecutor

import pytest
import psycopg

from tomte.engine.database.postgre import lexical_ops
from tomte.engine.exceptions import TomteConfigurationError
from memory_fixtures import simple_entity, simple_fact, prepare_write
from test_postgres_transactions import local_db
from test_plain_retrieval import vector


def seed(case, text, *, name='Synthetic memory', tenant=None, user=None, score=.5, facts=None):
    payload = simple_entity(name, facts=facts or [simple_fact('detail', text)])
    op = prepare_write(tenant or case.tenant, user or case.users[0], payload, vector=vector(score))
    assert case.db.commit_graph_updates([op])
    return payload


def search(case, query, *, mode='bm25', limit=5, tenant=None, user=None):
    return case.db.get_ranked_memory_context(tenant or case.tenant, user or case.users[0],
        query, vector(1.), limit, mode=mode)


def test_bm25_formula_exact_codes_and_duplicate_query_terms(local_db):
    c = local_db
    target = seed(c, 'ERR1001')
    seed(c, 'ERR1002')
    rows = search(c, 'err1001')
    assert [r['entity_id'] for r in rows] == [target['entity_id']]
    assert rows[0]['bm25_score'] == pytest.approx(math.log(2))
    assert search(c, 'ERR1001 ERR1001')[0]['bm25_score'] == pytest.approx(math.log(2))
    assert search(c, 'ERR1999') == []
    assert search(c, '***') == []


def test_whole_fields_and_true_term_frequency(local_db):
    c = local_db
    # ERR1001 crosses a 2,000-character embedding chunk boundary.
    target = seed(c, 'x ' * 999 + 'ERR1001 ' + 'repeat ' * 300)
    assert search(c, 'ERR1001')[0]['entity_id'] == target['entity_id']
    row = c.observer.execute('SELECT terms,document_length FROM tomte_memory_lexical WHERE entity_uuid=%s',
                             (target['entity_id'],)).fetchone()
    assert row[0] == {'x': 999, 'err1001': 1, 'repeat': 300}
    assert row[1] == 1300  # No tsvector 256-position truncation.


def test_corpus_statistics_and_results_are_owner_scoped(local_db):
    c = local_db
    own = seed(c, 'ERR1001')
    seed(c, 'ERR1002')
    before = search(c, 'ERR1001')[0]['bm25_score']
    seed(c, 'ERR1001 ' * 10, user=c.users[1])
    foreign = c.tenant + '-other'
    try:
        seed(c, 'ERR1001', tenant=foreign)
        rows = search(c, 'ERR1001')
        assert [r['entity_id'] for r in rows] == [own['entity_id']]
        assert rows[0]['bm25_score'] == pytest.approx(before)
        assert search(c, 'ERR1001', tenant=c.tenant + '-absent') == []
        assert c.db.get_ranked_history_candidates(foreign, c.users[0], [own['entity_id']],
                                                  'ERR1001', vector(1.), 5, mode='bm25') == []
    finally:
        c.db.delete_user_memory(foreign, c.users[0])


def test_duplicate_fact_attributes_do_not_crowd_entity_results(local_db):
    c = local_db
    a = seed(c, '', facts=[simple_fact('error', 'ERR1001') for _ in range(8)])
    b = seed(c, 'ERR1001')
    assert {r['entity_id'] for r in search(c, 'ERR1001', limit=2)} == {a['entity_id'], b['entity_id']}


def test_mutation_replaces_current_terms_and_preserves_history_then_delete_cascades(local_db):
    c = local_db
    old = seed(c, 'ERR1001')
    new = deepcopy(old)
    new['facts'][0]['value']['detail']['text'] = 'ERR1002'
    new['facts'][0]['history_values'].append({'value': old['facts'][0]['value'], '_archived_at': '2026-11-09T12:00:00Z'})
    assert c.db.commit_graph_updates([prepare_write(c.tenant, c.users[0], new, old=old)])
    assert search(c, 'ERR1001')[0]['entity_id'] == old['entity_id']
    assert search(c, 'ERR1002')[0]['entity_id'] == old['entity_id']
    rows = c.observer.execute('SELECT history_index,terms FROM tomte_memory_lexical WHERE entity_uuid=%s ORDER BY history_index',
                             (old['entity_id'],)).fetchall()
    assert rows == [(-1, {'err1002': 1}), (0, {'err1001': 1})]
    corrected = deepcopy(new)
    corrected['facts'][0]['value']['detail']['text'] = 'ERR1003'
    assert c.db.commit_graph_updates([prepare_write(c.tenant, c.users[0], corrected, old=new)])
    assert search(c, 'ERR1002') == []  # A correction without archival leaves no stale index row.
    c.db.delete_user_memory(c.tenant, c.users[0])
    assert c.observer.execute('SELECT count(*) FROM tomte_memory_lexical WHERE entity_uuid=%s',
                             (old['entity_id'],)).fetchone()[0] == 0


def test_failed_lexical_write_rolls_back_canonical_and_vector_updates(local_db, monkeypatch):
    c = local_db
    old = seed(c, 'ERR1001', score=.3)
    new = deepcopy(old)
    new['facts'][0]['value']['detail']['text'] = 'ERR1002'
    original = lexical_ops.replace_entity_index
    def fail_after_index(*args):
        original(*args)
        raise RuntimeError('synthetic lexical failure')
    monkeypatch.setattr(lexical_ops, 'replace_entity_index', fail_after_index)
    assert c.db.commit_graph_updates([prepare_write(c.tenant, c.users[0], new, old=old, vector=vector(.9))]) is False
    assert c.db.get_user_graph_skeleton(c.tenant, c.users[0])[0]['payload'] == old
    assert c.db.get_raw_semantic_context(c.tenant, c.users[0], vector(1.))[0]['field_cosine'] == pytest.approx(.3)
    assert search(c, 'ERR1001') and not search(c, 'ERR1002')


def test_backfill_readiness_idempotency_and_atomic_failure(local_db, monkeypatch):
    c = local_db
    seed(c, 'ERR1001')
    seed(c, 'ERR1002')
    other = seed(c, 'ERR2001', user=c.users[1])
    before = c.db.get_user_graph_skeleton(c.tenant, c.users[0])
    vector_before = c.db.get_raw_semantic_context(c.tenant, c.users[0], vector(1.))
    with c.db.transaction():
        c.db.conn.execute('DELETE FROM tomte_memory_lexical WHERE tenant_id=%s AND user_id=%s', (c.tenant, c.users[0]))
        c.db.conn.execute('UPDATE tomte_memory_engine SET lexical_indexed=FALSE WHERE tenant_id=%s', (c.tenant,))
    with pytest.raises(TomteConfigurationError, match='rebuild_lexical_index'):
        search(c, 'ERR1001')
    assert len(c.db.get_raw_semantic_context(c.tenant, c.users[0], vector(1.))) == 2
    original = lexical_ops.replace_entity_index
    def fail(*args):
        original(*args)
        raise RuntimeError('synthetic rebuild failure')
    with monkeypatch.context() as patch:
        patch.setattr(lexical_ops, 'replace_entity_index', fail)
        with pytest.raises(RuntimeError, match='synthetic rebuild'):
            c.db.rebuild_lexical_index(c.tenant, c.users[0])
    assert c.observer.execute('SELECT count(*) FROM tomte_memory_lexical WHERE tenant_id=%s AND user_id=%s',
                             (c.tenant, c.users[0])).fetchone()[0] == 0
    assert c.db.rebuild_lexical_index(c.tenant, c.users[0]) == 2
    assert c.db.rebuild_lexical_index(c.tenant, c.users[0]) == 2
    # The raw graph has no ordering contract; compare each complete stored row by UUID.
    after_graph = c.db.get_user_graph_skeleton(c.tenant, c.users[0])
    assert {str(r['id']): r for r in after_graph} == {str(r['id']): r for r in before}
    # SQL-only marker changes are excluded from canonical memory and embedding checks.
    after = c.db.get_raw_semantic_context(c.tenant, c.users[0], vector(1.))
    assert [(r['entity_id'], r['summary_cosine'], r['field_cosine']) for r in after] == [
        (r['entity_id'], r['summary_cosine'], r['field_cosine']) for r in vector_before]
    assert search(c, 'ERR1001')
    assert not c.observer.execute('SELECT lexical_indexed FROM tomte_memory_engine WHERE id=%s',
                                 (other['entity_id'],)).fetchone()[0]


def test_hybrid_budget_distinct_entities_and_no_keyword_fallback(local_db):
    c = local_db
    target = seed(c, 'ERR1001', score=.1)
    for i in range(12):
        seed(c, 'A general automotive problem', score=.95-i*.04)
    base = c.db.get_raw_semantic_context(c.tenant, c.users[0], vector(1.), 5)
    hybrid = search(c, 'ERR1001', mode='hybrid')
    assert len(hybrid) == len({r['entity_id'] for r in hybrid}) == len(base) == 5
    assert target['entity_id'] not in {r['entity_id'] for r in base}
    assert target['entity_id'] in {r['entity_id'] for r in hybrid}
    assert search(c, 'ZZZ999', mode='hybrid') == base


def test_matching_old_history_is_projected_in_lexical_modes(local_db, monkeypatch):
    from tomte.engine.core.config import settings
    from tomte.engine.core.memory import public_entity
    from tomte.engine.workflows.memory_search import read_memory_views
    c = local_db
    fact = simple_fact('repair', 'Current repair completed')
    for i in range(10):
        fact['history_values'].append({'value': simple_fact('repair', 'ERR1001' if i == 0 else 'Later repair completed')['value'],
                                       '_archived_at': '2026-11-09T12:00:00Z'})
    target = seed(c, '', facts=[fact])
    monkeypatch.setattr(settings, 'memory_history_bytes', 150)
    for mode in ('bm25', 'hybrid'):
        monkeypatch.setattr(settings, 'stored_retrieval_mode', mode)
        views, _ = read_memory_views(c.tenant, c.users[0], 'ERR1001', vector(1.), 5, c.db, include_unresolved=False)
        assert views[0]['entity_id'] == target['entity_id']
        public = public_entity(views[0])
        assert public['facts'][0]['history_values'][0]['value']['detail']['text'] == 'ERR1001'


def test_backfill_conflicting_with_writer_fails_cleanly_and_retry_uses_latest_payload(local_db):
    c = local_db
    old = seed(c, 'ERR1001')
    new = deepcopy(old)
    new['facts'][0]['value']['detail']['text'] = 'ERR1002'
    worker = c.db.fork()
    try:
        worker.conn.execute("SET lock_timeout='100ms'")
        with ThreadPoolExecutor(max_workers=1) as pool:
            with c.db.transaction():
                assert c.db.commit_graph_updates([prepare_write(c.tenant, c.users[0], new, old=old)])
                future = pool.submit(worker.rebuild_lexical_index, c.tenant, c.users[0])
                with pytest.raises(psycopg.errors.LockNotAvailable):
                    future.result(timeout=5)
                # Another connection still sees the previous complete canonical/index pair.
                row = c.observer.execute('SELECT terms FROM tomte_memory_lexical WHERE entity_uuid=%s',
                                         (old['entity_id'],)).fetchone()[0]
                assert row == {'err1001': 1}
            assert worker.rebuild_lexical_index(c.tenant, c.users[0]) == 1
            assert search(c, 'ERR1002') and not search(c, 'ERR1001')
    finally:
        worker.close()
