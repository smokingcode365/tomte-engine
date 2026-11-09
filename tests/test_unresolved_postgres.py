"""Opt-in SQL invariants, run against an isolated temporary database."""
from copy import deepcopy
import pytest

from test_postgres_transactions import local_db
from test_unresolved_memories import pending_payload
from memory_fixtures import simple_entity, simple_fact, prepare_write
from tomte.engine.core.unresolved import resolution_operations
from tomte.engine.database.postgre.operations import bulk_commit_updates
from tomte.engine.workers.control import StaleSnapshot

VECTOR = [1.] + [0.] * 1535


def test_pending_rows_cannot_crowd_out_normal_memories_and_are_scoped(local_db):
    db, tenant, user = local_db.db, local_db.tenant, local_db.users[0]
    normal = simple_entity('Tesla', kind='car', domain='assets_and_possessions', facts=[simple_fact('model', 'Tesla')])
    pending = [pending_payload() for _ in range(7)]
    assert db.commit_graph_updates([prepare_write(tenant, user, e) for e in [normal, *pending]])
    for read in (
        lambda: db.get_recent_entities(tenant, user, 1),
        lambda: db.get_frequent_entities(tenant, user, 1),
        lambda: db.get_important_entities(tenant, user, limit=1),
        lambda: db.get_raw_semantic_context(tenant, user, VECTOR, 1),
        lambda: db.semantic_retrieval(tenant, user, VECTOR, 1),
        lambda: db.discover_session_entities(tenant, user, VECTOR, .1, 1),
        lambda: db.find_session_entity_mentions(tenant, user, 'Tesla and my car', VECTOR, 1),
        lambda: db.get_entities_by_ids(tenant, user, [p['entity_id'] for p in pending] + [normal['entity_id']]),
    ):
        assert [e['entity_id'] for e in read()] == [normal['entity_id']]
    assert [str(e['id']) for e in db.get_resolution_candidates(tenant, user, [], '', VECTOR, 'my car', 1)] == [normal['entity_id']]
    assert 'unresolved_memories' not in db.get_ontology_dictionaries(tenant, VECTOR)
    assert len(db.get_unresolved_memories(tenant, user, VECTOR, 3)) == 3
    assert len(db.get_unresolved_memories(tenant, user, VECTOR, 3, 2)) == 5
    for other_tenant, other_user in [(tenant, local_db.users[1]), (tenant + '-other', user)]:
        assert db.get_unresolved_memories(other_tenant, other_user, VECTOR, 3) == []


def test_resolution_is_atomic_and_replay_cannot_reopen_or_apply_old_snapshot(local_db):
    db, tenant, user = local_db.db, local_db.tenant, local_db.users[0]
    old = simple_entity('Tesla', kind='car', domain='assets_and_possessions', facts=[simple_fact('model', 'Tesla')])
    pending = pending_payload()
    assert db.commit_graph_updates([prepare_write(tenant, user, e) for e in [old, pending]])
    stored = db.get_unresolved_memories(tenant, user, VECTOR)[0]
    changed = deepcopy(old)
    changed['facts'].append(simple_fact('repainting', 'Painted red, reported 3 November 2026.'))
    target = prepare_write(tenant, user, changed, old=old)
    target['_accepted_unresolved_ids'] = [pending['entity_id']]
    markers = resolution_operations([target], {pending['entity_id']: stored})
    with pytest.raises(RuntimeError, match='rollback'):
        with db.transaction():
            assert db.claim_consolidation_batch(tenant, user, 'unresolved-test')
            assert db.commit_graph_updates([target, *markers])
            raise RuntimeError('rollback')
    assert db.get_unresolved_memories(tenant, user, VECTOR)[0]['payload'] == pending
    assert db.get_raw_semantic_context(tenant, user, VECTOR, 1)[0]['facts'] == old['facts']
    assert not db.is_consolidation_processed(tenant, user, 'unresolved-test')
    with db.transaction():
        assert db.claim_consolidation_batch(tenant, user, 'unresolved-test')
        assert db.commit_graph_updates([target, *markers])
    assert db.get_unresolved_memories(tenant, user, VECTOR) == []
    assert len(db.get_raw_semantic_context(tenant, user, VECTOR, 1)[0]['facts']) == 2
    assert len(db.get_user_graph_skeleton(tenant, user)) == 2  # Retain the resolved record internally.
    with pytest.raises(StaleSnapshot):
        bulk_commit_updates(db.conn, [target, *markers])
    with pytest.raises(ValueError, match='together'):
        bulk_commit_updates(db.conn, [target])
    with pytest.raises(ValueError, match='scoped target writes'):
        bulk_commit_updates(db.conn, markers)
    foreign = deepcopy(target)
    foreign['user_id'] = local_db.users[1]
    with pytest.raises(ValueError, match='scoped target writes'):
        bulk_commit_updates(db.conn, [foreign, *markers])
