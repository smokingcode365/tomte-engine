"""Public retrieval contract and scope checks for pending memory."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from test_relevant_entities_api import retrieval
from test_live_raw_retrieval import live_workflow
from test_unresolved_memories import pending_payload
from tomte.engine.core.memory import public_entity
from tomte.engine.core.unresolved import pending_records
from tomte.engine.exceptions import TomteSecurityViolationError
from tomte.engine.providers.postgres_provider import PostgresProvider


def row(payload, tenant='tenant-a', user='user-a'):
    return dict(id=payload['entity_id'], entity_id=payload['entity_id'], payload=payload,
                tenant_id=tenant, user_id=user)


def test_combined_api_returns_separate_bounded_pending_records_with_one_embedding(retrieval):
    r = retrieval
    pending = pending_payload()
    read = Mock(return_value=[row(pending)] * 5)
    r.db.get_unresolved_memories = read
    # Defensively exclude pending payloads if a custom normal provider includes them.
    r.db.get_raw_semantic_context.return_value.append({**pending, 'tenant_id': 'tenant-a', 'user_id': 'user-a'})
    before = deepcopy(pending)
    result = r.client.get_relevant_memories('user-a', 'My painted car')
    assert set(result) == {'entities', 'unresolved_memories', 'context_text'}
    assert result['entities'] == [public_entity(r.row)]
    assert result['unresolved_memories'] == [public_entity(pending)]
    assert result['context_text'].index(r.row['entity_id']) < result['context_text'].index(pending['entity_id'])
    assert result['unresolved_memories'][0]['reported_at'] == '2026-11-03T00:00:00Z'
    assert '_unresolved' not in result['unresolved_memories'][0]
    assert pending == before
    assert [call[0] for call in r.model.mock_calls] == ['embed_text']
    read.assert_called_once_with('tenant-a', 'user-a', [1., 0.], limit=3, recent_limit=0)


def test_legacy_entity_api_remains_a_list_and_never_fetches_pending(retrieval):
    r = retrieval
    r.db.get_unresolved_memories = Mock(side_effect=AssertionError('Unexpected pending read'))
    assert r.client.get_relevant_entities('user-a', 'My car') == [public_entity(r.row)]


@pytest.mark.parametrize('scope', [('other', 'user-a'), ('tenant-a', 'other')])
def test_foreign_pending_records_fail_before_returning_memory(scope):
    db = SimpleNamespace(get_unresolved_memories=lambda *a, **k: [row(pending_payload(), *scope)])
    with pytest.raises(TomteSecurityViolationError):
        pending_records(db, 'tenant-a', 'user-a', [1., 0.], 3)


def test_resolved_records_are_not_returned_even_by_a_faulty_provider(retrieval):
    from uuid import uuid4
    payload = pending_payload()
    payload['_unresolved'].update(status='resolved', resolved_at='2026-11-05T00:00:00Z', resolved_entity_ids=[str(uuid4())])
    retrieval.db.get_unresolved_memories = Mock(return_value=[row(payload)])
    assert retrieval.client.get_relevant_memories('user-a', 'My car')['unresolved_memories'] == []


def test_storage_normalization_preserves_internal_lifecycle_for_flat_rows():
    payload = pending_payload()
    normalized = PostgresProvider._normalize_memory_rows([{**payload, 'id': payload['entity_id']}])[0]
    assert normalized['_unresolved'] == payload['_unresolved']


def test_chat_places_pending_memory_after_ordinary_context_and_outside_normal_lists(live_workflow):
    workflow, db, cache = live_workflow
    payload = pending_payload()
    db.get_unresolved_memories = Mock(return_value=[row(payload, 't', 'u')])
    result = workflow._handle_registered_user('t', 's', 'u', 'My car')
    assert result['unresolved_memories'] == [public_entity(payload)]
    assert all(e['entity_domain'] != 'unresolved_memories' for e in result['query_relevant_memories'] + result['background_memories'])
    assert result['context_text'].index('UNRESOLVED MEMORIES') > result['context_text'].index('BACKGROUND/CONTINUITY')
    assert 'Which car is not established.' in result['context_text']
    assert '_unresolved' not in result['context_text']
