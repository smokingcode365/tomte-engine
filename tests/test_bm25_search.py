"""Optional lexical mode routing and public API contracts."""
from contextlib import contextmanager
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from tomte.engine import Tomte
from tomte.engine.clients.admin_client import TomteAdminClient
from tomte.engine.core.config import TomteSettings, settings
from tomte.engine.core.search import reciprocal_rank_fusion
from tomte.engine.exceptions import TomteError, TomteSecurityViolationError
from tomte.engine.workflows.memory_search import read_memory_views
from memory_fixtures import simple_entity, simple_fact


def test_mode_defaults_to_vector_and_rejects_unknown_modes():
    assert TomteSettings.model_fields['stored_retrieval_mode'].default == 'vector'
    with pytest.raises(ValidationError):
        TomteSettings(stored_retrieval_mode='typo')


def test_fusion_deduplicates_with_deterministic_ties():
    ranking, scores = reciprocal_rank_fusion(['b', 'b', 'a'], ['a', 'c'])
    assert ranking == ['a', 'b', 'c']
    assert scores['b'] == pytest.approx(1/61)
    assert reciprocal_rank_fusion(['b'], ['a'])[0] == ['a', 'b']
    assert reciprocal_rank_fusion([], []) == ([], {})


@pytest.mark.parametrize('mode', ['vector', 'bm25', 'hybrid'])
def test_shared_read_routes_mode_and_keeps_namespace_checks(monkeypatch, mode):
    db = Mock()
    entity = dict(simple_entity(facts=[simple_fact('detail', 'ERR1001')]), tenant_id='t', user_id='u')
    db.get_raw_semantic_context.return_value = [entity]
    db.get_ranked_memory_context.return_value = [entity]
    monkeypatch.setattr(settings, 'stored_retrieval_mode', mode)
    views, unresolved = read_memory_views('t', 'u', 'ERR1001', [1.], 5, db, include_unresolved=False)
    assert views[0]['entity_id'] == entity['entity_id'] and unresolved == []
    if mode == 'vector':
        db.get_ranked_memory_context.assert_not_called()
        db.get_raw_semantic_context.assert_called_once_with('t', 'u', [1.], limit=5)
    else:
        db.get_raw_semantic_context.assert_not_called()
        db.get_ranked_memory_context.assert_called_once_with('t', 'u', 'ERR1001', [1.], 5, mode=mode)
    entity['user_id'] = 'foreign'
    with pytest.raises(TomteSecurityViolationError, match='namespace'):
        read_memory_views('t', 'u', 'ERR1001', [1.], 5, db, include_unresolved=False)


def test_admin_rebuild_is_guarded_and_does_not_call_models(monkeypatch):
    db, llm = Mock(), Mock()
    db.rebuild_lexical_index.return_value = 4
    @contextmanager
    def operation(*args):
        yield db, llm
    @contextmanager
    def guard(*args):
        yield True
    db.user_guard.side_effect = guard
    monkeypatch.setattr(Tomte, 'operation_context', operation)
    assert TomteAdminClient('t').rebuild_lexical_index('u') == 4
    db.user_guard.assert_called_once_with('t', 'u')
    db.rebuild_lexical_index.assert_called_once_with('t', 'u')
    assert llm.mock_calls == []
    @contextmanager
    def busy(*args):
        yield False
    db.user_guard.side_effect = busy
    with pytest.raises(TomteError, match='busy'):
        TomteAdminClient('t').rebuild_lexical_index('u')
    assert db.rebuild_lexical_index.call_count == 1
