"""Optional dictionary failures must not undo authoritative memory."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from tomte.engine.workflows.consolidation import ConsolidationWorkflow


class RecordingDB:
    def __init__(self):
        self.committed = False
        self.memory = False

    @contextmanager
    def transaction(self):
        try:
            yield
        except Exception:
            self.memory = False
            raise
        else:
            self.committed = True

    def claim_consolidation_batch(self, *args):
        return not self.committed

    def commit_graph_updates(self, operations):
        self.memory = True
        return True

    def dictionary_write_guard(self):
        raise AssertionError('Core memory must not acquire the dictionary lock')


def test_feedback_failure_does_not_rollback_memory_or_repeat_feedback():
    db = RecordingDB()
    workflow = ConsolidationWorkflow(db=db, llm=Mock())
    def fail(operations):
        assert db.committed and db.memory
        raise ValueError('synthetic optional failure')
    workflow.dict_service = SimpleNamespace(apply_feedback=Mock(side_effect=fail))
    workflow._commit_refined_operations([], 't', 'u', 'batch')
    assert db.committed and db.memory
    assert workflow.last_dictionary_feedback == 'failed'
    workflow._commit_refined_operations([], 't', 'u', 'batch')
    assert workflow.dict_service.apply_feedback.call_count == 1


def test_core_failure_still_rolls_back_and_never_calls_feedback():
    db = RecordingDB()
    db.commit_graph_updates = Mock(return_value=False)
    workflow = ConsolidationWorkflow(db=db, llm=Mock())
    workflow.dict_service = SimpleNamespace(apply_feedback=Mock())
    with pytest.raises(RuntimeError, match='memory write failed'):
        workflow._commit_refined_operations([], 't', 'u', 'batch')
    assert not db.committed
    workflow.dict_service.apply_feedback.assert_not_called()
