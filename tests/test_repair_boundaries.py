from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from tomte.engine.core.config import settings
from tomte.engine.core.enums import ChatSenderRole
from tomte.engine.core.resolution import EntityResolution
from tomte.engine.exceptions import TomteInvalidInputError
from tomte.engine.workflows.chat_hook import ChatHookWorkflow


def test_oversized_message_rejected_before_any_provider_call():
    hook = object.__new__(ChatHookWorkflow)
    hook.db, hook.cache, hook.llm, hook.security = [Mock() for _ in range(4)]
    with pytest.raises(TomteInvalidInputError):
        hook.on_chat('tenant', 'session', 'x' * (settings.max_input_length + 1), ChatSenderRole.USER, 'user')
    for provider in (hook.db, hook.cache, hook.llm, hook.security):
        assert provider.mock_calls == []


def test_persona_reads_current_database_even_if_cached():
    hook = object.__new__(ChatHookWorkflow)
    hook.cache, hook.db = Mock(), Mock()
    hook.cache.get_persona_entities.return_value = [{'value': 'old'}]
    hook.db.get_important_entities.return_value = [{'value': 'current'}]
    assert hook._get_or_refresh_persona('t', 'u') == [{'value': 'current'}]
    hook.cache.get_persona_entities.assert_not_called()


def test_unresolved_cannot_carry_target_authorization():
    with pytest.raises(ValidationError):
        EntityResolution(extracted_entity_id='local', kind='UNRESOLVED', reasoning='unclear',
                         targets=[dict(db_id='00000000-0000-0000-0000-000000000001',
                                       resolution_confidence=1, evidence='EXPLICIT')])
