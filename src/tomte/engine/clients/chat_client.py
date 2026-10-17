# tomte/engine/clients/chat_client.py

from typing import Optional
from tomte.engine.workflows.chat_hook import ChatHookWorkflow
from tomte.engine.core.enums import ChatSenderRole
from tomte.engine.core.schema import ChatHookPayload
from tomte.engine.exceptions import TomteError
from tomte.engine import tomtelog

class TomteChatClient:
    """
    The primary SDK facade for the Tomte Engine.
    Handles real-time context retrieval and safely queues data for background consolidation.
    """
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id
        # We lazy-load the workflow to ensure Tomte.init() has been called by the host app
        self._workflow: Optional[ChatHookWorkflow] = None

    @property
    def workflow(self) -> ChatHookWorkflow:
        # Inspecting this legacy property must not switch future managed calls
        # to a cached workflow holding the global template connection.
        return self._workflow if self._workflow is not None else ChatHookWorkflow()

    def _on_chat(self, **kwargs):
        from tomte.engine import Tomte
        from tomte.engine.workers.control import WorkControl
        from contextlib import nullcontext
        from tomte.engine.core.config import settings
        from tomte.engine.exceptions import TomteInvalidInputError
        message = kwargs.get('message')
        if not isinstance(message, str) or not message.strip() or len(message) > settings.max_input_length:
            raise TomteInvalidInputError('Invalid message length')
        # Explicitly injected standalone workflows are caller-owned (legacy API).
        if self._workflow is not None:
            return self._workflow.on_chat(**kwargs)
        cache_only = not kwargs.get('user_id') or kwargs.get('sender_type') == ChatSenderRole.ASSISTANT
        with Tomte.operation_context(WorkControl(timeout=60), cache_only=cache_only) as (db, llm):
            scope = Tomte._usage.scope(self.tenant_id, kwargs.get('user_id'), 'foreground') if Tomte._usage else nullcontext()
            with scope:
                return ChatHookWorkflow(db=db, cache=Tomte.get_cache(), llm=llm,
                                        cache_only=cache_only).on_chat(**kwargs)

    def send_user_message(
        self, 
        session_id: str, 
        message: str, 
        user_id: Optional[str] = None
    ) -> ChatHookPayload:
        """
        Processes a user message, retrieves RAG context, and queues the message for memory extraction.
        
        Args:
            session_id: The unique ID for the current chat session.
            message: The raw text input from the user.
            user_id: The permanent ID of the user (leave None for Guests).
            
        Returns:
            ChatHookPayload containing the chat history, RAG context string, and sanitized message.
        """
        try:
            return self._on_chat(
                tenant_id=self.tenant_id,
                session_id=session_id,
                message=message,
                sender_type=ChatSenderRole.USER,
                user_id=user_id
            )
        except TomteError as e:
            tomtelog.error("Tomte memory unavailable during user message")
            return ChatHookPayload(error="Memory unavailable; continue chat without memory.")
        except Exception as e:
            tomtelog.error("Tomte memory failed during user message")
            return ChatHookPayload(error="An unexpected system error occurred.")

    def log_assistant_response(
        self, 
        session_id: str, 
        message: str, 
        user_id: Optional[str] = None
    ) -> ChatHookPayload:
        """
        Logs the AI's response to the cache history and the consolidation queue.
        Call this immediately after your AI finishes streaming its response to the user.
        """
        try:
            return self._on_chat(
                tenant_id=self.tenant_id,
                session_id=session_id,
                message=message,
                sender_type=ChatSenderRole.ASSISTANT,
                user_id=user_id
            )
        except TomteError as e:
            tomtelog.error("Tomte memory unavailable logging assistant response")
            return ChatHookPayload(error="Memory unavailable; continue chat without memory.")
        except Exception as e:
            tomtelog.error("Tomte memory failed logging assistant response")
            return ChatHookPayload(error="An unexpected system error occurred.")
