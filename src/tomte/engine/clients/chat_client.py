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
        if self._workflow is None:
            self._workflow = ChatHookWorkflow()
        return self._workflow

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
            return self.workflow.on_chat(
                tenant_id=self.tenant_id,
                session_id=session_id,
                message=message,
                sender_type=ChatSenderRole.USER,
                user_id=user_id
            )
        except TomteError as e:
            tomtelog.error(f"TomteEngine Error during user message: {e}")
            return ChatHookPayload(error=str(e))
        except Exception as e:
            tomtelog.error(f"Critical System Error during user message: {e}")
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
            return self.workflow.on_chat(
                tenant_id=self.tenant_id,
                session_id=session_id,
                message=message,
                sender_type=ChatSenderRole.ASSISTANT,
                user_id=user_id
            )
        except TomteError as e:
            tomtelog.error(f"TomteEngine Error logging assistant response: {e}")
            return ChatHookPayload(error=str(e))
        except Exception as e:
            tomtelog.error(f"Critical System Error logging assistant response: {e}")
            return ChatHookPayload(error="An unexpected system error occurred.")