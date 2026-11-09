# tomte/engine/clients/chat_client.py

from typing import Optional
from tomte.engine.workflows.chat_hook import ChatHookWorkflow
from tomte.engine.core.enums import ChatSenderRole
from tomte.engine.core.schema import ChatHookPayload, MessageReceipt, PendingMessagesPayload
from tomte.engine.core.memory_text import (
    MemoryTextFormat, resolve_memory_text_format,
)
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

    def _validate_user(self, user_id: str) -> None:
        from tomte.engine.utils.keys import validate_id
        from tomte.engine.exceptions import TomteInvalidIdError
        if not validate_id(self.tenant_id) or not validate_id(user_id):
            raise TomteInvalidIdError("Invalid tenant or user ID")

    @staticmethod
    def _validate_limit(limit: int) -> None:
        from tomte.engine.exceptions import TomteInvalidInputError
        if type(limit) is not int or limit < 1:
            raise TomteInvalidInputError("Limit must be a positive integer")

    def store_user_message(self, user_id: str, session_id: str, message: str) -> MessageReceipt:
        """Queue a user message for background extraction; return only a receipt."""
        return self._store_message(user_id, session_id, message, 'user')

    def store_assistant_message(self, user_id: str, session_id: str, message: str) -> MessageReceipt:
        """Queue assistant context; it is not independent evidence about the user."""
        return self._store_message(user_id, session_id, message, 'assistant')

    def _store_message(self, user_id, session_id, message, role):
        from tomte.engine import Tomte
        from tomte.engine.exceptions import TomteInvalidIdError
        from tomte.engine.utils.keys import validate_id
        from tomte.engine.utils.security import get_security_guard
        self._validate_user(user_id)
        if not validate_id(session_id):
            raise TomteInvalidIdError("Invalid session ID")
        message = get_security_guard().sanitize_input(message)
        with Tomte.operation_context(cache_only=True):
            return Tomte.get_cache().store_message(self.tenant_id, user_id, session_id, message, role)

    def get_profile_entities(self, user_id: str) -> list[dict]:
        """Return all identity_and_demographics entities, without a model call."""
        return self._read_background_entities(user_id, profile=True)

    def get_most_mentioned_entities(self, user_id: str, limit: int = 5) -> list[dict]:
        """Rank entities by summed fact mention counts; unresolved memories are excluded."""
        self._validate_limit(limit)
        return self._read_background_entities(user_id, profile=False, limit=limit)

    def _read_background_entities(self, user_id, *, profile, limit=5):
        from tomte.engine import Tomte
        from tomte.engine.core.memory import public_entity
        from tomte.engine.exceptions import TomteSecurityViolationError
        self._validate_user(user_id)
        with Tomte.operation_context() as (db, _):
            entities = (db.get_profile_entities(self.tenant_id, user_id) if profile else
                        db.get_frequent_entities(self.tenant_id, user_id, limit=limit))
            for entity in entities:
                if entity.get('tenant_id') != self.tenant_id or entity.get('user_id') != user_id:
                    raise TomteSecurityViolationError("Retrieved entity namespace mismatch")
            return [public_entity(entity) for entity in entities]

    def get_pending_messages(self, user_id: str, session_id: Optional[str] = None,
                             limit: int = 20) -> PendingMessagesPayload:
        """Read the newest unprocessed messages, returned in chronological order.

        Includes waiting and in-flight batches, excluding batches already committed
        to PostgreSQL. Does not consume messages or call a model. This is a live
        snapshot, not a message archive or a guarantee that extraction saves a fact.
        """
        from datetime import datetime, timezone
        import math
        from tomte.engine import Tomte
        from tomte.engine.exceptions import TomteInvalidIdError, TomteSecurityViolationError
        from tomte.engine.utils.keys import validate_id
        from tomte.engine.workflows.consolidation import ConsolidationWorkflow
        self._validate_user(user_id)
        self._validate_limit(limit)
        if session_id is not None and not validate_id(session_id):
            raise TomteInvalidIdError("Invalid session ID")
        with Tomte.operation_context() as (db, _):
            batches = Tomte.get_cache().get_pending_message_batches(
                self.tenant_id, user_id, session_id=session_id, limit=limit)
            messages, total = [], 0
            for batch in batches:
                sid = batch['session_id']
                if not validate_id(sid) or (session_id is not None and sid != session_id):
                    raise TomteSecurityViolationError("Pending message session mismatch")
                queued, processing = batch['queued'], batch['processing']
                count = batch['queued_count']
                if type(count) is not int or count < len(queued):
                    raise ValueError("Invalid pending message count")
                # Validate ownership even for a batch that will be excluded below.
                for item in queued + processing:
                    if (item.get('tenant_id'), item.get('user_id'), item.get('session_id')) != (
                            self.tenant_id, user_id, sid):
                        raise TomteSecurityViolationError("Pending message namespace mismatch")
                if processing and db.is_consolidation_processed(self.tenant_id, user_id,
                        ConsolidationWorkflow._batch_id(self.tenant_id, user_id, processing)):
                    processing = []
                total += count + len(processing)
                for status, items in (('processing', processing), ('queued', queued)):
                    for item in items:
                        stamp = item['timestamp']
                        if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                                or item['role'] not in ('user', 'assistant')
                                or not isinstance(item['content'], str)
                                or not isinstance(item['item_id'], str) or not item['item_id']):
                            raise ValueError("Invalid pending message")
                        messages.append((stamp, {
                            'message_id': item['item_id'], 'session_id': sid,
                            'role': item['role'], 'content': item['content'],
                            'received_at': datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
                            'status': status,
                        }))
            messages.sort(key=lambda row: row[0])
            selected = [row[1] for row in messages[-limit:]]
            return {'messages': selected, 'has_more': total > len(selected)}

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

    def get_relevant_entities(
        self, user_id: str, message: str, limit: Optional[int] = None
    ) -> list[dict]:
        """Read semantically similar entities without ingesting the message.

        Accepts any nonempty text, including statements. Uses query embeddings
        but no generative LLM. Union of summary and plain-field top-limit entities
        can return up to twice limit. Current facts are complete; large histories use
        existing index ranking and byte limits, with history_is_partial exposed.
        No chat/session state, activity or mention statistics are changed.
        Errors propagate; an empty list means retrieval completed with no rows.
        """
        return self._read_relevant(user_id, message, limit, include_unresolved=False)['entities']

    def get_context(self, user_id: str, message: str) -> str:
        """Return relevant background memory as text for the caller's agent.

        Includes stored entities, unresolved reports and pending conversation.
        Uses configured retrieval/format defaults and the shared window cache;
        no ingestion or generative LLM. Source labels and coverage notices remain
        in the text. Use get_relevant_memories for structured results or options.
        Retrieval errors propagate through the existing API.
        """
        return self.get_relevant_memories(user_id, message, include_pending=True)['context_text']

    def answer_question(self, user_id: str, question: str) -> str:
        """Answer from this user's memory, with at most three retrieval rounds.

        Uses the configured model and one pending snapshot. Returns answer text;
        never stores the question or generated answer. Failures propagate.
        """
        from contextlib import nullcontext
        from tomte.engine import Tomte
        from tomte.engine.core.answering import ANSWER_TIMEOUT_SECONDS
        from tomte.engine.core.config import settings
        from tomte.engine.exceptions import TomteInvalidInputError
        from tomte.engine.workers.control import WorkControl
        from tomte.engine.workflows.answering import AnswerWorkflow

        self._validate_user(user_id)
        if (not isinstance(question, str) or not question.strip()
                or len(question) > settings.max_input_length or '\x00' in question):
            raise TomteInvalidInputError('Invalid question')
        control = WorkControl(timeout=ANSWER_TIMEOUT_SECONDS)
        with Tomte.operation_context(control) as (db, llm):
            scope = Tomte._usage.scope(self.tenant_id, user_id, 'foreground') if Tomte._usage else nullcontext()
            with scope:
                return AnswerWorkflow(self.tenant_id, user_id, db, llm, Tomte.get_cache(), control).run(question)

    def get_relevant_memories(
        self, user_id: str, message: str, limit: Optional[int] = None, *,
        output_format: MemoryTextFormat | None = None,
        include_pending: bool = False,
    ) -> dict:
        """Read entities, unresolved statements and optional queued conversation.

        One shared query embedding, no generative LLM and no ingestion. With
        include_pending=True, embed uncached pending windows and update their
        disposable Redis cache. Queue and durable memories are never mutated.
        Unresolved statements and pending windows have separate result limits;
        neither is an assertion about a particular returned entity.
        context_text uses the configured memory_text_format unless overridden;
        the entity lists remain available as structured public data.
        """
        if type(include_pending) is not bool:
            from tomte.engine.exceptions import TomteInvalidInputError
            raise TomteInvalidInputError('include_pending must be a boolean')
        output_format = resolve_memory_text_format(output_format)
        result = self._read_relevant(user_id, message, limit, include_unresolved=True,
                                     include_pending=include_pending)
        from tomte.engine.workflows.memory_search import render_retrieved_memory
        result['context_text'] = render_retrieved_memory(result, output_format)
        return result

    def _read_relevant(self, user_id, message, limit, *, include_unresolved, include_pending=False):
        from contextlib import nullcontext
        from tomte.engine import Tomte
        from tomte.engine.core.config import settings
        from tomte.engine.core.memory import public_entity
        from tomte.engine.workflows.memory_search import read_memory_views
        from tomte.engine.exceptions import (
            TomteInvalidIdError, TomteInvalidInputError,
        )
        from tomte.engine.utils.keys import validate_id
        from tomte.engine.workers.control import WorkControl

        if not validate_id(self.tenant_id) or not validate_id(user_id):
            raise TomteInvalidIdError("Invalid retrieval tenant or user ID")
        if not isinstance(message, str) or not message.strip() or len(message) > settings.max_input_length:
            raise TomteInvalidInputError("Invalid message length")
        limit = settings.semantic_retrieval_limit if limit is None else limit
        if type(limit) is not int or limit < 1:
            raise TomteInvalidInputError("Retrieval limit must be a positive integer")

        control = WorkControl(timeout=60)
        with Tomte.operation_context(control) as (db, llm):
            scope = Tomte._usage.scope(self.tenant_id, user_id, "foreground") if Tomte._usage else nullcontext()
            with scope:
                pending = None
                if include_pending:
                    from tomte.engine.workflows.pending_search import PendingMemorySearch
                    pending = PendingMemorySearch(self.tenant_id,user_id,db,Tomte.get_cache(),control)
                vector = llm.embed_text(message)
                pending_result = pending.search(vector,llm) if pending is not None else {}
                views, unresolved = read_memory_views(self.tenant_id, user_id, message, vector, limit, db,
                    include_unresolved=include_unresolved)
                return {'entities': [public_entity(entity, entity.get("_history_selection")) for entity in views],
                        'unresolved_memories': unresolved, **pending_result}

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
