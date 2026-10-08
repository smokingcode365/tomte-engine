from typing import Optional, Dict, Any, List
from tomte.engine.core.config import settings
from tomte.engine.core.enums import ChatSenderRole, MemoryDomain, ProviderDictionaryKeys
from tomte.engine.exceptions import TomteInvalidIdError, TomteInvalidInputError, TomteSecurityViolationError
from tomte.engine import tomtelog, Tomte
from tomte.engine.core.schema import ChatHookPayload
from tomte.engine.database.db_interface import DatabaseProvider
from tomte.engine.database.cache_interfaces import CacheProvider
from tomte.engine.llm.interfaces import LLMProvider
from tomte.engine.utils.keys import validate_id
from tomte.engine.utils.security import SecurityGuard, get_security_guard
from tomte.engine.utils.cognitive import CognitiveProcessor

class ChatHookWorkflow: 
    def __init__(
        self, 
    ):
        self.db: DatabaseProvider = Tomte.get_db()
        self.cache: CacheProvider = Tomte.get_cache()
        self.llm: LLMProvider = Tomte.get_llm()
        self.cog: CognitiveProcessor = CognitiveProcessor()
        self.security: SecurityGuard = get_security_guard()

    def on_chat(
        self, 
        tenant_id: str, 
        session_id: str, 
        message: str, 
        sender_type: ChatSenderRole,
        user_id: Optional[str] = None
    ) -> ChatHookPayload :
        """
        Main entry point. Routes the message based on sender and auth status.
        """
        # 1. Front-Door Security
        self._input_validation(tenant_id, session_id, user_id, sender_type, message)
        clean_msg = self._handle_security_and_sanitization(message, user_id)
        # 2. AI Path
        if sender_type == ChatSenderRole.ASSISTANT:
            return self._handle_assistant_sender(tenant_id, session_id, user_id, clean_msg)

        # 3. Guest Path
        if not user_id:
            return self._handle_guest_sender(tenant_id, session_id, clean_msg)

        # 4. Registered User Path (The Engine)
        return self._handle_registered_user(tenant_id, session_id, user_id, clean_msg)

    def _input_validation(self, tenant_id: str, session_id: str, user_id: Optional[str],sender_type: ChatSenderRole,message: str) :
        """Validates the input IDs."""
        if not validate_id(tenant_id) or not validate_id(session_id) or not validate_id(user_id, can_be_empty=True):
            raise TomteInvalidIdError(f"Invalid ID: {tenant_id}, {session_id}, {user_id}")
        if not message or len(message.strip()) == 0:
            raise TomteInvalidInputError("Message cannot be empty.")
        if sender_type not in [ChatSenderRole.ASSISTANT, ChatSenderRole.USER]:
            raise TomteInvalidInputError(f"Invalid sender type: {sender_type}")
    # ==========================================
    # ROUTING HELPERS
    # ==========================================

    def _handle_security_and_sanitization(self, message: str, user_id: Optional[str]) -> str:
        """Sanitizes input and checks for injection."""
        clean_msg = self.security.sanitize_input(message)
        if not self.security.check_for_injection(clean_msg):
            tomtelog.warn(f"Security: Injection attempt blocked for User: {user_id or 'Guest'}")
            raise TomteSecurityViolationError(f"Security violation detected.Security: Injection attempt blocked for User: {user_id or 'Guest'}")
        return clean_msg

    def _handle_assistant_sender(self, tenant_id: str, session_id: str, user_id: Optional[str], message: str) -> ChatHookPayload:
        """Logs AI responses to history."""
        tomtelog.debug(f"AI Path: Logging response to cache for {user_id or 'Guest'}")        
        self.cache.append_chat_history(tenant_id, session_id, message, role=ChatSenderRole.ASSISTANT.value, user_id=user_id)
        if user_id:
            self.cache.enqueue_for_consolidation(tenant_id, user_id, session_id, message, role=ChatSenderRole.ASSISTANT.value)
        return self._build_payload(user_id=user_id,clean_message=message)

    def _handle_guest_sender(self, tenant_id: str, session_id: str, message: str) -> ChatHookPayload:
        """Handles fast-path guest logic. No database or vector operations."""
        tomtelog.debug(f"Guest Path: Processing session {session_id} for tenant {tenant_id}")
        
        # History Handshake (Append Last Rule)
        chat_history = self.cache.get_chat_history(tenant_id, session_id, user_id=None,limit=settings.history_chat_number)
        self.cache.append_chat_history(tenant_id, session_id, message, role=ChatSenderRole.USER.value, user_id=None)
        return self._build_payload(
            context_text="", 
            chat_history=chat_history, 
            clean_message=message, 
            user_id=None
        )

    def _handle_registered_user(self, tenant_id: str, session_id: str, user_id: str, message: str) -> ChatHookPayload:
        """Executes the full RAG pipeline, persona logic, and history tracking."""
        tomtelog.debug(f"User Path: Running full RAG for {user_id}")
        query_vector = self.llm.embed_text(message)
        # 1. Heartbeat
        self.db.upsert_user_activity(tenant_id, user_id)

        # 2. RAG Assembly
        persona = self._get_or_refresh_persona(tenant_id, user_id)
        recent, frequent, semantic = self._fetch_live_pulse(tenant_id, user_id, query_vector)
        
        context_text = self._build_knowledge_context(persona, frequent, recent, semantic)

        # 3. History Handshake (Append Last Rule)
        chat_history = self.cache.get_chat_history(tenant_id, session_id, user_id=user_id,limit=settings.history_chat_number)
        self.cache.append_chat_history(tenant_id, session_id, message, role=ChatSenderRole.USER.value, user_id=user_id)
        self.cache.enqueue_for_consolidation(tenant_id, user_id, session_id, message, role=ChatSenderRole.USER.value,vector=query_vector)
        return self._build_payload(
            context_text=context_text,
            chat_history=chat_history,
            clean_message=message,
            user_id=user_id
        )

    # ==========================================
    # DATA RETRIEVAL & COGNITIVE HELPERS
    # ==========================================

    def _get_or_refresh_persona(self, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Handles cache-first lookup for the user's permanent identity."""
        entities = self.cache.get_persona_entities(tenant_id, user_id)
        if not entities:
            tomtelog.info(f"Cache Miss: Fetching Persona from DB for {user_id}")
            entities = self.db.get_important_entities(tenant_id, user_id,domain=MemoryDomain.IDENTITY)
            self.cache.set_persona_entities(tenant_id, user_id, entities)
        return entities

    def _fetch_live_pulse(self, tenant_id: str, user_id: str, query_vector: List[float]) -> tuple:
        """Embeds the prompt and fetches the semantic, recent, and frequent memory layers."""        
        recent = self.db.get_recent_entities(tenant_id, user_id)
        frequent = self.db.get_frequent_entities(tenant_id, user_id)
        semantic = self.db.semantic_retrieval(tenant_id, user_id, query_vector)
        return recent, frequent, semantic

    def _build_knowledge_context(self, persona: List[Dict], frequent: List[Dict], recent: List[Dict], semantic: List[Dict]) -> str:
        """Deduplicates the 4 streams of knowledge and formats them for the LLM."""
        merged_entities = self.cog.merge_entities(persona, frequent, recent, semantic)
        return self.cog.format_entities(merged_entities)


    def _build_payload(
        self, 
        context_text: str = "", 
        chat_history: Optional[List[Dict[str, Any]]] = None, 
        clean_message: str = "", 
        user_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> ChatHookPayload:
        """Centralized factory for the Chat Hook return payload to avoid magic strings."""
        if error:
            return ChatHookPayload(error=error)
            
        return ChatHookPayload(
            context_text=context_text,
            chat_history=chat_history or [],
            user_id=user_id,
            clean_message=clean_message
        )
