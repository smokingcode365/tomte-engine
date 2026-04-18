from typing import Optional, Dict, Any, List
from tomte.engine.core.enums import ChatSenderRole, MemoryDomain, ProviderDictionaryKeys
from tomte.engine import log
from tomte.engine.core.schema import ChatHookPayload
from tomte.engine.database.db_interface import DatabaseProvider
from tomte.engine.database.cache_interfaces import CacheProvider
from tomte.engine.llm.interfaces import LLMProvider
from tomte.engine.utils.security import SecurityGuard
from tomte.engine.utils.cognitive import CognitiveProcessor

class ChatHookWorkflow: 
    def __init__(
        self, 
        providers: Dict[ProviderDictionaryKeys, Any], 
        cognitive: Any, 
        security: Any
    ):
        self.db: DatabaseProvider = providers[ProviderDictionaryKeys.DB]
        self.cache: CacheProvider = providers[ProviderDictionaryKeys.CACHE]
        self.llm: LLMProvider = providers[ProviderDictionaryKeys.LLM]
        
        self.cog: CognitiveProcessor = cognitive
        self.security: SecurityGuard = security

    def on_chat(
        self, 
        tenant_id: str, 
        session_id: str, 
        message: str, 
        sender_type: ChatSenderRole,
        user_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Main entry point. Routes the message based on sender and auth status.
        """
        # 1. Front-Door Security
        clean_msg = self._handle_security(message, user_id)
        if not clean_msg:
            return self._build_payload(error="Security violation detected.")

        # 2. AI Path
        if sender_type == ChatSenderRole.ASSISTANT:
            return self._handle_assistant_sender(tenant_id, session_id, user_id, clean_msg)

        # 3. Guest Path
        if not user_id:
            return self._handle_guest_sender(tenant_id, session_id, clean_msg)

        # 4. Registered User Path (The Engine)
        return self._handle_registered_user(tenant_id, session_id, user_id, clean_msg)

    # ==========================================
    # ROUTING HELPERS
    # ==========================================

    def _handle_security(self, message: str, user_id: Optional[str]) -> Optional[str]:
        """Sanitizes input and checks for injection. Returns None if blocked."""
        clean_msg = self.security.sanitize_input(message)
        if self.security.check_for_injection(clean_msg):
            log.warn(f"Security: Injection attempt blocked for User: {user_id or 'Guest'}")
            return None
        return clean_msg

    def _handle_assistant_sender(self, tenant_id: str, session_id: str, user_id: Optional[str], message: str) -> None:
        """Logs AI responses to history. Returns None as no LLM generation is needed."""
        log.debug(f"AI Path: Logging response to cache for {user_id or 'Guest'}")        
        # Note: CacheProvider automatically handles the dual-write to the 
        # consolidation queue here if user_id is present.
        self.cache.append_chat_history(tenant_id, session_id, message, role=ChatSenderRole.ASSISTANT, user_id=user_id)
        return self._build_payload(user_id=user_id,clean_message=message)

    def _handle_guest_sender(self, tenant_id: str, session_id: str, message: str) -> Dict[str, Any]:
        """Handles fast-path guest logic. No database or vector operations."""
        log.debug(f"Guest Path: Processing session {session_id} for tenant {tenant_id}")
        
        # History Handshake (Append Last Rule)
        chat_history = self.cache.get_chat_history(tenant_id, session_id, user_id=None)
        self.cache.append_chat_history(tenant_id, session_id, message, role=ChatSenderRole.USER, user_id=None)
        
        return self._build_payload(
            context_text="", 
            chat_history=chat_history, 
            clean_message=message, 
            user_id=None
        )

    def _handle_registered_user(self, tenant_id: str, session_id: str, user_id: str, message: str) -> Dict[str, Any]:
        """Executes the full RAG pipeline, persona logic, and history tracking."""
        log.debug(f"User Path: Running full RAG for {user_id}")
        
        # 1. Heartbeat
        self.db.upsert_user_activity(tenant_id, user_id)

        # 2. RAG Assembly
        persona = self._get_or_refresh_persona(tenant_id, user_id)
        recent, frequent, semantic = self._fetch_live_pulse(tenant_id, user_id, message)
        
        context_text = self._build_knowledge_context(persona, frequent, recent, semantic)

        # 3. History Handshake (Append Last Rule)
        chat_history = self.cache.get_chat_history(tenant_id, session_id, user_id=user_id)
        self.cache.append_chat_history(tenant_id, session_id, message, role=ChatSenderRole.USER, user_id=user_id)
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
            log.info(f"Cache Miss: Fetching Persona from DB for {user_id}")
            entities = self.db.get_important_entities(tenant_id, user_id,domain=MemoryDomain.IDENTITY)
            self.cache.set_persona_entities(tenant_id, user_id, entities)
        return entities

    def _fetch_live_pulse(self, tenant_id: str, user_id: str, message: str) -> tuple:
        """Embeds the prompt and fetches the semantic, recent, and frequent memory layers."""
        query_vector = self.llm.embed_text(message)
        
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