from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

class CacheProvider(ABC):
    """
    Abstract base class for Cache Providers in the Tomte Engine.
    Defines the contract for base utility methods, session-isolated chat history,
    dual-write consolidation queues, and persona caching.
    """

    # ==========================================
    # UTILITY METHODS
    # ==========================================

    @abstractmethod
    def acquire_lock(self, lock_key: str, lock_value: str, ttl_seconds: int) -> bool:
        """Attempts to acquire a distributed lock using a unique lock_value. Returns True if successful."""
        pass

    @abstractmethod
    def release_lock(self, lock_key: str, lock_value: str) -> bool:
        """Safely releases a distributed lock only if the lock_value matches. Returns True if deleted."""
        pass

    @abstractmethod
    def set_value(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> None:
        """Sets a raw string value, optionally with a Time-To-Live (TTL)."""
        pass

    @abstractmethod
    def get_value(self, key: str) -> Optional[str]:
        """Retrieves a raw string value by key. Returns None if missing."""
        pass


    @abstractmethod
    def push_to_list(self, list_key: str, message: str) -> None:
        """Appends a string to the end of a list structure."""
        pass

    @abstractmethod
    def get_list(self, list_key: str, limit: int) -> List[str]:
        """Fetches the most recent `limit` strings from a list."""
        pass

    @abstractmethod
    def add_to_set(self, set_key: str, values: List[str]) -> None:
        """Adds multiple values to a deduplicated set structure."""
        pass

    @abstractmethod
    def pop_set(self, set_key: str) -> List[str]:
        """Returns all members of a set and instantly clears it."""
        pass

    # ==========================================
    # CHAT HOOK & DUAL-WRITE METHODS
    # ==========================================

    @abstractmethod
    def append_chat_history(
        self, 
        tenant_id: str, 
        session_id: str, 
        message: str, 
        role: str, 
        user_id: Optional[str] = None
    ) -> None:
        """
        Appends a message to the live chat context.
        Implementations MUST:
        1. Maintain session isolation based on provider config.
        """
        pass

    @abstractmethod
    def get_chat_history(
        self, 
        tenant_id: str, 
        session_id: str, 
        user_id: Optional[str] = None, 
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """
        Retrieves the most recent messages for the LLM prompt.
        Returns a list of dictionaries containing 'role', 'content', and 'timestamp'.
        """
        pass

    @abstractmethod
    def enqueue_for_consolidation(
        self, 
        tenant_id: str, 
        user_id: str, 
        session_id: str, 
        message: str, 
        role: str, 
        vector: Optional[List[float]] = None
    ) -> None:
        """
        Appends high-fidelity message data (including vectors) to a background 
        consolidation queue. This is isolated from the live chat context window.
        """
        pass
    # ==========================================
    # PERSONA CACHING METHODS
    # ==========================================

    @abstractmethod
    def get_persona_entities(self, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
        """
        Retrieves the cached representation of the user's Persona entities.
        Returns an empty list if there is a cache miss.
        """
        pass

    @abstractmethod
    def set_persona_entities(
        self, 
        tenant_id: str, 
        user_id: str, 
        entities: List[Dict[str, Any]], 
        ttl: int = 1800
    ) -> None:
        """
        Caches the user's Persona entities (raw dictionaries) with a standard TTL.
        """
        pass



    @abstractmethod
    def get_next_idle_session_4_consolidation(self, idle_seconds: int = 300) -> Optional[str]:
        """Finds the absolute oldest session reference that has been idle."""
        ...

    @abstractmethod
    def pop_session_chat_log_4_consolidation(self, tenant_id: str, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        """Pulls the chronological chat log payload for consolidation."""
        ...

    @abstractmethod
    def clear_session_4_consolidation(self, tenant_id: str, user_id: str, session_id: str) -> None:
        """Removes the chat log and removes the session from the active pool."""
        ...