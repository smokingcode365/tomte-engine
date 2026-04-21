"""
Default Redis implementation of the CacheProvider.
"""
import uuid
import redis
import json
import time
from typing import List, Optional, Dict, Any
from tomte.engine.core.config import settings
from tomte.engine.database.cache_interfaces import CacheProvider
from tomte.engine import tomtelog
from tomte.engine.utils.keys import generate_common_key, generate_persona_key, get_section_key_active_sessions
from tomte.engine.utils.keys import generate_consolidation_key, generate_session_ref
from tomte.engine.utils.security import get_security_guard

class RedisCacheProvider(CacheProvider):
    def __init__(self, url: str = "redis://localhost:6379/0", **kwargs):
        """
        Initializes the Redis connection pool. 
        kwargs can be used to pass passwords, SSL certs, etc.
        """
        self.client = redis.Redis.from_url(url, decode_responses=True, **kwargs)

    # ==========================================
    # YOUR ORIGINAL UTILITY METHODS (UNTOUCHED)
    # ==========================================

    def acquire_lock(self, lock_key: str,lock_value: str, ttl_seconds: int) -> bool:
        """
        Uses SETNX to acquire a lock. If it returns True, the worker is safe to proceed.
        If False, another worker is already processing this tenant/user.
        """
        # nx=True means "Only set if it does not already exist"
        acquired = self.client.set(lock_key, lock_value, nx=True, ex=ttl_seconds)
        return bool(acquired)

 
    def release_lock(self, lock_key: str, lock_value: str) -> bool:
        """
        Safely releases the lock using an atomic Lua script.
        Prevents a stale worker from deleting a new worker's lock.
        """
        lua_script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        # eval(script, num_keys, key1, arg1)
        result = self.client.eval(lua_script, 1, lock_key, lock_value)
        return bool(result)

    def set_value(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> None:
        self.client.set(key, value, ex=ttl_seconds)

    def get_value(self, key: str) -> Optional[str]:
        result = self.client.get(key)
        if result is None:
            return None
        return str(result)

    def push_to_list(self, list_key: str, message: str) -> None:
        """Appends a chat message to the right side of the list (RPUSH)."""
        self.client.rpush(list_key, message)

    def get_list(self, list_key: str, limit: int) -> List[str]:
        """Fetches the most recent X messages from the list."""
        # Redis LRANGE is inclusive. To get last 10, we want index -limit to -1
        if limit <= 0:
            return []
        
        # E.g., LRANGE list_key -10 -1
        results = self.client.lrange(list_key, -limit, -1)
        return results if results else []

    def add_to_set(self, set_key: str, values: List[str]) -> None:
        """
        Adds values to a Redis Set (SADD). 
        Redis automatically deduplicates identical strings.
        """
        if not values:
            return
        self.client.sadd(set_key, *values)

    def pop_set(self, set_key: str) -> List[str]:
        """
        Returns all members of a Set (SMEMBERS) and instantly deletes the key,
        clearing the cache for the next cycle.
        """
        pipeline = self.client.pipeline()
        pipeline.smembers(set_key)
        pipeline.delete(set_key)
        results = pipeline.execute()
        
        # Results[0] is the output of smembers (a set of strings)
        return list(results[0]) if results[0] else []


    # ==========================================
    # NEW LOGIC: CHAT HOOK & DUAL-WRITE
    # ==========================================

    

    def append_chat_history(self, tenant_id: str, session_id: str, message: str, role: str, user_id: Optional[str] = None) -> None:
        """
        Formats the message to JSON, trims the context window, and handles dual-writes.
        """
        history_key = generate_common_key(tenant_id, session_id, user_id)
        
        payload = {"role": role, "content": message, "timestamp": time.time()}
        json_payload = json.dumps(payload)

        # 1. Pipeline for Live Context (Cap at 20 msgs, TTL 24h)
        pipeline = self.client.pipeline()
        pipeline.rpush(history_key, json_payload)
        pipeline.ltrim(history_key, -settings.history_chat_number, -1)
        pipeline.expire(history_key, 86400) 
        pipeline.execute()


    def get_chat_history(
        self, 
        tenant_id: str, 
        session_id: str, 
        user_id: Optional[str] = None, 
        limit: int = settings.history_chat_number
    ) -> List[Dict[str, Any]]:
        """Retrieves history using safe JSON parsing to prevent 'Poison Pill' crashes."""
        history_key = generate_common_key(tenant_id, session_id, user_id)
        raw_items = self.get_list(history_key, limit)
        parsed_history = []
        for item in raw_items:
            try:
                parsed_history.append(json.loads(item))
            except json.JSONDecodeError as e:
                # Log the corruption so ops can investigate, but do NOT crash the user's chat
                tomtelog.error(f"Corrupted JSON in Redis history key: {history_key}", exc=e)
                continue
                
        return parsed_history

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
        Saves message data to a session-specific queue and updates the global 
        activity tracker for background workers.
        """
        # 1. Define the unique session identifier for the worker
        session_ref = generate_session_ref(tenant_id, session_id, user_id)
        queue_key = generate_consolidation_key(tenant_id, user_id, session_id)
        
        # 2. Build the heavy payload (includes vector and metadata)
        # This data is purely for consolidation, not for UI display.
        payload = {
            "tenant_id": tenant_id, # Included as requested for worker context
            "user_id": user_id,
            "role": role,
            "content": get_security_guard().scrub_pii(message),
            "vector": vector,
            "timestamp": time.time()
        }
        json_payload = json.dumps(payload)

        # 3. Execute atomic update
        pipeline = self.client.pipeline()
        # Add to the chronological list
        pipeline.rpush(queue_key, json_payload)
        # Update the ZSET score (current time) to track the 'last modified' state
        # The worker uses this to find sessions idle for > 5 minutes.
        pipeline.zadd(get_section_key_active_sessions(), {session_ref: payload["timestamp"]})
        pipeline.execute()

    # --- Persona Operations ---
    
    def get_persona_entities(self, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Uses your get_value method to pull raw JSON."""
        key = generate_persona_key(tenant_id, user_id)
        data = self.get_value(key) # Using YOUR base method
        if data:
            try:
                return json.loads(data)
            except Exception as e:
                tomtelog.error(f"Failed to decode persona entities for key {key}", exc=e)
                return []
        return []

    def set_persona_entities(self, tenant_id: str, user_id: str, entities: List[Dict[str, Any]], ttl: int = 1800) -> None:
        """Uses your set_value method to store JSON."""
        key = generate_persona_key(tenant_id, user_id)
        self.set_value(key, json.dumps(entities), ttl_seconds=ttl) # Using YOUR base method