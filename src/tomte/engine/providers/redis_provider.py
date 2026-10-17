"""
Default Redis implementation of the CacheProvider.
"""
import uuid
import redis
import json
import time
from typing import List, Optional, Dict, Any, Tuple
from tomte.engine.core.config import settings
from tomte.engine.core.extraction import RECENT_CONTEXT_TURNS
from tomte.engine.database.cache_interfaces import CacheProvider
from tomte.engine import tomtelog
from tomte.engine.exceptions import TomteSessionStateUnavailable
from tomte.engine.utils.keys import generate_session_entity_key
from tomte.engine.utils.keys import generate_common_key, generate_persona_key, get_section_key_active_sessions
from tomte.engine.utils.keys import generate_consolidation_key, generate_consolidation_batch_key, generate_session_ref, parse_session_ref
from tomte.engine.utils.security import get_security_guard

class RedisCacheProvider(CacheProvider):
    def __init__(self, url: str = "redis://localhost:6379/0", **kwargs):
        """
        Initializes the Redis connection pool. 
        kwargs can be used to pass passwords, SSL certs, etc.
        """
        kwargs.setdefault('socket_connect_timeout', 2)
        kwargs.setdefault('socket_timeout', 2)
        self.client = redis.Redis.from_url(url, decode_responses=True, **kwargs)
        self._scan_offset = 0

    def promote_session_entities(self, tenant_id, user_id, session_id, entity_ids,
                                 retention, ttl):
        key = generate_session_entity_key(tenant_id, user_id, session_id)
        if (type(retention) is not int or type(ttl) is not int or retention < 1 or ttl < 1
                or any(not isinstance(eid, str) or not eid for eid in entity_ids)):
            raise ValueError("Invalid session entity update")
        # First input ID has highest priority. Reverse insertion preserves that order.
        ids = list(dict.fromkeys(entity_ids))
        lua = """
        for i = #ARGV, 3, -1 do
            redis.call('lrem', KEYS[1], 0, ARGV[i])
            redis.call('lpush', KEYS[1], ARGV[i])
        end
        if redis.call('exists', KEYS[1]) == 1 then
            redis.call('ltrim', KEYS[1], 0, tonumber(ARGV[1]) - 1)
            redis.call('expire', KEYS[1], tonumber(ARGV[2]))
        end
        return 1
        """
        try:
            self.client.eval(lua, 1, key, retention, ttl, *ids)
        except (redis.ConnectionError, redis.TimeoutError):
            raise TomteSessionStateUnavailable("Session entity update unavailable") from None

    def get_session_entity_ids(self, tenant_id, user_id, session_id, limit):
        key = generate_session_entity_key(tenant_id, user_id, session_id)
        if type(limit) is not int or limit < 1:
            raise ValueError("Session entity limit must be positive")
        try:
            return self.client.lrange(key, 0, limit - 1)
        except (redis.ConnectionError, redis.TimeoutError):
            raise TomteSessionStateUnavailable("Session entity read unavailable") from None

    def touch_session_entities(self, tenant_id, user_id, session_id, ttl):
        key = generate_session_entity_key(tenant_id, user_id, session_id)
        if type(ttl) is not int or ttl < 1:
            raise ValueError("Session entity TTL must be positive")
        try:
            self.client.expire(key, ttl)  # EXPIRE never creates a missing key.
        except (redis.ConnectionError, redis.TimeoutError):
            raise TomteSessionStateUnavailable("Session entity touch unavailable") from None

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
                tomtelog.error("Corrupted JSON in Redis history")
                continue
                
        return parsed_history

    def enqueue_for_consolidation(
        self, 
        tenant_id: str, 
        user_id: str, 
        session_id: str, 
        message: str, 
        role: str, 
        vector: Optional[List[float]] = None,
        recent_context: Optional[List[Dict[str, Any]]] = None
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
            "item_id": str(uuid.uuid4()),
            "session_id": session_id,
            "tenant_id": tenant_id, # Included as requested for worker context
            "user_id": user_id,
            "role": role,
            "content": get_security_guard().scrub_pii(message),
            "vector": vector,
            "timestamp": time.time(),
            "recent_context": [{"role": turn["role"],
                                "content": get_security_guard().scrub_pii(turn["content"])}
                               for turn in (recent_context or [])[-RECENT_CONTEXT_TURNS:]]
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
                tomtelog.error("Failed to decode persona entities")
                return []
        return []

    def set_persona_entities(self, tenant_id: str, user_id: str, entities: List[Dict[str, Any]], ttl: int = 1800) -> None:
        """Uses your set_value method to store JSON."""
        key = generate_persona_key(tenant_id, user_id)
        self.set_value(key, json.dumps(entities), ttl_seconds=ttl) # Using YOUR base method


    def get_next_idle_session_4_consolidation(self, idle_seconds: int = 300) -> Optional[Tuple[str, str, str]]:
        """
        Fetches exactly ONE session that has been idle for the required time.
        """
        cutoff_time = time.time() - idle_seconds
        
        # LIMIT 0 1 ensures we only grab the absolute oldest idle session
        idle_refs = self.client.zrangebyscore(
            get_section_key_active_sessions(), 
            "-inf", 
            cutoff_time,
            start=0,
            num=1
        )
        return parse_session_ref(idle_refs[0]) if idle_refs else None

    def pop_session_chat_log_4_consolidation(self, tenant_id: str, user_id: str, session_id: str) -> list[dict]:
        """Freeze one batch; retries read it while later messages remain queued."""
        queue_key = generate_consolidation_key(tenant_id, user_id, session_id)
        batch_key = generate_consolidation_batch_key(tenant_id, user_id, session_id)
        lua = """
        if redis.call('exists', KEYS[2]) == 0 then
            if redis.call('exists', KEYS[1]) == 0 then return {} end
            redis.call('rename', KEYS[1], KEYS[2])
        end
        local items = redis.call('lrange', KEYS[2], 0, -1)
        for i, raw in ipairs(items) do
            local ok, item = pcall(cjson.decode, raw)
            if not ok or type(item) ~= 'table' then return items end
            -- Upgrade old queues once, storing IDs so retries use the same batch.
            if not item.item_id or not item.session_id then
                item.item_id = item.item_id or (ARGV[1] .. ':' .. i)
                item.session_id = item.session_id or ARGV[2]
                raw = cjson.encode(item)
                redis.call('lset', KEYS[2], i - 1, raw)
                items[i] = raw
            end
        end
        return items
        """
        raw_items = self.client.eval(lua, 2, queue_key, batch_key, str(uuid.uuid4()), session_id)
        return [json.loads(item) for item in raw_items]

    def clear_session_4_consolidation(self, tenant_id: str, user_id: str, session_id: str, chat_items: Optional[List[Dict[str, Any]]] = None) -> bool:
        """Acknowledge only the supplied snapshot, never newly queued messages."""
        queue_key = generate_consolidation_key(tenant_id, user_id, session_id)
        batch_key = generate_consolidation_batch_key(tenant_id, user_id, session_id)
        session_ref = generate_session_ref(tenant_id, session_id, user_id)
        ids = [item['item_id'] for item in chat_items] if chat_items else []
        lua = """
        local expected = cjson.decode(ARGV[2])
        local items = redis.call('lrange', KEYS[2], 0, -1)
        if #expected > 0 then
            if #items ~= #expected then return 0 end
            for i, raw in ipairs(items) do
                if cjson.decode(raw).item_id ~= expected[i] then return 0 end
            end
            redis.call('del', KEYS[2])
        elseif #items > 0 then
            return 0
        end
        if redis.call('llen', KEYS[1]) == 0 and redis.call('llen', KEYS[2]) == 0 then
            redis.call('zrem', KEYS[3], ARGV[1])
        end
        return 1
        """
        return bool(self.client.eval(lua, 3, queue_key, batch_key, get_section_key_active_sessions(), session_ref, json.dumps(ids)))

    def renew_lock(self, lock_key, token, ttl_seconds=120):
        return bool(self.client.eval("if redis.call('get',KEYS[1])==ARGV[1] then return redis.call('expire',KEYS[1],ARGV[2]) else return 0 end",
                                     1, lock_key, token, ttl_seconds))

    @staticmethod
    def worker_state_key(tenant, user):
        return f'Tomte:worker:state:{tenant}:{user}'

    def worker_state(self, tenant, user):
        raw = self.client.get(self.worker_state_key(tenant, user))
        return json.loads(raw) if raw else None

    def worker_terminal(self, tenant, user, batch):
        raw = self.client.get(f'Tomte:worker:terminal:{tenant}:{user}:{batch}')
        return json.loads(raw) if raw else None

    def worker_candidates(self, now, idle_seconds, limit=64):
        # Rotate the bounded idle scan: a locked oldest user cannot starve others.
        due = self.client.zrangebyscore('Tomte:worker:due', '-inf', now, start=0, num=limit)
        idle = self.client.zrangebyscore(get_section_key_active_sessions(), '-inf', now-idle_seconds,
                                        start=self._scan_offset, num=limit)
        self._scan_offset = self._scan_offset + len(idle) if len(idle) == limit else 0
        from tomte.engine.utils.keys import validate_id
        scopes = [tuple(parse_session_ref(ref)) for ref in dict.fromkeys(due + idle)]
        return [scope for scope in scopes if len(scope) == 3 and all(validate_id(value) for value in scope)]

    def worker_save(self, tenant, user, token, state, due):
        ref = generate_session_ref(tenant, state['session'], user)
        lua = """
        if redis.call('get', KEYS[1]) ~= ARGV[1] then return 0 end
        redis.call('set', KEYS[2], ARGV[2])
        redis.call('zadd', KEYS[3], ARGV[3], ARGV[4])
        return 1
        """
        return bool(self.client.eval(lua, 3, f'Tomte:lock:consolidation:{tenant}:{user}',
            self.worker_state_key(tenant, user), 'Tomte:worker:due', token, json.dumps(state), due, ref))

    def worker_batch(self, tenant, user, session):
        try:
            items = self.pop_session_chat_log_4_consolidation(tenant, user, session)
        except (json.JSONDecodeError, TypeError):
            items = None
        raw = self.client.lrange(generate_consolidation_batch_key(tenant, user, session), 0, -1)
        return items, raw

    def worker_finish(self, tenant, user, token, state, raw_items, outcome):
        """Token, state and exact frozen snapshot checked in the same ACK/SKIP."""
        session = state['session']
        ref = generate_session_ref(tenant, session, user)
        lua = """
        if redis.call('get', KEYS[1]) ~= ARGV[1] then return 0 end
        local state = redis.call('get', KEYS[2])
        if not state or cjson.decode(state).batch ~= ARGV[2] then return 0 end
        local expected = cjson.decode(ARGV[3])
        local actual = redis.call('lrange', KEYS[3], 0, -1)
        if #expected ~= #actual then return 0 end
        for i, raw in ipairs(actual) do if raw ~= expected[i] then return 0 end end
        if ARGV[6] == 'skipped' then
            if redis.call('exists', KEYS[7]) == 0 then redis.call('set', KEYS[7], ARGV[5]) end
        else
            redis.call('set', KEYS[7], ARGV[5], 'EX', 604800)
        end
        redis.call('del', KEYS[2], KEYS[3])
        redis.call('zrem', KEYS[4], ARGV[4])
        if redis.call('llen', KEYS[5]) == 0 then redis.call('zrem', KEYS[6], ARGV[4]) end
        return 1
        """
        terminal = {'batch': state['batch'], 'outcome': outcome, 'attempts': state['attempts'],
                    'reason': state.get('reason'), 'finished_at': time.time()}
        return bool(self.client.eval(lua, 7, f'Tomte:lock:consolidation:{tenant}:{user}',
            self.worker_state_key(tenant, user), generate_consolidation_batch_key(tenant, user, session),
            'Tomte:worker:due', generate_consolidation_key(tenant, user, session),
            get_section_key_active_sessions(), f'Tomte:worker:terminal:{tenant}:{user}:{state["batch"]}',
            token, state['batch'], json.dumps(raw_items), ref, json.dumps(terminal), outcome))

    def pending_input_chars(self, tenant, user):
        """Scoped queued + waiting snapshot input. Running work is reported separately."""
        from tomte.engine.utils.keys import validate_id
        if not validate_id(tenant) or not validate_id(user):
            raise ValueError('Invalid usage namespace')
        state = self.worker_state(tenant, user)
        total = 0
        for ref in self.client.zrange(get_section_key_active_sessions(), 0, -1):
            t, u, s = parse_session_ref(ref)
            if (t, u) != (tenant, user):
                continue
            keys = [generate_consolidation_key(t, u, s)]
            if not state or state['session'] != s or state['status'] == 'WAIT':
                keys.append(generate_consolidation_batch_key(t, u, s))
            for key in keys:
                for raw in self.client.lrange(key, 0, -1):
                    total += len(json.loads(raw).get('content', ''))
        return total
