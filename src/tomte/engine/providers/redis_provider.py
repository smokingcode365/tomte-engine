"""
Default Redis implementation of the CacheProvider.
"""
import redis
from typing import List, Optional
from tomte.engine.database.interfaces import CacheProvider

class RedisCacheProvider(CacheProvider):
    def __init__(self, url: str = "redis://localhost:6379/0", **kwargs):
        """
        Initializes the Redis connection pool. 
        kwargs can be used to pass passwords, SSL certs, etc.
        """
        self.client = redis.Redis.from_url(url, decode_responses=True, **kwargs)


    def acquire_lock(self, lock_key: str, ttl_seconds: int) -> bool:
        """
        Uses SETNX to acquire a lock. If it returns True, the worker is safe to proceed.
        If False, another worker is already processing this tenant/user.
        """
        # nx=True means "Only set if it does not already exist"
        acquired = self.client.set(lock_key, "LOCKED", nx=True, ex=ttl_seconds)
        return bool(acquired)

    def set_value(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> None:
        self.client.set(key, value, ex=ttl_seconds)

    def get_value(self, key: str) -> Optional[str]:
        result = self.client.get(key)
        if result is None:
            return None
        return str(result)
        
    def release_lock(self, lock_key: str) -> None:
        self.client.delete(lock_key)

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