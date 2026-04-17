from typing import Protocol, List, Optional

class CacheProvider(Protocol):
    """The contract for short-term memory, fast-path routing, and worker locks."""

    # --- Mutex Locks (For Consolidation Worker) ---
    def acquire_lock(self, lock_key: str, ttl_seconds: int) -> bool:
        ...
        
    def release_lock(self, lock_key: str) -> None:
        ...
    
    # --- Basic Key-Value (For Session Cores) ---
    def set_value(self, key: str, value: str, ttl_seconds: Optional[int] = None) -> None:
        ...

    def get_value(self, key: str) -> Optional[str]:
        ...

    # --- Lists (For Guest Chat Transcripts) ---
    def push_to_list(self, list_key: str, message: str) -> None:
        ...

    def get_list(self, list_key: str, limit: int) -> List[str]:
        ...
        
    # --- Sets (For Taxonomy/Lexicon Deduplication) ---
    def add_to_set(self, set_key: str, values: List[str]) -> None:
        ...
        
    def pop_set(self, set_key: str) -> List[str]:
        ...