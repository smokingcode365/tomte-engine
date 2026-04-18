from typing import Protocol, List, Optional

from typing import Protocol, List, Dict, Any, Optional
from uuid import UUID

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




class DatabaseProvider(Protocol):
    """
    The contract for the Tomte Engine's permanent Knowledge Graph.
    Supports multi-tenancy via (tenant_id, user_id) composite logic.
    """

    # --- JIT Provisioning & Health ---
    def ensure_infrastructure(self, vector_dimension: int) -> None:
        """Verifies extensions and creates tables if they don't exist."""
        ...

    def is_ready(self) -> bool:
        """Checks if the database is connected and tables are provisioned."""
        ...

    # --- Workflow 1: Chat Hook (Read Ops) ---
    def upsert_user_activity(self, tenant_id: str, user_id: str) -> None:
        """Heartbeat: Registers user/tenant combo and updates timestamp."""
        ...

    def get_baseline_facts(self, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Pulls 'Identity' domain entities for core profile context."""
        ...

    def semantic_retrieval(
        self, tenant_id: str, user_id: str, query_vector: List[float], limit: int
    ) -> List[Dict[str, Any]]:
        """Performs RAG via vector similarity + synaptic time decay."""
        ...

    def get_recent_entities(self, tenant_id: str, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """Temporal Context: Fetches the last N updated entities."""
        ...

    def get_important_entities(self, tenant_id: str, user_id: str, limit: int = 5) -> List[Dict]:
        """Fetches the most important entities for the user."""
        ...
        
    # --- Workflow 2: Dictionary Builder ---
    def get_ontology_dictionaries(self, tenant_id: str, query_vector: List[float]) -> Dict[str, List[str]]:
        """Extracts existing Taxonomy and Lexicon for the tenant."""
        ...

    # --- Workflow 3: Consolidation (Write Ops) ---
    def get_user_graph_skeleton(self, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Retrieves entity_ids and aliases for anchoring new facts."""
        ...

    def find_exact_fact(
        self, tenant_id: str, user_id: str, entity_id: str, attribute_name: str
    ) -> Optional[UUID]:
        """Checks for existing facts to decide between Reinforce or Mutate."""
        ...

    def commit_graph_updates(self, updates: List[Any]) -> bool:
        """Executes a batch of graph mutations in an atomic transaction."""
        ...