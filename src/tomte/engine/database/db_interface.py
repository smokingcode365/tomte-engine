from typing import Protocol, List, Dict, Any, Optional, ContextManager
from uuid import UUID

from tomte.engine.core.config import settings
from tomte.engine.core.enums import MemoryDomain

class DatabaseProvider(Protocol):
    """
    The contract for the Tomte Engine's permanent Knowledge Graph.
    Supports multi-tenancy via (tenant_id, user_id) composite logic.
    """

    def transaction(self) -> ContextManager[None]:
        """Atomic scope; nested scopes join and failed children abort the whole scope."""
        ...

    def delete_user_memory(self, tenant_id: str, user_id: str) -> None:
        """Atomically delete only the specified user's memory and registry rows."""
        ...

    # --- JIT Provisioning & Health ---
    def ensure_infrastructure(self, vector_dimension: int) -> None:
        """Verifies extensions and creates tables if they don't exist."""
        ...

    def is_ready(self) -> bool:
        """Checks if the database is connected and tables are provisioned."""
        ...

    # --- Workflow 1: Chat Hook (Read Ops) ---
    # The four entity retrieval methods return flat entity dictionaries with
    # entity_id, facts, aliases, and any available retrieval metadata (e.g. scores).
    # Storage payload wrappers are normalized by the provider; invalid rows raise.
    def upsert_user_activity(self, tenant_id: str, user_id: str) -> None:
        """Heartbeat: Registers user/tenant combo and updates timestamp."""
        ...

    # def get_baseline_facts(self, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
    #     """Pulls 'Identity' domain entities for core profile context."""
    #     ...

    def semantic_retrieval(
        self, tenant_id: str, user_id: str, query_vector: List[float], limit: int
    ) -> List[Dict[str, Any]]:
        """Performs RAG via vector similarity + synaptic time decay."""
        ...

    def get_recent_entities(self, tenant_id: str, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """Temporal Context: Fetches the last N updated entities."""
        ...
    def get_frequent_entities(self, tenant_id: str, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """Long-term Context: Fetches entities with the highest mention_count (excluding identity)."""
        ...
    def get_important_entities(self, tenant_id: str, user_id: str, domain: Optional[MemoryDomain], limit: int|None ) -> List[Dict]:
        """Fetches the most important entities for the user."""
        ...
        
    # --- Workflow 2: Dictionary Builder ---
    def get_ontology_dictionaries(self, tenant_id: str, query_vector: List[float]) -> Dict[str, Dict[str, List[str]]]:
        """Extracts existing Taxonomy and Lexicon for the tenant."""
        ...

    # --- Workflow 3: Consolidation (Write Ops) ---
    def is_consolidation_processed(self, tenant_id: str, user_id: str, batch_id: str) -> bool:
        """Check a committed batch identity before preparing model work."""
        ...

    def claim_consolidation_batch(self, tenant_id: str, user_id: str, batch_id: str) -> bool:
        """Within transaction(), claim once alongside memory and dictionary writes."""
        ...

    def get_user_graph_skeleton(self, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Retrieves entity_ids and aliases for anchoring new facts."""
        ...

    def get_resolution_candidates(self, tenant_id: str, user_id: str, names: List[str],
                                  domain: str, vector: List[float], context: str,
                                  limit: int = 5) -> List[Dict[str, Any]]:
        """Scoped, bounded storage records for identity resolution; no automatic matches."""
        ...

    def find_exact_fact(
        self, tenant_id: str, user_id: str, entity_id: str, attribute_name: str
    ) -> Optional[UUID]:
        """Checks for existing facts to decide between Reinforce or Mutate."""
        ...

    def commit_graph_updates(self, updates: List[Any]) -> bool:
        """Executes a batch of graph mutations in an atomic transaction."""
        ...

    # --- Dictionary semantic anchors (bootstrap + feedback) ---
    def get_semantic_keywords(
        self, vector: List[float], threshold: float, id_limit: int
    ) -> List[Dict[str, Any]]:
        """Top semantic anchors and their registry keywords for dictionary bootstrap."""
        ...

    def find_dictionary_matches_above_threshold(
        self, vector: List[float], threshold: float
    ) -> List[Dict[str, Any]]:
        """Anchors whose embedding similarity exceeds ``threshold``."""
        ...

    def get_dictionary_anchor_details(self, dictionary_id: UUID) -> Optional[Dict[str, Any]]:
        """Embedding and aggregate hit stats for nudging math."""
        ...

    def update_dictionary_anchor_vector(self, dictionary_id: UUID, new_vector: List[float]) -> None:
        ...

    def upsert_dictionary_keyword(self, dictionary_id: UUID, keyword: str, keyword_type: str) -> None:
        ...

    def create_dictionary_anchor(self, vector: List[float]) -> UUID:
        ...



    def get_anchor_keywords(self, dictionary_id: str) -> List[Dict[str, Any]]:
        """Keyword rows for one semantic anchor (maintenance / merge UIs)."""
        ...

    def execute_dictionary_trim(self) -> int:
        ...
    def get_cross_anchor_merge_candidates(self) -> List[Dict[str, Any]]:
        ...
    def consolidate_duplicate_anchors(self, id_a: str, id_b: str, merged_vector: List[float], merged_kws: Dict) -> bool:
        ...
    def get_anchors_for_internal_merge(self) -> List[str]:
        ...
    def apply_internal_merges(self, anchor_id: str, merges: List[Dict[str, Any]]) -> None:
        ...
    def mark_anchor_internal_merge_complete(self, anchor_id: str) -> None:
        ...

    def close(self) -> None:
        """Release the provider connection after all transaction scopes finish."""
        ...
