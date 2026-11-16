from typing import Protocol, List, Dict, Any, Optional, ContextManager
from uuid import UUID

from tomte.engine.core.config import settings
from tomte.engine.core.enums import MemoryDomain
from tomte.engine.core.resolution import CANDIDATE_LIMIT

class DatabaseProvider(Protocol):
    """
    The contract for durable entities, embedded facts and their search index.
    Supports multi-tenancy via (tenant_id, user_id) composite logic.
    """
    supports_answer_fact_search = False
    supports_catalogue_navigation = False

    def get_catalogue_domains(self, tenant_id, user_id, limit):
        """Bounded scoped domain rows and is_partial; excludes unresolved memories."""
        raise NotImplementedError("Catalogue navigation is not supported")

    def get_catalogue_entities(self, tenant_id, user_id, domains, limit):
        """Scoped name/type/alias rows; domains=None means all, [] means none.

        Always exclude unresolved memories. Return is_partial when bounded.
        """
        raise NotImplementedError("Catalogue navigation is not supported")

    def get_catalogue_facts(self, tenant_id, user_id, entity_ids, limit):
        """Scoped fact UUIDs, attributes and current/history field names; no values."""
        raise NotImplementedError("Catalogue navigation is not supported")

    def get_answer_catalog(self, tenant_id, user_id, limit):
        """Scoped actual field paths and counts, including history; no fact values."""
        raise NotImplementedError("Answer field catalogue is not supported")

    def get_answer_facts(self, tenant_id, user_id, vector, paths, limit, max_packet_bytes):
        """Scoped fact snapshots plus entity identity, ranked within matching entities.

        Concrete entity_id/fact_id paths select those exact facts and histories.
        Legacy domain/type/attribute/field paths still rank matching entities.
        Empty paths means a separate broad vector search. Return total/oversized
        counts and bounded rows; do not silently truncate individual values.
        """
        raise NotImplementedError("Answer fact search is not supported")

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
    # Entity retrieval methods return flat entity dictionaries with
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

    def get_unresolved_memories(self, tenant_id: str, user_id: str, vector: List[float],
                                limit: int = 3, recent_limit: int = 0) -> List[Dict[str, Any]]:
        """Scoped pending storage rows: bounded semantic results plus optional recent rows.

        Ordinary entity reads must exclude unresolved_memories (including resolved rows).
        """
        raise NotImplementedError("Unresolved memory lookup is not supported")

    def get_raw_semantic_context(
        self, tenant_id: str, user_id: str, query_vector: List[float], limit: int | None = None
    ) -> List[Dict[str, Any]]:
        """Union of summary top-limit and plain-field top-limit DISTINCT entities.

        At most 2*limit flat scoped entities, sorted by best cosine, with stable
        UUID ties. No age/frequency weighting or generative relevance selection.
        """
        raise NotImplementedError("Raw semantic context is not supported")

    def replace_retrieval_index(self, tenant_id, user_id, entity_id, expected_payload, entries, *, summary_embedding=None):
        """Replace plain vectors and optionally the summary if the scoped payload is unchanged."""
        raise NotImplementedError("Retrieval index rebuild is not supported")

    def get_ranked_memory_context(self, tenant_id, user_id, query, vector, limit, *, mode):
        """Opt-in BM25/hybrid stored entities; scoped, distinct, with vector-compatible hybrid budget."""
        raise NotImplementedError("BM25 retrieval is not supported")

    def get_ranked_history_candidates(self, tenant_id, user_id, entity_ids, query, vector, limit, *, mode):
        """Mode-aware history handles per entity, using the full user's BM25 corpus statistics."""
        raise NotImplementedError("BM25 history retrieval is not supported")

    def rebuild_lexical_index(self, tenant_id, user_id):
        """Atomically rebuild one user's derived lexical data; no memory/vector/model changes."""
        raise NotImplementedError("Lexical index rebuild is not supported")

    def discover_session_entities(self, tenant_id: str, user_id: str,
                                  query_vector: List[float], threshold: float,
                                  limit: int) -> List[Dict[str, Any]]:
        """Non-persona flat entities ranked by raw cosine, stable entity-ID tie-breaker."""
        raise NotImplementedError("Session discovery is not supported")

    def find_session_entity_mentions(self, tenant_id: str, user_id: str,
                                     message: str, query_vector: List[float], limit: int) -> List[Dict[str, Any]]:
        """Exact normalized non-persona name/alias phrases; scoped and bounded."""
        raise NotImplementedError("Session discovery is not supported")

    def get_entities_by_ids(self, tenant_id: str, user_id: str,
                            entity_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch current flat entities in requested order; skip missing scoped IDs."""
        raise NotImplementedError("Session fetch is not supported")

    def get_recent_entities(self, tenant_id: str, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """Temporal Context: Fetches the last N updated entities."""
        ...
    def get_frequent_entities(self, tenant_id: str, user_id: str, limit: int) -> List[Dict[str, Any]]:
        """Fetch ordinary entities by summed fact mention counts, including identity."""
        ...
    def get_profile_entities(self, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
        """All scoped identity_and_demographics entities, with flat storage payloads."""
        raise NotImplementedError("Profile entity retrieval is not supported")
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
        """Claim once with memory/search writes; optional dictionary feedback is separate."""
        ...

    def get_user_graph_skeleton(self, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
        """Full scoped storage records with canonical UUIDs and internal metadata; not a public projection."""
        ...

    def get_resolution_candidates(self, tenant_id: str, user_id: str, names: List[str],
                                  domain: str, vector: List[float], context: str,
                                  limit: int = CANDIDATE_LIMIT) -> List[Dict[str, Any]]:
        """Scoped, bounded storage records for identity resolution; no automatic matches.

        For a known incoming domain, separate thoughts_and_plans from all other
        domains before applying the limit. Empty domain is untyped contextual
        lookup and must allow both. Other domains may match across categories.
        Rank exact incoming names/aliases, then cosine similarity; ordinary
        domain equality is only a tie-break. Do not promote names found elsewhere
        in context (retained as an interface argument for compatibility).
        identity_matches_total counts exact name/alias matches before limiting;
        a small count does not establish semantic candidate completeness.
        """
        ...

    def find_exact_fact(
        self, tenant_id: str, user_id: str, entity_id: str, attribute_name: str
    ) -> Optional[UUID]:
        """Legacy existence lookup returning the entity row UUID, not a selected fact UUID."""
        ...

    def commit_graph_updates(self, updates: List[Any]) -> bool:
        """Atomically write scoped payloads and index deltas, checking expected payloads and immutable history."""
        ...

    # --- Dictionary semantic anchors (bootstrap + feedback) ---
    def get_history_candidates(self, tenant_id: str, user_id: str, entity_ids: List[str], vector: List[float], limit: int) -> List[Dict[str, Any]]:
        """Scoped positional history candidates ranked per entity; never returned as memory."""
        raise NotImplementedError("History candidate lookup is not supported")

    def get_semantic_keywords(
        self, vector: List[float], threshold: float, id_limit: int,
        keyword_type: Optional[str] = None, keyword_limit: int = 5, keyword_context: str = "",
    ) -> List[Dict[str, Any]]:
        """Optional typed top keywords, filtered before limiting; otherwise legacy anchors."""
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

    def upsert_dictionary_keyword(self, dictionary_id: UUID, keyword: str, keyword_type: str, keyword_context: str = "") -> None:
        ...

    def create_dictionary_anchor(self, vector: List[float]) -> UUID:
        ...



    def get_anchor_keywords(self, dictionary_id: str) -> List[Dict[str, Any]]:
        """Keyword rows including keyword_type and keyword_context for one anchor."""
        ...

    def execute_dictionary_trim(self) -> int:
        ...
    def get_cross_anchor_merge_candidates(self) -> List[Dict[str, Any]]:
        ...
    def consolidate_duplicate_anchors(self, id_a: str, id_b: str, merged_vector: List[float], merged_kws: Dict) -> bool:
        """Merge only matching role/context groups; keyword map keys include both."""
        ...
    def get_anchors_for_internal_merge(self) -> List[str]:
        ...
    def apply_internal_merges(self, anchor_id: str, merges: List[Dict[str, Any]]) -> None:
        """Apply approved survivors/victims only within each supplied role and parent context."""
        ...
    def mark_anchor_internal_merge_complete(self, anchor_id: str) -> None:
        ...

    def close(self) -> None:
        """Release the provider connection after all transaction scopes finish."""
        ...
