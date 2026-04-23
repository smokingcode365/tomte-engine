import psycopg
from psycopg.rows import dict_row
from typing import List, Dict, Any, Optional
from uuid import UUID
from pgvector.psycopg import register_vector

from tomte.engine.core.config import settings
from tomte.engine.database.db_interface import DatabaseProvider
from tomte.engine.core.enums import MemoryDomain
from tomte.engine.exceptions import TomteDatabaseConnectionError
from tomte.engine import tomtelog

# Import separated logic
from tomte.engine.database.postgre import dictionary_ops, schema
from tomte.engine.database.postgre import operations

class PostgresProvider(DatabaseProvider):
    def __init__(self, connection_string: str):
        try:
            self.conn = psycopg.connect(connection_string, row_factory=dict_row)
            self.conn.autocommit = False 
            register_vector(self.conn)
            tomtelog.info("PostgreSQL Provider initialized.")
        except Exception as e:
            tomtelog.error("Database connection failed", exc=e)
            raise TomteDatabaseConnectionError(f"PG Connection Failed: {e}")

    # --- JIT Provisioning & Health ---
    def ensure_infrastructure(self, vector_dimension: int) -> None:
        schema.ensure_infrastructure(self.conn, vector_dimension)

    def is_ready(self) -> bool:
        return self.conn is not None and not self.conn.closed

    # --- Workflow 1: Chat Hook (Read Ops) ---
    def upsert_user_activity(self, tenant_id: str, user_id: str) -> None:
        try:
            operations.upsert_user_activity(self.conn, tenant_id, user_id)
        except Exception as e:
            tomtelog.error(f"Failed to update activity for {user_id}", exc=e)

    # def get_baseline_facts(self, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
    #     return operations.get_baseline_facts(self.conn, tenant_id, user_id)

    def get_important_entities(self, tenant_id: str, user_id: str, domain: Optional[MemoryDomain]=None, limit: int|None = None) -> List[Dict[str, Any]]:
        # Passed None to domain to fetch across all types globally, matching your limit parameter
        return operations.get_entities_by_priority(self.conn, tenant_id, user_id, domain, limit)

    def get_frequent_entities(self, tenant_id: str, user_id: str, limit: int|None = None) -> List[Dict[str, Any]]:
        return operations.get_frequent_entities(self.conn, tenant_id, user_id, limit)

    def semantic_retrieval(self, tenant_id: str, user_id: str, query_vector: List[float], limit: int|None = None) -> List[Dict[str, Any]]:
        return operations.get_semantic_context(self.conn, tenant_id, user_id, query_vector, limit)

    def get_recent_entities(self, tenant_id: str, user_id: str, limit: int|None = None) -> List[Dict[str, Any]]:
        return operations.get_recent_entities(self.conn, tenant_id, user_id, limit)

    # --- Workflow 2: Dictionary Builder ---
    def get_ontology_dictionaries(self, tenant_id: str, query_vector: List[float]) -> Dict[str, Dict[str, List[str]]]:
        return operations.get_ontology_dictionaries(self.conn, tenant_id, query_vector)

    # --- Workflow 3: Consolidation (Write Ops) ---
    def get_user_graph_skeleton(self, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
        try:
            skeleton = operations.get_user_graph_skeleton(self.conn, tenant_id, user_id)
            # CRITICAL FIX: Close the read transaction before the Phase 3 LLM call!
            self.conn.commit() 
            return skeleton
        except Exception as e:
            self.conn.rollback()
            raise e

    def find_exact_fact(self, tenant_id: str, user_id: str, entity_id: str, attribute_name: str) -> Optional[UUID]:
        return operations.find_exact_fact(self.conn, tenant_id, user_id, entity_id, attribute_name)

    def commit_graph_updates(self, updates: List[Any]) -> bool:
        try:
            return operations.bulk_commit_updates(self.conn, updates)
        except Exception as e:
            tomtelog.error("Bulk commit failed", exc=e)
            return False

    # --- Dictionary semantic anchors (bootstrap + feedback) ---
    def get_semantic_keywords(
        self, vector: List[float], threshold: float, id_limit: int
    ) -> List[Dict[str, Any]]:
        try:
            keywords = dictionary_ops.get_semantic_keywords(self.conn, vector, threshold, id_limit)
            # CRITICAL FIX: Close the read transaction before the Phase 2 Extractor LLM call!
            self.conn.commit() 
            return keywords
        except Exception as e:
            self.conn.rollback()
            raise e

    def find_dictionary_matches_above_threshold(
        self, vector: List[float], threshold: float
    ) -> List[Dict[str, Any]]:
        return dictionary_ops.find_matches_above_threshold(self.conn, vector, threshold)

    def get_dictionary_anchor_details(self, dictionary_id: UUID) -> Optional[Dict[str, Any]]:
        return dictionary_ops.get_anchor_details(self.conn, dictionary_id)

    def update_dictionary_anchor_vector(self, dictionary_id: UUID, new_vector: List[float]) -> None:
        dictionary_ops.update_anchor_vector(self.conn, dictionary_id, new_vector)

    def upsert_dictionary_keyword(self, dictionary_id: UUID, keyword: str, keyword_type: str) -> None:
        dictionary_ops.upsert_keyword(self.conn, dictionary_id, keyword, keyword_type)

    def create_dictionary_anchor(self, vector: List[float]) -> UUID:
        return dictionary_ops.create_new_anchor(self.conn, vector)

    def commit_transaction(self) -> None:
        self.conn.commit()

    def rollback_transaction(self) -> None:
        self.conn.rollback()

    def get_anchor_keywords(self, dictionary_id: str) -> List[Dict[str, Any]]:
        try:
            kws = dictionary_ops.get_anchor_keywords(self.conn, dictionary_id)
            self.conn.commit() # CRITICAL: Close the read transaction!
            return kws
        except Exception as e:
            self.conn.rollback()
            raise e

    # ==========================================
    # DICTIONARY MAINTENANCE
    # ==========================================
    def execute_dictionary_trim(self) -> int:
        try:
            n = dictionary_ops.execute_trim_job(self.conn)
            self.conn.commit()
            return n
        except Exception as e:
            self.conn.rollback()
            tomtelog.error(f"Dictionary trim job failed: {e}")
            raise

    def get_cross_anchor_merge_candidates(self) -> List[Dict[str, Any]]:
        try:
            candidates = dictionary_ops.get_cross_anchor_merge_candidates(self.conn)
            self.conn.commit() # CRITICAL: Close the read transaction!
            return candidates
        except Exception as e:
            self.conn.rollback()
            raise e

    def consolidate_duplicate_anchors(self, id_a: str, id_b: str, merged_vector: List[float], merged_kws: Dict) -> bool:
        try:
            dictionary_ops.consolidate_duplicate_anchors(self.conn, id_a, id_b, merged_vector, merged_kws)
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            tomtelog.error(f"Cross-anchor merge DB commit failed: {e}")
            return False

    def get_anchors_for_internal_merge(self) -> List[str]:
        """Fetches pending anchors. Commits immediately to close the read transaction."""
        try:
            anchors = dictionary_ops.get_anchors_for_internal_merge(self.conn)
            self.conn.commit()
            return anchors
        except Exception as e:
            self.conn.rollback()
            raise e

    def apply_internal_merges(self, anchor_id: str, merges: List[Dict[str, Any]]) -> None:
        """Applies all LLM-approved merges for a single anchor in one quick atomic commit."""
        try:
            for merge in merges:
                dictionary_ops.execute_internal_keyword_merge(
                    self.conn,
                    anchor_id,
                    merge["kw_type"],
                    merge["survivor"],
                    merge["victims"],
                    merge["total_hits"],
                )
            # Mark it complete in the same transaction
            dictionary_ops.mark_anchor_internal_merge_complete(self.conn, anchor_id)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            tomtelog.error(f"Failed to apply internal merges for {anchor_id}: {e}")

    def mark_anchor_internal_merge_complete(self, anchor_id: str) -> None:
        """Marks an anchor complete if no merges were needed."""
        try:
            dictionary_ops.mark_anchor_internal_merge_complete(self.conn, anchor_id)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()