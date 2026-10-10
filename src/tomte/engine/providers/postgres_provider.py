from typing import List, Dict, Any, Optional
from uuid import UUID

from tomte.engine.core.config import settings
from tomte.engine.core.assertions import adapt_assertions
from tomte.engine.database.db_interface import DatabaseProvider
from tomte.engine.core.enums import MemoryDomain
from tomte.engine.exceptions import TomteDatabaseConnectionError
from tomte.engine import tomtelog

# Import separated logic
from tomte.engine.database.postgre import dictionary_ops, schema
from tomte.engine.database.postgre import operations, transactions

class PostgresProvider(DatabaseProvider):
    def __init__(self, connection_string: str):
        try:
            self.conn = transactions.connect(connection_string)
            tomtelog.info("PostgreSQL Provider initialized.")
        except Exception as e:
            tomtelog.error("Database connection failed", exc=e)
            raise TomteDatabaseConnectionError(f"PG Connection Failed: {e}")

    def transaction(self):
        """Join nested scopes; any failed child rolls back the whole unit of work."""
        return transactions.transaction(self.conn)

    def close(self) -> None:
        transactions.close(self.conn)

    def delete_user_memory(self, tenant_id: str, user_id: str) -> None:
        """Delete only this user's memory rows and registry entry atomically."""
        operations.delete_user_memory(self.conn, tenant_id, user_id)

    # --- JIT Provisioning & Health ---
    def ensure_infrastructure(self, vector_dimension: int) -> None:
        schema.ensure_infrastructure(self.conn, vector_dimension)

    def is_ready(self) -> bool:
        return self.conn is not None and not self.conn.closed

    # --- Workflow 1: Chat Hook (Read Ops) ---

    @staticmethod
    def _normalize_memory_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Expose flat entities while preserving payload fields and row metadata."""
        entities = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise ValueError(f"Invalid retrieved memory row at index {index}: expected a dictionary")
            if "payload" in row:
                if not isinstance(row["payload"], dict):
                    raise ValueError(f"Invalid retrieved memory row at index {index}: expected a dictionary payload")
                entity = dict(row["payload"])
                # Explicit row metadata takes precedence over conflicting payload fields.
                entity.update({key: value for key, value in row.items() if key != "payload"})
            else:
                entity = dict(row)
            if not isinstance(entity.get("entity_id"), str) or not entity["entity_id"].strip():
                raise ValueError(f"Invalid retrieved memory row at index {index}: missing a nonempty entity_id")
            if isinstance(entity.get("id"), UUID):
                entity["id"] = str(entity["id"])  # Portable through Redis persona JSON.
            entity["facts"] = adapt_assertions(entity.get("facts", []), str(entity.get("id") or entity["entity_id"]))
            entities.append(entity)
        return entities

    def upsert_user_activity(self, tenant_id: str, user_id: str) -> None:
        operations.upsert_user_activity(self.conn, tenant_id, user_id)

    # def get_baseline_facts(self, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
    #     return operations.get_baseline_facts(self.conn, tenant_id, user_id)

    def get_important_entities(self, tenant_id: str, user_id: str, domain: Optional[MemoryDomain]=None, limit: int|None = None) -> List[Dict[str, Any]]:
        # Passed None to domain to fetch across all types globally, matching your limit parameter
        with self.transaction():
            return self._normalize_memory_rows(operations.get_entities_by_priority(self.conn, tenant_id, user_id, domain, limit))

    def get_frequent_entities(self, tenant_id: str, user_id: str, limit: int|None = None) -> List[Dict[str, Any]]:
        with self.transaction():
            return self._normalize_memory_rows(operations.get_frequent_entities(self.conn, tenant_id, user_id, limit))

    def semantic_retrieval(self, tenant_id: str, user_id: str, query_vector: List[float], limit: int|None = None) -> List[Dict[str, Any]]:
        with self.transaction():
            return self._normalize_memory_rows(operations.get_semantic_context(self.conn, tenant_id, user_id, query_vector, limit))

    def get_recent_entities(self, tenant_id: str, user_id: str, limit: int|None = None) -> List[Dict[str, Any]]:
        with self.transaction():
            return self._normalize_memory_rows(operations.get_recent_entities(self.conn, tenant_id, user_id, limit))

    # --- Workflow 2: Dictionary Builder ---
    def get_ontology_dictionaries(self, tenant_id: str, query_vector: List[float]) -> Dict[str, Dict[str, List[str]]]:
        return operations.get_ontology_dictionaries(self.conn, tenant_id, query_vector)

    # --- Workflow 3: Consolidation (Write Ops) ---
    def is_consolidation_processed(self, tenant_id: str, user_id: str, batch_id: str) -> bool:
        return operations.is_consolidation_processed(self.conn, tenant_id, user_id, batch_id)

    def claim_consolidation_batch(self, tenant_id: str, user_id: str, batch_id: str) -> bool:
        return operations.claim_consolidation_batch(self.conn, tenant_id, user_id, batch_id)

    def get_user_graph_skeleton(self, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
        return operations.get_user_graph_skeleton(self.conn, tenant_id, user_id)

    def get_resolution_candidates(self, tenant_id, user_id, names, domain, vector, context, limit=5):
        return operations.get_resolution_candidates(self.conn, tenant_id, user_id, names, domain, vector, context, limit)

    def find_exact_fact(self, tenant_id: str, user_id: str, entity_id: str, attribute_name: str) -> Optional[UUID]:
        return operations.find_exact_fact(self.conn, tenant_id, user_id, entity_id, attribute_name)

    def commit_graph_updates(self, updates: List[Any]) -> bool:
        try:
            return operations.bulk_commit_updates(self.conn, updates)
        except Exception as e:
            if transactions.in_transaction(self.conn):
                raise
            tomtelog.error("Bulk commit failed", exc=e)
            return False

    # --- Dictionary semantic anchors (bootstrap + feedback) ---
    def get_semantic_keywords(
        self, vector: List[float], threshold: float, id_limit: int,
        keyword_type: Optional[str] = None, keyword_limit: int = 5,
    ) -> List[Dict[str, Any]]:
        return dictionary_ops.get_semantic_keywords(self.conn, vector, threshold, id_limit,
                                                    keyword_type, keyword_limit)

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



    def get_anchor_keywords(self, dictionary_id: str) -> List[Dict[str, Any]]:
        return dictionary_ops.get_anchor_keywords(self.conn, dictionary_id)

    # ==========================================
    # DICTIONARY MAINTENANCE
    # ==========================================
    def execute_dictionary_trim(self) -> int:
        return dictionary_ops.execute_trim_job(self.conn)

    def get_cross_anchor_merge_candidates(self) -> List[Dict[str, Any]]:
        return dictionary_ops.get_cross_anchor_merge_candidates(self.conn)

    def consolidate_duplicate_anchors(self, id_a: str, id_b: str, merged_vector: List[float], merged_kws: Dict) -> bool:
        try:
            dictionary_ops.consolidate_duplicate_anchors(self.conn, id_a, id_b, merged_vector, merged_kws)
            return True
        except Exception as e:
            if transactions.in_transaction(self.conn):
                raise
            tomtelog.error(f"Cross-anchor merge DB commit failed: {e}")
            return False

    def get_anchors_for_internal_merge(self) -> List[str]:
        return dictionary_ops.get_anchors_for_internal_merge(self.conn)

    def apply_internal_merges(self, anchor_id: str, merges: List[Dict[str, Any]]) -> None:
        dictionary_ops.apply_internal_merges(self.conn, anchor_id, merges)

    def mark_anchor_internal_merge_complete(self, anchor_id: str) -> None:
        dictionary_ops.mark_anchor_internal_merge_complete(self.conn, anchor_id)
