from typing import List, Dict, Any, Optional
from uuid import UUID
import psycopg
import json
from contextlib import contextmanager
from functools import wraps
from tomte.engine.exceptions import TomteSessionStateUnavailable
from tomte.engine.utils.keys import validate_id

from tomte.engine.core.config import settings
from tomte.engine.core.assertions import adapt_assertions
from tomte.engine.database.db_interface import DatabaseProvider
from tomte.engine.core.enums import MemoryDomain
from tomte.engine.exceptions import TomteDatabaseConnectionError
from tomte.engine import tomtelog

# Import separated logic
from tomte.engine.database.postgre import dictionary_ops, schema
from tomte.engine.database.postgre import operations, transactions

def dictionary_writer(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self.transaction():
            self.dictionary_write_guard()
            return method(self, *args, **kwargs)
    return guarded


class PostgresProvider(DatabaseProvider):
    def __init__(self, connection_string: str):
        self._connection_string = connection_string
        try:
            self.conn = transactions.connect(connection_string)
            self.conn.execute("SET statement_timeout = '15s'")
            self.conn.execute("SET lock_timeout = '2s'")
            tomtelog.info("PostgreSQL Provider initialized.")
        except Exception as e:
            tomtelog.error("Database connection failed")
            raise TomteDatabaseConnectionError("PG Connection Failed") from e

    def transaction(self):
        """Join nested scopes; any failed child rolls back the whole unit of work."""
        return transactions.transaction(self.conn)

    def fork(self):
        """A new owned connection; never copy an existing connection object."""
        return type(self)(self._connection_string)

    @contextmanager
    def user_guard(self, tenant_id, user_id):
        key = json.dumps(['tomte:user-execution:v1', tenant_id, user_id])
        acquired = self.conn.execute('SELECT pg_try_advisory_lock(hashtextextended(%s, 0)) AS locked', (key,)).fetchone()['locked']
        try:
            yield acquired
        finally:
            if acquired and not self.conn.closed:
                self.conn.execute('SELECT pg_advisory_unlock(hashtextextended(%s, 0))', (key,))

    def dictionary_write_guard(self):
        self.conn.execute("SELECT pg_advisory_xact_lock(hashtextextended('tomte:dictionary-write:v1', 0))")

    @contextmanager
    def maintenance_guard(self):
        acquired = self.conn.execute("SELECT pg_try_advisory_lock(hashtextextended('tomte:dictionary-write:v1', 0)) AS locked").fetchone()['locked']
        try:
            yield acquired
        finally:
            if acquired and not self.conn.closed:
                self.conn.execute("SELECT pg_advisory_unlock(hashtextextended('tomte:dictionary-write:v1', 0))")

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

    def get_raw_semantic_context(self, tenant_id: str, user_id: str, query_vector: List[float], limit: int | None = None) -> List[Dict[str, Any]]:
        limit = settings.semantic_retrieval_limit if limit is None else limit
        if type(limit) is not int or limit < 1:
            raise ValueError("Raw semantic limit must be a positive integer")
        if not validate_id(tenant_id) or not validate_id(user_id):
            raise ValueError("Invalid raw semantic namespace")
        try:
            with self.transaction():
                return self._normalize_memory_rows(operations.get_raw_semantic_context(self.conn, tenant_id, user_id, query_vector, limit))
        except (psycopg.OperationalError, psycopg.InterfaceError):
            raise TomteSessionStateUnavailable("Raw semantic context unavailable") from None

    def get_recent_entities(self, tenant_id: str, user_id: str, limit: int|None = None) -> List[Dict[str, Any]]:
        with self.transaction():
            return self._normalize_memory_rows(operations.get_recent_entities(self.conn, tenant_id, user_id, limit))

    def _session_retrieval(self, operation, tenant_id, user_id, *args):
        if not validate_id(tenant_id) or not validate_id(user_id):
            raise ValueError("Invalid session entity namespace")
        try:
            with self.transaction():
                return self._normalize_memory_rows(operation(self.conn, tenant_id, user_id, *args))
        except (psycopg.OperationalError, psycopg.InterfaceError):
            # No private query parameters or connection details in the fallback log.
            raise TomteSessionStateUnavailable("Session entity retrieval unavailable") from None

    def discover_session_entities(self, tenant_id, user_id, query_vector, threshold, limit):
        if limit < 1 or not -1 <= threshold <= 1:
            raise ValueError("Invalid session discovery bounds")
        return self._session_retrieval(operations.discover_session_entities, tenant_id, user_id,
                                       query_vector, threshold, limit)

    def find_session_entity_mentions(self, tenant_id, user_id, message, query_vector, limit):
        if limit < 1:
            raise ValueError("Session discovery limit must be positive")
        return self._session_retrieval(operations.find_session_entity_mentions, tenant_id, user_id,
                                       message, query_vector, limit)

    def get_entities_by_ids(self, tenant_id, user_id, entity_ids):
        if not entity_ids:
            return []
        return self._session_retrieval(operations.get_entities_by_ids, tenant_id, user_id,
                                       list(dict.fromkeys(entity_ids)))

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
            tomtelog.error("Bulk commit failed")
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

    @dictionary_writer
    def update_dictionary_anchor_vector(self, dictionary_id: UUID, new_vector: List[float]) -> None:
        dictionary_ops.update_anchor_vector(self.conn, dictionary_id, new_vector)

    @dictionary_writer
    def upsert_dictionary_keyword(self, dictionary_id: UUID, keyword: str, keyword_type: str) -> None:
        dictionary_ops.upsert_keyword(self.conn, dictionary_id, keyword, keyword_type)

    @dictionary_writer
    def create_dictionary_anchor(self, vector: List[float]) -> UUID:
        return dictionary_ops.create_new_anchor(self.conn, vector)



    def get_anchor_keywords(self, dictionary_id: str) -> List[Dict[str, Any]]:
        return dictionary_ops.get_anchor_keywords(self.conn, dictionary_id)

    # ==========================================
    # DICTIONARY MAINTENANCE
    # ==========================================
    @dictionary_writer
    def execute_dictionary_trim(self) -> int:
        return dictionary_ops.execute_trim_job(self.conn)

    def get_cross_anchor_merge_candidates(self) -> List[Dict[str, Any]]:
        return dictionary_ops.get_cross_anchor_merge_candidates(self.conn)

    @dictionary_writer
    def consolidate_duplicate_anchors(self, id_a: str, id_b: str, merged_vector: List[float], merged_kws: Dict) -> bool:
        try:
            dictionary_ops.consolidate_duplicate_anchors(self.conn, id_a, id_b, merged_vector, merged_kws)
            return True
        except Exception as e:
            if transactions.in_transaction(self.conn):
                raise
            tomtelog.error("Cross-anchor merge DB commit failed")
            return False

    def get_anchors_for_internal_merge(self) -> List[str]:
        return dictionary_ops.get_anchors_for_internal_merge(self.conn)

    @dictionary_writer
    def apply_internal_merges(self, anchor_id: str, merges: List[Dict[str, Any]]) -> None:
        dictionary_ops.apply_internal_merges(self.conn, anchor_id, merges)

    @dictionary_writer
    def mark_anchor_internal_merge_complete(self, anchor_id: str) -> None:
        dictionary_ops.mark_anchor_internal_merge_complete(self.conn, anchor_id)
