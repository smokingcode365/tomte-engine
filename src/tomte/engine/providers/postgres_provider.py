from typing import List, Dict, Any, Optional
from uuid import UUID
from pydantic import ValidationError
import psycopg
import json
from contextlib import contextmanager
from functools import wraps
from tomte.engine.exceptions import TomteSessionStateUnavailable
from tomte.engine.utils.keys import validate_id

from tomte.engine.core.config import settings
from tomte.engine.core.memory import Entity
from tomte.engine.database.db_interface import DatabaseProvider
from tomte.engine.core.enums import MemoryDomain
from tomte.engine.core.resolution import CANDIDATE_LIMIT
from tomte.engine.exceptions import TomteDatabaseConnectionError
from tomte.engine import tomtelog

# Import separated logic
from tomte.engine.database.postgre import dictionary_ops, schema
from tomte.engine.database.postgre import operations, transactions, lexical_ops, answer_ops, catalogue_ops
from tomte.engine.core.search import reciprocal_rank_fusion

def dictionary_writer(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        with self.transaction():
            self.dictionary_write_guard()
            return method(self, *args, **kwargs)
    return guarded


class PostgresProvider(DatabaseProvider):
    supports_answer_fact_search = True
    supports_catalogue_navigation = True

    @staticmethod
    def _validate_catalogue_request(tenant_id, user_id, limit, parents=(), *, uuids=False):
        if (not validate_id(tenant_id) or not validate_id(user_id) or type(limit) is not int
                or not 1 <= limit <= 2000 or not isinstance(parents, (list, tuple))
                or len(parents) > 2000):
            raise ValueError('Invalid catalogue scope or limits')
        for value in parents:
            if not isinstance(value, str) or not value or '\x00' in value:
                raise ValueError('Invalid catalogue parent')
            if uuids and str(UUID(value)) != value:
                raise ValueError('Invalid catalogue entity UUID')

    def get_catalogue_domains(self, tenant_id, user_id, limit):
        self._validate_catalogue_request(tenant_id, user_id, limit)
        return catalogue_ops.get_domains(self.conn, tenant_id, user_id, limit)

    def get_catalogue_entities(self, tenant_id, user_id, domains, limit):
        self._validate_catalogue_request(tenant_id, user_id, limit, () if domains is None else domains)
        return catalogue_ops.get_entities(self.conn, tenant_id, user_id, domains, limit)

    def get_catalogue_facts(self, tenant_id, user_id, entity_ids, limit):
        self._validate_catalogue_request(tenant_id, user_id, limit, entity_ids, uuids=True)
        return catalogue_ops.get_facts(self.conn, tenant_id, user_id, entity_ids, limit)

    def get_answer_catalog(self, tenant_id, user_id, limit):
        if not validate_id(tenant_id) or not validate_id(user_id) or type(limit) is not int or not 1 <= limit <= 2000:
            raise ValueError('Invalid answer catalogue scope or limit')
        return answer_ops.get_catalog(self.conn, tenant_id, user_id, limit)

    def get_answer_facts(self, tenant_id, user_id, vector, paths, limit, max_packet_bytes):
        if (not validate_id(tenant_id) or not validate_id(user_id) or type(limit) is not int
                or not 1 <= limit <= 1000 or type(max_packet_bytes) is not int or max_packet_bytes < 1):
            raise ValueError('Invalid answer fact scope or limits')
        if not isinstance(paths, list) or len(paths) > 2000:
            raise ValueError('Invalid answer field paths')
        for path in paths:
            if (not isinstance(path, dict) or set(path) not in (
                    {'entity_id', 'fact_id'},
                    {'entity_domain', 'entity_type', 'attribute_name', 'field_name'})
                    or any(not isinstance(v, str) or not v or '\x00' in v for v in path.values())):
                raise ValueError('Invalid answer field path')
            if 'entity_id' in path and any(str(UUID(v)) != v for v in path.values()):
                raise ValueError('Invalid answer fact UUID')
        return answer_ops.get_facts(self.conn, tenant_id, user_id, vector, paths, limit, max_packet_bytes)

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
            try:
                stored = row["payload"] if "payload" in row else {
                    field.alias or name: entity[field.alias or name]
                    for name, field in Entity.model_fields.items() if (field.alias or name) in entity}
                payload = Entity.model_validate(stored).model_dump(mode="json")
            except (ValidationError, KeyError):
                raise ValueError(f"Invalid retrieved memory row at index {index}: invalid stored entity") from None
            if entity.get("id") is not None and str(entity["id"]) != payload["entity_id"]:
                raise ValueError("Retrieved entity identity disagrees with database UUID")
            if "payload" in row and row["payload"].get("entity_id") != payload["entity_id"]:
                raise ValueError("Retrieved payload identity disagrees with row")
            entity.update(payload)
            entities.append(entity)
        return entities

    @staticmethod
    def _validate_storage_rows(rows):
        """Raw admin/identity reads still validate the same canonical boundary."""
        normalized = PostgresProvider._normalize_memory_rows(rows)
        for row, entity in zip(rows, normalized):
            if any(key in row and row[key] != entity[key] for key in ("entity_id", "entity_name", "entity_type", "entity_domain")):
                raise ValueError("Stored row identity metadata disagrees with payload")
        return rows

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

    def get_profile_entities(self, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
        if not validate_id(tenant_id) or not validate_id(user_id):
            raise ValueError("Invalid profile namespace")
        with self.transaction():
            return self._normalize_memory_rows(operations.get_profile_entities(self.conn, tenant_id, user_id))

    def semantic_retrieval(self, tenant_id: str, user_id: str, query_vector: List[float], limit: int|None = None) -> List[Dict[str, Any]]:
        with self.transaction():
            return self._normalize_memory_rows(operations.get_semantic_context(self.conn, tenant_id, user_id, query_vector, limit))

    def get_unresolved_memories(self, tenant_id, user_id, vector, limit=3, recent_limit=0):
        if not validate_id(tenant_id) or not validate_id(user_id):
            raise ValueError("Invalid unresolved memory namespace")
        if type(limit) is not int or not 1 <= limit <= 20 or type(recent_limit) is not int or not 0 <= recent_limit <= 10:
            raise ValueError("Invalid unresolved memory limits")
        with self.transaction():
            rows = self._validate_storage_rows(operations.get_unresolved_memories(
                self.conn, tenant_id, user_id, vector, limit, recent_limit))
            if any(row['tenant_id'] != tenant_id or row['user_id'] != user_id for row in rows):
                raise ValueError("Unresolved memory outside requested namespace")
            return rows

    def get_raw_semantic_context(self, tenant_id: str, user_id: str, query_vector: List[float], limit: int | None = None) -> List[Dict[str, Any]]:
        """Up to limit from each independent ranking, deduplicated (at most 2*limit)."""
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

    def replace_retrieval_index(self, tenant_id, user_id, entity_id, expected_payload, entries, *, summary_embedding=None):
        if not validate_id(tenant_id) or not validate_id(user_id):
            raise ValueError("Invalid retrieval-index namespace")
        with self.transaction():
            operations.replace_retrieval_index(self.conn, tenant_id, user_id, entity_id, expected_payload, entries,
                                               summary_embedding=summary_embedding)

    @staticmethod
    def _validate_ranked_search(tenant_id, user_id, query, limit, mode):
        if not validate_id(tenant_id) or not validate_id(user_id):
            raise ValueError("Invalid lexical retrieval namespace")
        if type(limit) is not int or limit < 1 or mode not in ('bm25', 'hybrid'):
            raise ValueError("Invalid lexical retrieval options")
        if not isinstance(query, str) or not query.strip() or '\x00' in query or len(query) > settings.max_input_length:
            raise ValueError("Invalid lexical retrieval query")

    def get_ranked_memory_context(self, tenant_id, user_id, query, vector, limit, *, mode):
        self._validate_ranked_search(tenant_id, user_id, query, limit, mode)
        with self.transaction():
            depth = max(20, 2 * limit)
            lexical = self._normalize_memory_rows(lexical_ops.get_bm25_context(
                self.conn, tenant_id, user_id, query, limit if mode == 'bm25' else depth))
            if mode == 'bm25':
                return lexical
            base = self.get_raw_semantic_context(tenant_id, user_id, vector, limit)
            if not lexical:
                return base
            deeper = self.get_raw_semantic_context(tenant_id, user_id, vector, depth)
            vector_ids = list(dict.fromkeys(e['entity_id'] for e in base + deeper))[:depth]
            ranking, scores = reciprocal_rank_fusion(vector_ids, [e['entity_id'] for e in lexical])
            by_id = {e['entity_id']: e for e in lexical + deeper + base}
            # Retain the actual original union size, not a larger result budget.
            return [{**by_id[eid], 'hybrid_score': scores[eid]} for eid in ranking[:len(base)]]

    def get_ranked_history_candidates(self, tenant_id, user_id, entity_ids, query, vector, limit, *, mode):
        self._validate_ranked_search(tenant_id, user_id, query, limit, mode)
        with self.transaction():
            depth = max(20, limit)
            lexical = lexical_ops.get_bm25_history(self.conn, tenant_id, user_id, entity_ids, query,
                                                  limit if mode == 'bm25' else depth)
            if mode == 'bm25':
                return lexical
            semantic = self.get_history_candidates(tenant_id, user_id, entity_ids, vector, depth)
            result = []
            for entity_id in dict.fromkeys(map(str, entity_ids)):
                groups = [[r for r in rows if str(r['entity_uuid']) == entity_id] for rows in (semantic, lexical)]
                handles = [[(str(r['fact_id']), r['history_index']) for r in rows] for rows in groups]
                ranking, scores = reciprocal_rank_fusion(*handles)
                by_key = {(str(r['fact_id']), r['history_index']): r for rows in groups for r in rows}
                result.extend({**by_key[key], 'hybrid_score': scores[key]} for key in ranking[:limit])
            return result

    def rebuild_lexical_index(self, tenant_id, user_id):
        if not validate_id(tenant_id) or not validate_id(user_id):
            raise ValueError("Invalid lexical index namespace")
        return lexical_ops.rebuild_user_index(self.conn, tenant_id, user_id)

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
        with self.transaction():
            return self._validate_storage_rows(operations.get_user_graph_skeleton(self.conn, tenant_id, user_id))

    def get_resolution_candidates(self, tenant_id, user_id, names, domain, vector, context, limit=CANDIDATE_LIMIT):
        with self.transaction():
            return self._validate_storage_rows(operations.get_resolution_candidates(self.conn, tenant_id, user_id, names, domain, vector, context, limit))

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
    def get_history_candidates(self, tenant_id, user_id, entity_ids, vector, limit):
        return operations.get_history_candidates(self.conn, tenant_id, user_id, entity_ids, vector, limit)

    def get_semantic_keywords(
        self, vector: List[float], threshold: float, id_limit: int,
        keyword_type: Optional[str] = None, keyword_limit: int = 5, keyword_context: str = "",
    ) -> List[Dict[str, Any]]:
        return dictionary_ops.get_semantic_keywords(self.conn, vector, threshold, id_limit,
                                                    keyword_type, keyword_limit, keyword_context)

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
    def upsert_dictionary_keyword(self, dictionary_id: UUID, keyword: str, keyword_type: str, keyword_context: str = "") -> None:
        dictionary_ops.upsert_keyword(self.conn, dictionary_id, keyword, keyword_type, keyword_context)

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
