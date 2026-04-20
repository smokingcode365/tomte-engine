import psycopg
from psycopg.rows import dict_row
from typing import List, Dict, Any, Optional
from uuid import UUID
from pgvector.psycopg import register_vector

from tomte.engine.core.config import settings
from tomte.engine.database.db_interface import DatabaseProvider
from tomte.engine.core.enums import MemoryDomain
from tomte.engine.exceptions import DatabaseConnectionError
from tomte.engine.log import log

# Import separated logic
from tomte.engine.database.postgre import schema
from tomte.engine.database.postgre import operations

class PostgresProvider(DatabaseProvider):
    def __init__(self, connection_string: str):
        try:
            self.conn = psycopg.connect(connection_string, row_factory=dict_row)
            self.conn.autocommit = False 
            register_vector(self.conn)
            log.info("PostgreSQL Provider initialized.")
        except Exception as e:
            log.error("Database connection failed", exc=e)
            raise DatabaseConnectionError(f"PG Connection Failed: {e}")

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
            log.error(f"Failed to update activity for {user_id}", exc=e)

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
        return operations.get_user_graph_skeleton(self.conn, tenant_id, user_id)

    def find_exact_fact(self, tenant_id: str, user_id: str, entity_id: str, attribute_name: str) -> Optional[UUID]:
        return operations.find_exact_fact(self.conn, tenant_id, user_id, entity_id, attribute_name)

    def commit_graph_updates(self, updates: List[Any]) -> bool:
        try:
            return operations.bulk_commit_updates(self.conn, updates)
        except Exception as e:
            log.error("Bulk commit failed", exc=e)
            return False