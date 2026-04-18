import psycopg
import json
from psycopg.rows import dict_row
from typing import List, Dict, Any, Optional
from uuid import UUID
from datetime import datetime

from tomte.engine.core.enums import MemoryDomain
from tomte.engine.database.interfaces import DatabaseProvider
from tomte.engine.exceptions import DatabaseConnectionError

class PostgresProvider(DatabaseProvider):
    def __init__(self, connection_string: str):
        try:
            self.conn_str = connection_string
            self.conn = psycopg.connect(self.conn_str, row_factory=dict_row)
            self.conn.autocommit = False 
        except Exception as e:
            raise DatabaseConnectionError(f"Failed to connect to PostgreSQL: {str(e)}")

    # --- JIT Provisioning ---

    def ensure_infrastructure(self, vector_dimension: int) -> None:
        safe_dim = int(vector_dimension) 
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_registry (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_active_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (tenant_id, user_id)
                );
            """)

            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS memory_engine (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    embedding vector({safe_dim}),
                    importance INT DEFAULT 3,
                    mention_count INT DEFAULT 1,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_mentioned_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_importance ON memory_engine (importance DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_tenant_user ON memory_engine (tenant_id, user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_entity ON memory_engine (entity_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_vector ON memory_engine USING ivfflat (embedding vector_cosine_ops);")
            
            self.conn.commit()

    # --- Workflow 1: Chat Hook (The Trifecta Pull) ---

    def get_important_entities(
            self, 
            tenant_id: str, 
            user_id: str, 
            domain: Optional[MemoryDomain] = None, 
            limit: int = 5
        ) -> List[Dict]:
            """
            Target: The 'Permanent' anchors. 
            Defaults to 'identity_and_demographics' so the AI always remembers 
            names and addresses first.
            """
            # Logic: If 'all', we set filter to None to fetch across the whole DB
            filter_type = domain.value if domain else None
            
            sql = """
                SELECT payload FROM memory_engine 
                WHERE tenant_id = %s AND user_id = %s 
                AND (%s::text IS NULL OR entity_type = %s)
                ORDER BY importance DESC, mention_count DESC 
                LIMIT %s;
            """
            try:
                with self.conn.cursor() as cur:
                    cur.execute(sql, (tenant_id, user_id, filter_type, filter_type, limit))
                    return cur.fetchall()
            except psycopg.errors.UndefinedTable:
                return []

    def get_recent_entities(
        self, 
        tenant_id: str, 
        user_id: str, 
        domain: Optional[MemoryDomain] = None, 
        limit: int = 3
    ) -> List[Dict[str, Any]]:
        """Target: What just happened in the chat."""
        filter_type = domain.value if domain else None

        sql = """
            SELECT payload FROM memory_engine 
            WHERE tenant_id = %s AND user_id = %s 
            AND (%s::text IS NULL OR entity_type = %s)
            ORDER BY last_mentioned_at DESC LIMIT %s;
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (tenant_id, user_id, filter_type, filter_type, limit))
                return cur.fetchall()
        except psycopg.errors.UndefinedTable:
            return []

    def get_frequent_entities(self, tenant_id: str, user_id: str, limit: int = 3) -> List[Dict[str, Any]]:
        """Target: Long-term Obsessions (excluding identity to avoid duplicates)."""
        sql = """
            SELECT payload FROM memory_engine 
            WHERE tenant_id = %s AND user_id = %s AND entity_type != 'identity'
            ORDER BY mention_count DESC LIMIT %s;
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (tenant_id, user_id, limit))
                return cur.fetchall()
        except psycopg.errors.UndefinedTable:
            return []

    def semantic_retrieval(self, tenant_id: str, user_id: str, query_vector: List[float], limit: int) -> List[Dict[str, Any]]:
        sql = """
            SELECT payload, 
            (1 - (embedding <=> %s)) * (mention_count * exp(-0.0001 * extract(epoch from (now() - last_mentioned_at)))) as relevance_score 
            FROM memory_engine 
            WHERE tenant_id = %s AND user_id = %s 
            ORDER BY relevance_score DESC LIMIT %s;
        """
        try:
            with self.conn.cursor() as cur:
                cur.execute(sql, (query_vector, tenant_id, user_id, limit))
                return cur.fetchall()
        except psycopg.errors.UndefinedTable:
            return []

    # --- Workflow 3: Consolidation (Write Ops) ---

    def commit_graph_updates(self, updates: List[Dict[str, Any]]) -> bool:
        """
        FIXED: Now correctly saves the 'importance' value to the DB.
        """
        try:
            with self.conn.cursor() as cur:
                for op in updates:
                    mode = op.get('mode')
                    
                    if mode == 'INSERT':
                        cur.execute("""
                            INSERT INTO memory_engine (tenant_id, user_id, entity_id, entity_type, payload, embedding, importance)
                            VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s);
                        """, (op['tenant_id'], op['user_id'], op['entity_id'], op['entity_type'], 
                              json.dumps(op['payload']), op['embedding'], op.get('importance', 3)))
                    
                    elif mode == 'MUTATE':
                        cur.execute("""
                            UPDATE memory_engine 
                            SET payload = %s::jsonb, embedding = %s, importance = %s, 
                                last_mentioned_at = NOW(), mention_count = mention_count + 1
                            WHERE id = %s AND tenant_id = %s AND user_id = %s;
                        """, (json.dumps(op['payload']), op['embedding'], op.get('importance', 3), 
                              op['db_id'], op['tenant_id'], op['user_id']))

                    elif mode == 'REINFORCE':
                        cur.execute("""
                            UPDATE memory_engine 
                            SET mention_count = mention_count + 1, last_mentioned_at = NOW()
                            WHERE id = %s AND tenant_id = %s AND user_id = %s;
                        """, (op['db_id'], op['tenant_id'], op['user_id']))

                self.conn.commit()
                return True
        except Exception as e:
            self.conn.rollback()
            print(f"Transaction failed: {e}")
            return False

    # ... remaining methods (upsert_user_activity, is_ready, etc.) ...