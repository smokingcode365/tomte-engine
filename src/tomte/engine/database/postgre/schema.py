import psycopg
from tomte.engine.log import log

def ensure_infrastructure(conn: psycopg.Connection, vector_dimension: int):
    """Handles JIT table and index creation."""
    safe_dim = int(vector_dimension)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # User Registry Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_registry (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_active_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (tenant_id, user_id)
                );
            """)

            # Core Memory Table
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
            
            # Indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_importance ON memory_engine (importance DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_tenant_user ON memory_engine (tenant_id, user_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_vector ON memory_engine USING ivfflat (embedding vector_cosine_ops);")
            
            conn.commit()
    except Exception as e:
        conn.rollback()
        log.error("Failed to ensure PG infrastructure", exc=e)
        raise