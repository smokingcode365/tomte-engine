import psycopg
from tomte.engine.log import log

def ensure_infrastructure(conn: psycopg.Connection, vector_dimension: int):
    """Handles JIT table and index creation with performance indexing."""
    safe_dim = int(vector_dimension)
    try:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            # 1. User Registry Table
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_registry (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_active_at TIMESTAMPTZ DEFAULT NOW(),
                    PRIMARY KEY (tenant_id, user_id)
                );
            """)

            # 2. Core Memory Table
            # Added entity_name for human-readability/AI-context
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS memory_engine (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    entity_name TEXT NOT NULL, 
                    entity_domain TEXT NOT NULL,
                    entity_type TEXT NOT NULL,
                    payload JSONB NOT NULL,
                    embedding vector({safe_dim}),
                    importance INT DEFAULT 3,
                    mention_count INT DEFAULT 1,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_mentioned_at TIMESTAMPTZ DEFAULT NOW()
                );
            """)
            
            # --- 3. Performance Indexes ---

            # High-priority sorting index
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_importance ON memory_engine (importance DESC);")
            
            # The "Senior Move": Composite lookup index for multi-tenancy + Domain filtering
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_mem_composite_lookup 
                ON memory_engine (tenant_id, user_id, entity_domain);
            """)

            # Anchor lookups (Critical for the Mutator/Consolidation worker)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_entity_id ON memory_engine (tenant_id, user_id, entity_id);")
            
            # Categorization & Search lookups
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_entity_type ON memory_engine (tenant_id, user_id, entity_type);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_entity_name ON memory_engine (tenant_id, user_id, entity_name);")

            # Vector similarity search (Cosine Ops)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_mem_vector 
                ON memory_engine USING ivfflat (embedding vector_cosine_ops);
            """)
            
            conn.commit()
            log.info("PostgreSQL infrastructure and indexes ensured.")
            
    except Exception as e:
        conn.rollback()
        log.error("Failed to ensure PG infrastructure", exc=e)
        raise