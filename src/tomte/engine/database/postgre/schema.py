# tomte/engine/database/postgre/schema.py
import psycopg
from tomte.engine.log import log

def ensure_infrastructure(conn: psycopg.Connection, vector_dimension: int):
    """
    Master entry point to ensure all Tomte Engine database structures exist.
    """
    ensure_core_infrastructure(conn, vector_dimension)
    ensure_dictionary_infrastructure(conn, vector_dimension)
    log.info("Total PostgreSQL infrastructure verified.")

def ensure_core_infrastructure(conn: psycopg.Connection, vector_dimension: int):
    """Handles the user registry and the core Bounded Hybrid memory engine."""
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
            
            # Performance Indexes
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_importance ON memory_engine (importance DESC);")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_mem_composite_lookup 
                ON memory_engine (tenant_id, user_id, entity_domain);
            """)
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_entity_id ON memory_engine (tenant_id, user_id, entity_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_entity_type ON memory_engine (tenant_id, user_id, entity_type);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_entity_name ON memory_engine (tenant_id, user_id, entity_name);")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_mem_vector 
                ON memory_engine USING ivfflat (embedding vector_cosine_ops);
            """)
            
            conn.commit()
            log.info("Core Memory infrastructure ensured.")
            
    except Exception as e:
        conn.rollback()
        log.error("Failed to ensure Core PG infrastructure", exc=e)
        raise

def ensure_dictionary_infrastructure(conn: psycopg.Connection, vector_dimension: int):
    """
    Handles the 1:Many Semantic Dictionary Registry.
    Separates conceptual anchors (Vector) from vocabulary keywords (SQL).
    """
    safe_dim = int(vector_dimension)
    try:
        with conn.cursor() as cur:
            # 1. Semantic Anchors (The Vector Anchor Points)
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS semantic_anchors (
                    dictionary_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    embedding vector({safe_dim}) NOT NULL
                );
            """)

            # 2. Keyword Registry (The vocabulary linked to anchors)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS keyword_registry (
                    id BIGSERIAL PRIMARY KEY,
                    dictionary_id UUID NOT NULL REFERENCES semantic_anchors(dictionary_id) ON DELETE CASCADE,
                    keyword TEXT NOT NULL,
                    keyword_type TEXT NOT NULL, -- 'ENTITY' or 'ATTRIBUTE'
                    hits INT DEFAULT 1,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    last_modified_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(dictionary_id, keyword, keyword_type)
                );
            """)

            # Performance Indexes for the Dictionary
            cur.execute("CREATE INDEX IF NOT EXISTS idx_dict_hits ON keyword_registry (hits DESC);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_dict_keyword ON keyword_registry (keyword);")
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_dict_vector_lookup 
                ON semantic_anchors USING ivfflat (embedding vector_cosine_ops);
            """)
            
            conn.commit()
            log.info("Semantic Dictionary infrastructure ensured.")
            
    except Exception as e:
        conn.rollback()
        log.error("Failed to ensure Dictionary PG infrastructure", exc=e)
        raise