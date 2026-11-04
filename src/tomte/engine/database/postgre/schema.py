# tomte/engine/database/postgre/schema.py
import psycopg
from tomte.engine.database.postgre.transactions import transactional
from tomte.engine import tomtelog

@transactional
def ensure_infrastructure(conn: psycopg.Connection, vector_dimension: int):
    """
    Master entry point to ensure all Tomte Engine database structures exist.
    """
    ensure_core_infrastructure(conn, vector_dimension)
    ensure_dictionary_infrastructure(conn, vector_dimension)
    tomtelog.info("Total PostgreSQL infrastructure verified.")

@transactional
def ensure_core_infrastructure(conn: psycopg.Connection, vector_dimension: int):
    """Handles the user registry and the core Bounded Hybrid memory engine."""
    safe_dim = int(vector_dimension)
    with conn.cursor() as cur:
        # CRITICAL FIX: Ensure both extensions are loaded for maximum compatibility
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")

        # 1. User Registry Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tomte_user_registry (
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                last_active_at TIMESTAMPTZ DEFAULT NOW(),
                PRIMARY KEY (tenant_id, user_id)
            );
        """)

        # Retain processed identities indefinitely; no chat payloads are stored.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tomte_processed_consolidations (
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                batch_id TEXT NOT NULL,
                processed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                PRIMARY KEY (tenant_id, user_id, batch_id)
            );
        """)

        # 2. Core Memory Table
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS tomte_memory_engine (
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

        # Existing duplicate identities require explicit repair, never silent data loss.
        cur.execute("""
            SELECT 1 FROM tomte_memory_engine
            GROUP BY tenant_id, user_id, entity_id HAVING count(*) > 1 LIMIT 1;
        """)
        if cur.fetchone():
            raise ValueError("Duplicate canonical entity IDs require repair before schema bootstrap")
        cur.execute("""
            CREATE UNIQUE INDEX IF NOT EXISTS uq_mem_canonical_identity
            ON tomte_memory_engine (tenant_id, user_id, entity_id);
        """)
        # Existing test memories are reset explicitly during rollout. NOT VALID
        # protects new writes without silently rewriting incompatible old data.
        cur.execute("""
            DO $$ BEGIN
                IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname='ck_mem_uuid_identity'
                    AND conrelid='tomte_memory_engine'::regclass) THEN
                    ALTER TABLE tomte_memory_engine ADD CONSTRAINT ck_mem_uuid_identity
                    CHECK (entity_id=id::text AND COALESCE(payload->>'entity_id','')=id::text) NOT VALID;
                END IF;
            END $$;
        """)
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_mem_owner_uuid ON tomte_memory_engine(id,tenant_id,user_id)")
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS tomte_memory_search (
                entity_uuid UUID NOT NULL,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                fact_id UUID NOT NULL,
                history_index INT NOT NULL CHECK(history_index >= -1),
                field_name TEXT NOT NULL,
                chunk_index INT NOT NULL CHECK(chunk_index >= 0),
                embedding vector({safe_dim}) NOT NULL,
                PRIMARY KEY(entity_uuid,fact_id,history_index,field_name,chunk_index),
                FOREIGN KEY(entity_uuid,tenant_id,user_id)
                    REFERENCES tomte_memory_engine(id,tenant_id,user_id) ON DELETE CASCADE
            );
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_search_owner ON tomte_memory_search(tenant_id,user_id)")

        # Performance Indexes
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_importance ON tomte_memory_engine (importance DESC);")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_mem_composite_lookup
            ON tomte_memory_engine (tenant_id, user_id, entity_domain);
        """)
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_entity_id ON tomte_memory_engine (tenant_id, user_id, entity_id);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_entity_type ON tomte_memory_engine (tenant_id, user_id, entity_type);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_mem_entity_name ON tomte_memory_engine (tenant_id, user_id, entity_name);")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_mem_vector
            ON tomte_memory_engine USING ivfflat (embedding vector_cosine_ops);
        """)

        # 3. Keyword Audit Log Table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tomte_keyword_audit_log (
                id BIGSERIAL PRIMARY KEY,
                dictionary_id UUID,
                keyword TEXT NOT NULL,
                keyword_type TEXT NOT NULL,
                hits INT,
                last_modified_at TIMESTAMPTZ,
                removed_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        tomtelog.info("Core Memory infrastructure ensured.")


@transactional
def ensure_dictionary_infrastructure(conn: psycopg.Connection, vector_dimension: int):
    """
    Handles the 1:Many Semantic Dictionary Registry.
    Separates conceptual anchors (Vector) from vocabulary keywords (SQL).
    """
    safe_dim = int(vector_dimension)
    with conn.cursor() as cur:
        # 1. Semantic Anchors (The Vector Anchor Points)
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS tomte_semantic_anchors (
                dictionary_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                embedding vector({safe_dim}) NOT NULL,
                last_internal_merge_at TIMESTAMPTZ DEFAULT NULL
            );
        """)

        # 2. Keyword Registry (The vocabulary linked to anchors)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tomte_keyword_registry (
                id BIGSERIAL PRIMARY KEY,
                dictionary_id UUID NOT NULL REFERENCES tomte_semantic_anchors(dictionary_id) ON DELETE CASCADE,
                keyword TEXT NOT NULL,
                keyword_type TEXT NOT NULL, -- 'ENTITY' or 'ATTRIBUTE'
                hits INT DEFAULT 1,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                last_modified_at TIMESTAMPTZ DEFAULT NOW(),
                UNIQUE(dictionary_id, keyword, keyword_type)
            );
        """)

        # Performance Indexes for the Dictionary
        # Add context without discarding existing vocabulary. Empty context is
        # retained for ENTITY/ATTRIBUTE; FIELD is scoped to its canonical parent.
        cur.execute("ALTER TABLE tomte_keyword_registry ADD COLUMN IF NOT EXISTS keyword_context TEXT NOT NULL DEFAULT ''")
        cur.execute("ALTER TABLE tomte_keyword_audit_log ADD COLUMN IF NOT EXISTS keyword_context TEXT NOT NULL DEFAULT ''")
        cur.execute("ALTER TABLE tomte_keyword_registry DROP CONSTRAINT IF EXISTS tomte_keyword_registry_dictionary_id_keyword_keyword_type_key")
        cur.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_dict_context ON tomte_keyword_registry(dictionary_id,keyword,keyword_type,keyword_context)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dict_hits ON tomte_keyword_registry (hits DESC);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_dict_keyword ON tomte_keyword_registry (keyword);")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_dict_vector_lookup
            ON tomte_semantic_anchors USING ivfflat (embedding vector_cosine_ops);
        """)

        tomtelog.info("Semantic Dictionary infrastructure ensured.")
