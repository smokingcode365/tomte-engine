import json
import psycopg
from tomte.engine.database.postgre.transactions import transactional
from typing import List, Dict, Any, Optional
from uuid import UUID
from tomte.engine.core.config import settings
from tomte.engine.core.enums import MemoryDomain


@transactional
def delete_user_memory(conn: psycopg.Connection, tenant_id: str, user_id: str) -> None:
    """Delete the explicitly selected user's memory and registry as one unit."""
    with conn.cursor() as cur:
        cur.execute("DELETE FROM tomte_memory_engine WHERE tenant_id = %s AND user_id = %s", (tenant_id, user_id))
        cur.execute("DELETE FROM tomte_user_registry WHERE tenant_id = %s AND user_id = %s", (tenant_id, user_id))

# --- Heartbeat ---
@transactional
def upsert_user_activity(conn: psycopg.Connection, tenant_id: str, user_id: str) -> None:
    sql = """
        INSERT INTO tomte_user_registry (tenant_id, user_id, last_active_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (tenant_id, user_id) DO UPDATE SET last_active_at = NOW();
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tenant_id, user_id))


# --- Workflow 1: Read Ops (The Live Pulse) ---

@transactional
def get_entities_by_priority(conn: psycopg.Connection, tenant_id: str, user_id: str, domain: Optional[MemoryDomain], limit: int) -> List[Dict[str, Any]]:
    """Generic priority fetch, filtered by broad Domain enum value if provided."""
    filter_val = domain.value if domain else None
    limit=limit if limit else settings.query_entities_limit
    sql = """
        SELECT payload FROM tomte_memory_engine 
        WHERE tenant_id = %s AND user_id = %s 
        AND (%s::text IS NULL OR entity_domain = %s)
        ORDER BY importance DESC, mention_count DESC LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tenant_id, user_id, filter_val, filter_val, limit))
        return cur.fetchall()

@transactional
def get_frequent_entities(conn: psycopg.Connection, tenant_id: str, user_id: str, limit: int) -> List[Dict[str, Any]]:
    """Pure fetch of most mentioned entities. No domain filtering."""
    limit=limit if limit else settings.query_entities_limit
    sql = """
        SELECT payload FROM tomte_memory_engine 
        WHERE tenant_id = %s AND user_id = %s 
        ORDER BY mention_count DESC LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tenant_id, user_id, limit))
        return cur.fetchall()

@transactional
def get_recent_entities(conn: psycopg.Connection, tenant_id: str, user_id: str, limit: int) -> List[Dict[str, Any]]:
    limit=limit if limit else settings.query_entities_limit
    sql = """
        SELECT payload FROM tomte_memory_engine 
        WHERE tenant_id = %s AND user_id = %s 
        ORDER BY last_mentioned_at DESC LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tenant_id, user_id, limit))
        return cur.fetchall()

@transactional
def get_semantic_context(conn: psycopg.Connection, tenant_id: str, user_id: str, vector: List[float], limit: int) -> List[Dict[str, Any]]:
    limit=limit if limit else settings.semantic_retrieval_limit
    sql = """
        SELECT payload, 
        (1 - (embedding <=> %s::vector)) * (mention_count * exp(-0.0001 * extract(epoch from (now() - last_mentioned_at)))) as relevance_score 
        FROM tomte_memory_engine 
        WHERE tenant_id = %s AND user_id = %s 
        ORDER BY relevance_score DESC LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (vector, tenant_id, user_id, limit))
        return cur.fetchall()

# --- Workflow 2: Dictionary Builder (Semantic Relative) ---

@transactional
def get_ontology_dictionaries(conn: psycopg.Connection, tenant_id: str, chat_vector: List[float]) -> Dict[str, Dict[str, List[str]]]:
    """Builds a relative dictionary from the semantic neighborhood."""
    sql = """
        WITH neighborhood AS (
            SELECT entity_domain, entity_type, payload 
            FROM tomte_memory_engine 
            WHERE tenant_id = %s
            ORDER BY embedding <=> %s::vector 
            LIMIT 20
        )
        SELECT 
            entity_domain, 
            entity_type,
            jsonb_path_query_array(payload, '$.facts[*].attribute_name') as attributes
        FROM neighborhood;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tenant_id, chat_vector))
        rows = cur.fetchall()

    onto = {}
    for row in rows:
        dom, typ = row['entity_domain'], row['entity_type']
        attrs = list(row['attributes']) if row['attributes'] else []
        
        if dom not in onto: onto[dom] = {}
        if typ not in onto[dom]: onto[dom][typ] = set()
        onto[dom][typ].update(attrs)

    return {d: {t: list(a) for t, a in types.items()} for d, types in onto.items()}

# --- Workflow 3: Consolidation Write Ops ---

@transactional
def get_user_graph_skeleton(conn: psycopg.Connection, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
    """Retrieves anchors including human-friendly name for extraction mapping."""
    sql = """
        SELECT id, entity_id, entity_name, entity_domain, entity_type, payload 
        FROM tomte_memory_engine 
        WHERE tenant_id = %s AND user_id = %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tenant_id, user_id))
        return cur.fetchall()

@transactional
def find_exact_fact(conn: psycopg.Connection, tenant_id: str, user_id: str, entity_id: str, attribute_name: str) -> Optional[UUID]:
    sql = """
        SELECT id FROM tomte_memory_engine 
        WHERE tenant_id = %s AND user_id = %s AND entity_id = %s
        AND payload @> jsonb_build_object('facts', jsonb_build_array(jsonb_build_object('attribute_name', %s::text)))
        LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tenant_id, user_id, entity_id, attribute_name))
        row = cur.fetchone()
        return row['id'] if row else None

@transactional
def bulk_commit_updates(conn: psycopg.Connection, updates: List[Dict[str, Any]]) -> bool:
    with conn.cursor() as cur:
        for op in updates:
            mode = op.get('mode')
            if mode == 'INSERT':
                cur.execute("""
                    INSERT INTO tomte_memory_engine (
                        tenant_id, user_id, entity_id, entity_name, entity_domain, entity_type, payload, embedding, importance
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector, %s);
                """, (
                    op['tenant_id'], op['user_id'], op['entity_id'], op['entity_name'],
                    op['entity_domain'], op['entity_type'],
                    json.dumps(op['payload']), op['embedding'], op.get('importance', 3)
                ))

            elif mode == 'MUTATE':
                cur.execute("""
                    UPDATE tomte_memory_engine SET
                        payload = %s::jsonb,
                        embedding = %s::vector,
                        importance = %s,
                        entity_name = %s,
                        entity_domain = %s,
                        entity_type = %s,
                        last_mentioned_at = NOW(),
                        mention_count = mention_count + 1
                    WHERE id = %s AND tenant_id = %s AND user_id = %s;
                """, (
                    json.dumps(op['payload']), op['embedding'], op.get('importance', 3),
                    op['entity_name'], op['entity_domain'], op['entity_type'],
                    op['db_id'], op['tenant_id'], op['user_id']
                ))
                
                # Prevent silent database failures if the ID doesn't actually exist
                if cur.rowcount == 0:
                    raise ValueError(f"MUTATE failed: Record {op.get('db_id')} not found or namespace mismatch.")

            elif mode == 'REINFORCE':
                # CRITICAL FIX: Save the updated payload and embedding alongside the mention_count bump
                cur.execute("""
                    UPDATE tomte_memory_engine
                    SET payload = %s::jsonb,
                        embedding = %s::vector,
                        mention_count = mention_count + 1,
                        last_mentioned_at = NOW()
                    WHERE id = %s AND tenant_id = %s AND user_id = %s;
                """, (
                    json.dumps(op['payload']), op['embedding'],
                    op['db_id'], op['tenant_id'], op['user_id']
                ))

                # Prevent silent database failures if the ID doesn't actually exist
                if cur.rowcount == 0:
                    raise ValueError(f"REINFORCE failed: Record {op.get('db_id')} not found or namespace mismatch.")
                    
        return True
