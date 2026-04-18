import json
import psycopg
from typing import List, Dict, Any, Optional
from uuid import UUID
from tomte.engine.core.enums import MemoryDomain

# --- Heartbeat ---
def upsert_user_activity(conn: psycopg.Connection, tenant_id: str, user_id: str) -> None:
    sql = """
        INSERT INTO user_registry (tenant_id, user_id, last_active_at)
        VALUES (%s, %s, NOW())
        ON CONFLICT (tenant_id, user_id) DO UPDATE SET last_active_at = NOW();
    """
    try:
        with conn.cursor() as cur:
            cur.execute(sql, (tenant_id, user_id))
            conn.commit()
    except Exception:
        conn.rollback()
        raise

# --- Workflow 1: Read Ops ---
# def get_baseline_facts(conn: psycopg.Connection, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
#     """Targets 'identity' entities to anchor the profile."""
#     sql = """
#         SELECT payload FROM memory_engine 
#         WHERE tenant_id = %s AND user_id = %s AND entity_type = 'identity'
#         ORDER BY importance DESC LIMIT 10;
#     """
#     with conn.cursor() as cur:
#         cur.execute(sql, (tenant_id, user_id))
#         return cur.fetchall()

def get_entities_by_priority(conn: psycopg.Connection, tenant_id: str, user_id: str, domain: Optional[MemoryDomain], limit: int) -> List[Dict[str, Any]]:
    filter_type = domain.value if domain else None
    sql = """
        SELECT payload FROM memory_engine 
        WHERE tenant_id = %s AND user_id = %s 
        AND (%s::text IS NULL OR entity_type = %s)
        ORDER BY importance DESC, mention_count DESC LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tenant_id, user_id, filter_type, filter_type, limit))
        return cur.fetchall()

def get_frequent_entities(conn: psycopg.Connection, tenant_id: str, user_id: str, limit: int) -> List[Dict[str, Any]]:
    """Target: Long-term Obsessions (excluding identity to avoid duplicates)."""
    sql = """
        SELECT payload FROM memory_engine 
        WHERE tenant_id = %s AND user_id = %s 
        ORDER BY mention_count DESC LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tenant_id, user_id, limit))
        return cur.fetchall()

def get_recent_entities(conn: psycopg.Connection, tenant_id: str, user_id: str, limit: int) -> List[Dict[str, Any]]:
    sql = """
        SELECT payload FROM memory_engine 
        WHERE tenant_id = %s AND user_id = %s 
        ORDER BY last_mentioned_at DESC LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tenant_id, user_id, limit))
        return cur.fetchall()

def get_semantic_context(conn: psycopg.Connection, tenant_id: str, user_id: str, vector: List[float], limit: int) -> List[Dict[str, Any]]:
    # Synaptic Decay Formula for Relevance Score:
    # $$R = (1 - (embedding \iff vector)) \times (mention\_count \times e^{-0.0001 \times \Delta t})$$
    sql = """
        SELECT payload, 
        (1 - (embedding <=> %s)) * (mention_count * exp(-0.0001 * extract(epoch from (now() - last_mentioned_at)))) as relevance_score 
        FROM memory_engine 
        WHERE tenant_id = %s AND user_id = %s 
        ORDER BY relevance_score DESC LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (vector, tenant_id, user_id, limit))
        return cur.fetchall()

# --- Workflow 2: Dictionary Builder ---
def get_ontology_dictionaries(conn: psycopg.Connection, tenant_id: str, query_vector: List[float]) -> Dict[str, List[str]]:
    sql = """
        SELECT DISTINCT entity_type FROM memory_engine WHERE tenant_id = %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tenant_id,))
        types = [row['entity_type'] for row in cur.fetchall()]
        return {"entity_types": types}

# --- Workflow 3: Consolidation Write Ops ---
def get_user_graph_skeleton(conn: psycopg.Connection, tenant_id: str, user_id: str) -> List[Dict[str, Any]]:
    sql = """
        SELECT id, entity_id, entity_type FROM memory_engine 
        WHERE tenant_id = %s AND user_id = %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tenant_id, user_id))
        return cur.fetchall()

def find_exact_fact(conn: psycopg.Connection, tenant_id: str, user_id: str, entity_id: str, attribute_name: str) -> Optional[UUID]:
    # Uses Postgres JSONB containment operator @> to scan the internal facts array natively
    sql = """
        SELECT id FROM memory_engine 
        WHERE tenant_id = %s AND user_id = %s AND entity_id = %s
        AND payload @> jsonb_build_object('facts', jsonb_build_array(jsonb_build_object('attribute_name', %s::text)))
        LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (tenant_id, user_id, entity_id, attribute_name))
        row = cur.fetchone()
        return row['id'] if row else None

def bulk_commit_updates(conn: psycopg.Connection, updates: List[Dict[str, Any]]) -> bool:
    try:
        with conn.cursor() as cur:
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
                        UPDATE memory_engine SET payload = %s::jsonb, embedding = %s, importance = %s, 
                        last_mentioned_at = NOW(), mention_count = mention_count + 1
                        WHERE id = %s AND tenant_id = %s AND user_id = %s;
                    """, (json.dumps(op['payload']), op['embedding'], op.get('importance', 3), 
                          op['db_id'], op['tenant_id'], op['user_id']))

                elif mode == 'REINFORCE':
                    cur.execute("""
                        UPDATE memory_engine SET mention_count = mention_count + 1, last_mentioned_at = NOW()
                        WHERE id = %s AND tenant_id = %s AND user_id = %s;
                    """, (op['db_id'], op['tenant_id'], op['user_id']))
            conn.commit()
            return True
    except Exception:
        conn.rollback()
        raise