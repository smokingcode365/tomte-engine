# tomte/engine/database/postgre/dictionary_ops.py

import json
from typing import Any, List, Dict, Optional
from uuid import UUID
import psycopg

def get_anchor_keywords(conn: psycopg.Connection, dictionary_id: UUID | str) -> List[Dict[str, Any]]:
    """Returns all keyword rows for an anchor (used by maintenance and internal merge)."""
    sql = """
        SELECT keyword, keyword_type, hits, created_at, last_modified_at
        FROM tomte_keyword_registry
        WHERE dictionary_id = %s
    """
    with conn.cursor() as cur:
        cur.execute(sql, (dictionary_id,))
        return cur.fetchall()


def get_semantic_keywords(
    conn: psycopg.Connection, 
    vector: List[float], 
    threshold: float, 
    id_limit: int
) -> List[Dict[str, Any]]:
    """Phase 1: Retrieves ALL keywords for the Top K unique semantic IDs."""
    sql = """
        -- CRITICAL FIX: Added a.score to the SELECT statement
        SELECT r.keyword, r.keyword_type, r.hits, a.dictionary_id, a.score
        FROM (
            SELECT dictionary_id, (1 - (embedding <=> %(vec)s::vector)) as score
            FROM tomte_semantic_anchors
            WHERE (1 - (embedding <=> %(vec)s::vector)) > %(thresh)s
            ORDER BY score DESC
            LIMIT %(limit)s
        ) a
        JOIN tomte_keyword_registry r ON a.dictionary_id = r.dictionary_id;
    """
    with conn.cursor() as cur:
        cur.execute(sql, {
            "vec": vector, 
            "thresh": threshold, 
            "limit": id_limit
        })
        return cur.fetchall()

# def find_anchor_match(conn: psycopg.Connection, vector: List[float], threshold: float) -> Optional[UUID]:
#     """
#     Phase 4: Checks if a new entity vector matches an existing semantic anchor.
#     """
#     sql = """
#         SELECT dictionary_id FROM tomte_semantic_anchors
#         WHERE (1 - (embedding <=> %s::vector)) > %s
#         ORDER BY (1 - (embedding <=> %s::vector)) DESC
#         LIMIT 1;
#     """
#     with conn.cursor() as cur:
#         cur.execute(sql, (vector, threshold, vector))
#         row = cur.fetchone()
#         return row['dictionary_id'] if row else None

def update_anchor_vector(conn: psycopg.Connection, dictionary_id: UUID, new_vector: List[float]) -> None:
    """Updates the semantic anchor vector after mathematical nudging."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE tomte_semantic_anchors SET embedding = %s::vector WHERE dictionary_id = %s;
        """, (new_vector, dictionary_id))

def upsert_keyword(conn: psycopg.Connection, dictionary_id: UUID, keyword: str, k_type: str) -> None:
    """Increments hits for an existing keyword or inserts a new one."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO tomte_keyword_registry (dictionary_id, keyword, keyword_type, hits)
            VALUES (%s, %s, %s, 1)
            ON CONFLICT (dictionary_id, keyword, keyword_type) 
            DO UPDATE SET hits = tomte_keyword_registry.hits + 1, last_modified_at = NOW();
        """, (dictionary_id, keyword, k_type))
   

def get_anchor_details(conn: psycopg.Connection, dictionary_id: UUID) -> Optional[Dict[str, Any]]:
    """Retrieves the current vector and total hits for nudging math."""
    sql = """
        -- CRITICAL FIX: Use COALESCE and LEFT JOIN
        SELECT a.embedding, COALESCE(SUM(r.hits), 0) as total_hits
        FROM tomte_semantic_anchors a
        LEFT JOIN tomte_keyword_registry r ON a.dictionary_id = r.dictionary_id
        WHERE a.dictionary_id = %s
        GROUP BY a.embedding;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (dictionary_id,))
        return cur.fetchone()

def create_new_anchor(conn: psycopg.Connection, vector: List[float]) -> UUID:
    """Phase 4.3 (Scenario B): Creates a new conceptual anchor ."""
    sql = "INSERT INTO tomte_semantic_anchors (embedding) VALUES (%s::vector) RETURNING dictionary_id;"
    with conn.cursor() as cur:
        cur.execute(sql, (vector,))
        return cur.fetchone()['dictionary_id']

def find_matches_above_threshold(conn: psycopg.Connection, vector: List[float], threshold: float) -> List[Dict[str, Any]]:
    """Phase 4.3 (Scenario A): Finds all anchors to update/nudge."""
    sql = """
        SELECT dictionary_id, (1 - (embedding <=> %(vec)s::vector)) as score
        FROM tomte_semantic_anchors
        WHERE (1 - (embedding <=> %(vec)s::vector)) > %(thresh)s
        ORDER BY score DESC;
    """
    with conn.cursor() as cur:
        # CRITICAL FIX: Pass arguments as a dictionary. 
        # Python passes the vector once; psycopg safely maps it to both placeholders.
        cur.execute(sql, {
            "vec": vector, 
            "thresh": threshold
        })
        return cur.fetchall()


def execute_trim_job(conn: psycopg.Connection) -> int:
    """Executes capacity-based trimming via atomic CTE."""
    sql = """
        WITH overweight_anchors AS (
            SELECT dictionary_id FROM tomte_keyword_registry 
            GROUP BY dictionary_id HAVING COUNT(*) > 100
        ),
        deleted_rows AS (
            DELETE FROM tomte_keyword_registry
            WHERE dictionary_id IN (SELECT dictionary_id FROM overweight_anchors)
            AND hits < 3 AND last_modified_at < NOW() - INTERVAL '1 year'
            RETURNING dictionary_id, keyword, keyword_type, hits, last_modified_at
        )
        INSERT INTO tomte_keyword_audit_log (dictionary_id, keyword, keyword_type, hits, last_modified_at)
        SELECT dictionary_id, keyword, keyword_type, hits, last_modified_at FROM deleted_rows;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.rowcount

def get_cross_anchor_merge_candidates(conn: psycopg.Connection) -> List[Dict[str, Any]]:
    """Uses pgvector self-join to find overlapping concepts."""
    sql = """
        SELECT a.dictionary_id as id_a, b.dictionary_id as id_b, 
               a.embedding as emb_a, b.embedding as emb_b
        FROM tomte_semantic_anchors a
        JOIN tomte_semantic_anchors b ON a.dictionary_id < b.dictionary_id
        WHERE (1 - (a.embedding <=> b.embedding)) > 0.9;
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return cur.fetchall()

def consolidate_duplicate_anchors(
    conn: psycopg.Connection, id_a: str, id_b: str, merged_vector: List[float], merged_kws: Dict
) -> None:
    """Atomically commits a cross-anchor merge."""
    with conn.cursor() as cur:
        cur.execute("UPDATE tomte_semantic_anchors SET embedding = %s::vector WHERE dictionary_id = %s", (merged_vector, id_a))
        cur.execute("DELETE FROM tomte_keyword_registry WHERE dictionary_id IN (%s, %s)", (id_a, id_b))
        
        for key, data in merged_kws.items():
            cur.execute("""
                INSERT INTO tomte_keyword_registry (dictionary_id, keyword, keyword_type, hits, created_at, last_modified_at)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (id_a, key[0], key[1], data['hits'], data['created_at'], data['last_modified_at']))
        
        cur.execute("DELETE FROM tomte_semantic_anchors WHERE dictionary_id = %s", (id_b,))

def get_anchors_for_internal_merge(conn: psycopg.Connection) -> List[str]:
    """Finds anchors that haven't been internally optimized in over a month."""
    sql = """
        SELECT dictionary_id FROM tomte_semantic_anchors
        WHERE last_internal_merge_at IS NULL 
           OR last_internal_merge_at < NOW() - INTERVAL '1 month';
    """
    with conn.cursor() as cur:
        cur.execute(sql)
        return [row['dictionary_id'] for row in cur.fetchall()]

def execute_internal_keyword_merge(
    conn: psycopg.Connection, anchor_id: str, kw_type: str, survivor: str, victims: List[str], total_hits: int
) -> None:
    """Updates the survivor's hits and deletes the victim synonyms."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE tomte_keyword_registry SET hits = %s, last_modified_at = NOW()
            WHERE dictionary_id = %s AND keyword_type = %s AND keyword = %s
        """, (total_hits, anchor_id, kw_type, survivor))
        
        cur.execute("""
            DELETE FROM tomte_keyword_registry 
            WHERE dictionary_id = %s AND keyword_type = %s AND keyword = ANY(%s)
        """, (anchor_id, kw_type, victims))

def mark_anchor_internal_merge_complete(conn: psycopg.Connection, anchor_id: str) -> None:
    with conn.cursor() as cur:
        cur.execute("UPDATE tomte_semantic_anchors SET last_internal_merge_at = NOW() WHERE dictionary_id = %s", (anchor_id,))
