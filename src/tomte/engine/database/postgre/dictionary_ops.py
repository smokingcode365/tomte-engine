# tomte/engine/database/postgre/dictionary_ops.py

import json
from typing import List, Dict, Any, Optional
from uuid import UUID
import psycopg

def get_semantic_keywords(
    conn: psycopg.Connection, 
    vector: List[float], 
    threshold: float, 
    id_limit: int
) -> List[Dict[str, Any]]:
    """
    Phase 1: Retrieves ALL keywords for the Top K unique semantic IDs.
    """
    sql = """
        SELECT r.keyword, r.keyword_type, r.hits, a.dictionary_id
        FROM (
            -- Subquery: Find the Top K unique concept IDs based on the vector
            SELECT dictionary_id
            FROM semantic_anchors
            WHERE (1 - (embedding <=> %s::vector)) > %s
            ORDER BY (1 - (embedding <=> %s::vector)) DESC
            LIMIT %s
        ) a
        JOIN keyword_registry r ON a.dictionary_id = r.dictionary_id;
    """
    with conn.cursor() as cur:
        # Vector is used for both filtering and ordering
        cur.execute(sql, (vector, threshold, vector, id_limit))
        return cur.fetchall()

def find_anchor_match(conn: psycopg.Connection, vector: List[float], threshold: float) -> Optional[UUID]:
    """
    Phase 4: Checks if a new entity vector matches an existing semantic anchor.
    """
    sql = """
        SELECT dictionary_id FROM semantic_anchors 
        WHERE (1 - (embedding <=> %s::vector)) > %s
        ORDER BY (1 - (embedding <=> %s::vector)) DESC
        LIMIT 1;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (vector, threshold, vector))
        row = cur.fetchone()
        return row['dictionary_id'] if row else None

def update_anchor_vector(conn: psycopg.Connection, dictionary_id: UUID, new_vector: List[float]) -> None:
    """Updates the semantic anchor vector after mathematical nudging."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE semantic_anchors SET embedding = %s::vector WHERE dictionary_id = %s;
        """, (new_vector, dictionary_id))

def upsert_keyword(conn: psycopg.Connection, dictionary_id: UUID, keyword: str, k_type: str) -> None:
    """Increments hits for an existing keyword or inserts a new one."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO keyword_registry (dictionary_id, keyword, keyword_type, hits)
            VALUES (%s, %s, %s, 1)
            ON CONFLICT (dictionary_id, keyword, keyword_type) 
            DO UPDATE SET hits = keyword_registry.hits + 1, last_modified_at = NOW();
        """, (dictionary_id, keyword, k_type))
   

def get_anchor_details(conn: psycopg.Connection, dictionary_id: UUID) -> Optional[Dict[str, Any]]:
    """Retrieves the current vector and total hits for nudging math."""
    sql = """
        -- CRITICAL FIX: Use COALESCE and LEFT JOIN
        SELECT a.embedding, COALESCE(SUM(r.hits), 0) as total_hits
        FROM semantic_anchors a
        LEFT JOIN keyword_registry r ON a.dictionary_id = r.dictionary_id
        WHERE a.dictionary_id = %s
        GROUP BY a.embedding;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (dictionary_id,))
        return cur.fetchone()

def create_new_anchor(conn: psycopg.Connection, vector: List[float]) -> UUID:
    """Phase 4.3 (Scenario B): Creates a new conceptual anchor ."""
    sql = "INSERT INTO semantic_anchors (embedding) VALUES (%s::vector) RETURNING dictionary_id;"
    with conn.cursor() as cur:
        cur.execute(sql, (vector,))
        return cur.fetchone()['dictionary_id']

def find_matches_above_threshold(conn: psycopg.Connection, vector: List[float], threshold: float) -> List[Dict[str, Any]]:
    """Phase 4.3 (Scenario A): Finds all anchors to update/nudge [cite: 164-166]."""
    sql = """
        SELECT dictionary_id, (1 - (embedding <=> %s::vector)) as score
        FROM semantic_anchors
        WHERE (1 - (embedding <=> %s::vector)) > %s
        ORDER BY score DESC;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (vector, vector, threshold))
        return cur.fetchall()