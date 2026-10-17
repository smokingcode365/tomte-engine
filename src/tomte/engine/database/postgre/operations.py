import json
import psycopg
from tomte.engine.database.postgre.transactions import transactional
from typing import List, Dict, Any, Optional
from uuid import UUID, uuid4
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
        SELECT id, tenant_id, user_id, entity_id, payload FROM tomte_memory_engine
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
        SELECT id, tenant_id, user_id, entity_id, payload FROM tomte_memory_engine
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
        SELECT id, tenant_id, user_id, entity_id, payload FROM tomte_memory_engine
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
        SELECT id, tenant_id, user_id, entity_id, payload,
        (1 - (embedding <=> %s::vector)) * (mention_count * exp(-0.0001 * extract(epoch from (now() - last_mentioned_at)))) as relevance_score 
        FROM tomte_memory_engine 
        WHERE tenant_id = %s AND user_id = %s 
        ORDER BY relevance_score DESC LIMIT %s;
    """
    with conn.cursor() as cur:
        cur.execute(sql, (vector, tenant_id, user_id, limit))
        return cur.fetchall()

@transactional
def get_raw_semantic_context(conn, tenant_id, user_id, vector, limit):
    """Ungated long-term context; unlike activation, includes persona memories."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, tenant_id, user_id, entity_id, payload,
                   NULLIF(1 - (embedding <=> %s::vector), 'NaN'::double precision) AS raw_cosine
            FROM tomte_memory_engine
            WHERE tenant_id = %s AND user_id = %s
            ORDER BY raw_cosine DESC NULLS LAST, entity_id ASC, id ASC LIMIT %s
        """, (vector, tenant_id, user_id, limit))
        return cur.fetchall()


@transactional
def discover_session_entities(conn, tenant_id, user_id, vector, threshold, limit):
    """Separate raw-cosine activation; never update frequency/recency statistics."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, tenant_id, user_id, entity_id, payload,
                   1 - (embedding <=> %s::vector) AS raw_cosine
            FROM tomte_memory_engine
            WHERE tenant_id = %s AND user_id = %s
              AND entity_domain <> %s
              AND 1 - (embedding <=> %s::vector) >= %s
            ORDER BY raw_cosine DESC, entity_id ASC, id ASC LIMIT %s
        """, (vector, tenant_id, user_id, MemoryDomain.IDENTITY.value, vector, threshold, limit))
        return cur.fetchall()


@transactional
def find_session_entity_mentions(conn, tenant_id, user_id, message, vector, limit):
    """Scoped SQL phrase lookup: no full catalog transfer/deserialization."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT m.id, m.tenant_id, m.user_id, m.entity_id, m.payload,
                   1 - (m.embedding <=> %s::vector) AS raw_cosine
            FROM tomte_memory_engine m
            WHERE m.tenant_id = %s AND m.user_id = %s
              AND m.entity_domain <> %s AND EXISTS (
                SELECT 1 FROM (
                    SELECT m.entity_name AS term
                    UNION ALL
                    SELECT alias #>> '{}' FROM jsonb_array_elements(
                        CASE WHEN jsonb_typeof(m.payload->'aliases') = 'array'
                             THEN m.payload->'aliases' ELSE '[]'::jsonb END
                    ) alias WHERE jsonb_typeof(alias) = 'string'
                ) terms
                CROSS JOIN LATERAL (
                    SELECT trim(regexp_replace(lower(term), '[^[:alnum:]_]+', ' ', 'g')) AS phrase
                ) normalized
                WHERE phrase <> '' AND strpos(
                    ' ' || trim(regexp_replace(lower(%s), '[^[:alnum:]_]+', ' ', 'g')) || ' ',
                    ' ' || phrase || ' ') > 0
            )
            ORDER BY raw_cosine DESC NULLS LAST, m.entity_id ASC, m.id ASC LIMIT %s
        """, (vector, tenant_id, user_id, MemoryDomain.IDENTITY.value, message, limit))
        return cur.fetchall()


@transactional
def get_entities_by_ids(conn, tenant_id, user_id, entity_ids):
    """One scoped query, current payloads, input order, absent IDs ignored."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT id, tenant_id, user_id, entity_id, payload
            FROM tomte_memory_engine
            WHERE tenant_id = %s AND user_id = %s AND entity_id = ANY(%s::text[])
            ORDER BY array_position(%s::text[], entity_id), id ASC
        """, (tenant_id, user_id, entity_ids, entity_ids))
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
        # Stable lock order also supports batches spanning multiple namespaces.
        namespaces = sorted({json.dumps([op['tenant_id'], op['user_id']])
                             for op in updates if op.get('mode') == 'INSERT'})
        for namespace in namespaces:
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (namespace,))
        matched = sorted((op for op in updates if "_expected_payload" in op),
                         key=lambda op:(op["tenant_id"], op["user_id"], str(op["db_id"])))
        for op in matched:
            cur.execute("SELECT payload FROM tomte_memory_engine WHERE id=%s AND tenant_id=%s AND user_id=%s FOR UPDATE",
                        (op["db_id"], op["tenant_id"], op["user_id"]))
            current = cur.fetchone()
            if current is None or current["payload"] != op["_expected_payload"]:
                from tomte.engine.workers.control import StaleSnapshot
                raise StaleSnapshot("Stale assertion snapshot; retry the frozen batch")
        # Reserve every INSERT UUID before validating links, including forward
        # references. No inference from extraction or legacy logical labels.
        for op in updates:
            if op.get('mode') == 'INSERT':
                op.setdefault('reserved_id', str(uuid4()))
                UUID(op['reserved_id'])
            if op['entity_id'].startswith('tmp_'):
                raise ValueError('Temporary handles cannot be stored identities')
        reserved = {(op['tenant_id'], op['user_id'], str(op['reserved_id']))
                    for op in updates if op.get('mode') == 'INSERT'}
        if len(reserved) != sum(op.get('mode') == 'INSERT' for op in updates):
            raise ValueError('Duplicate reserved entity UUID')
        for op in updates:
            old_facts = op.get('_expected_payload', {}).get('facts', [])
            for fact in op['payload'].get('facts', []):
                legacy = fact.get('related_entity_ids', [])
                if legacy and not any(old.get('related_entity_ids', []) == legacy
                                      and old.get('source_chat_id') == fact.get('source_chat_id')
                                      and old.get('attribute_name') == fact.get('attribute_name')
                                      and old.get('value') == fact.get('value') for old in old_facts):
                    raise ValueError('New legacy relationship annotations are not supported')
                for target in fact.get('related_entity_uuids', []):
                    if str(UUID(target)) != target:
                        raise ValueError('Relationship target must be a canonical UUID')
                    if (op['tenant_id'], op['user_id'], target) in reserved:
                        continue
                    cur.execute('SELECT id FROM tomte_memory_engine WHERE id=%s AND tenant_id=%s AND user_id=%s FOR KEY SHARE',
                                (target, op['tenant_id'], op['user_id']))
                    if cur.fetchone() is None:
                        raise ValueError('Relationship endpoint missing or outside user scope')
        for op in updates:
            mode = op.get('mode')
            if mode == 'INSERT':
                # Allocation remains inside the write transaction and its namespace lock.
                canonical = op['entity_id']
                while True:
                    cur.execute("SELECT 1 FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s AND entity_id=%s",
                                (op['tenant_id'], op['user_id'], canonical))
                    if not cur.fetchone():
                        break
                    canonical = f"{op['entity_id']}_{uuid4().hex}"
                op['entity_id'] = canonical
                op['payload']['entity_id'] = canonical
                cur.execute("""
                    INSERT INTO tomte_memory_engine (
                        id, tenant_id, user_id, entity_id, entity_name, entity_domain, entity_type, payload, embedding, importance
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector, %s);
                """, (
                    op['reserved_id'], op['tenant_id'], op['user_id'], op['entity_id'], op['entity_name'],
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
                        importance = %s,
                        mention_count = mention_count + 1,
                        last_mentioned_at = NOW()
                    WHERE id = %s AND tenant_id = %s AND user_id = %s;
                """, (
                    json.dumps(op['payload']), op['embedding'], op.get('importance', 3),
                    op['db_id'], op['tenant_id'], op['user_id']
                ))

                # Prevent silent database failures if the ID doesn't actually exist
                if cur.rowcount == 0:
                    raise ValueError(f"REINFORCE failed: Record {op.get('db_id')} not found or namespace mismatch.")
                    
        return True


@transactional
def is_consolidation_processed(conn: psycopg.Connection, tenant_id: str, user_id: str, batch_id: str) -> bool:
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM tomte_processed_consolidations WHERE tenant_id=%s AND user_id=%s AND batch_id=%s", (tenant_id, user_id, batch_id))
        return cur.fetchone() is not None


@transactional
def claim_consolidation_batch(conn: psycopg.Connection, tenant_id: str, user_id: str, batch_id: str) -> bool:
    """Unique insertion serializes rivals; caller joins memory/feedback in its scope."""
    with conn.cursor() as cur:
        cur.execute("""
            INSERT INTO tomte_processed_consolidations (tenant_id, user_id, batch_id)
            VALUES (%s, %s, %s) ON CONFLICT DO NOTHING RETURNING batch_id
        """, (tenant_id, user_id, batch_id))
        return cur.fetchone() is not None


@transactional
def get_resolution_candidates(conn, tenant_id, user_id, names, domain, vector, context, limit=5):
    """Bounded scoped identity search; similarity proposes candidates, never a match."""
    if not 1 <= limit <= 5:
        raise ValueError("Resolution candidate limit must be between one and five")
    with conn.cursor() as cur:
        cur.execute(r"""
            WITH scored AS (
                SELECT id, entity_id, entity_name, entity_domain, entity_type, payload,
                    lower(regexp_replace(trim(entity_name), '\s+', ' ', 'g')) = ANY(%(names)s)
                    OR EXISTS (SELECT 1 FROM jsonb_array_elements_text(COALESCE(payload->'aliases','[]'::jsonb)) a
                        WHERE lower(regexp_replace(trim(a), '\s+', ' ', 'g')) = ANY(%(names)s)) AS exact_match,
                    (length(entity_name)>2 AND strpos(lower(%(context)s),lower(entity_name))>0)
                    OR EXISTS (SELECT 1 FROM jsonb_array_elements_text(COALESCE(payload->'aliases','[]'::jsonb)) a
                        WHERE length(a)>2 AND strpos(lower(%(context)s),lower(a))>0) AS context_match,
                    entity_domain=%(domain)s AS domain_match,
                    (1 - (embedding <=> %(vector)s::vector)) AS similarity
                FROM tomte_memory_engine WHERE tenant_id=%(tenant)s AND user_id=%(user)s
            ) SELECT *, sum((exact_match OR context_match)::int) OVER () AS identity_matches_total FROM scored
            ORDER BY exact_match DESC, (context_match AND domain_match) DESC, domain_match DESC,
                     context_match DESC, similarity DESC NULLS LAST, id
            LIMIT %(limit)s
        """, {"tenant":tenant_id,"user":user_id,"names":names,"domain":domain,
               "vector":vector,"context":context,"limit":limit})
        return cur.fetchall()
