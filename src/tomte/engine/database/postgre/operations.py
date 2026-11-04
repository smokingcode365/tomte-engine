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

def _semantic_rows(conn, tenant_id, user_id, vector, limit, *, threshold=None, exclude_identity=False, weighted=False):
    """Best matching current/history chunk discovers its owning scoped entity."""
    with conn.cursor() as cur:
        cur.execute("""
            WITH scored AS (
                SELECT m.id,m.tenant_id,m.user_id,m.entity_id,m.payload,m.mention_count,m.last_mentioned_at,
                    NULLIF(GREATEST(1-(m.embedding <=> %(vector)s::vector),
                        (SELECT max(1-(s.embedding <=> %(vector)s::vector)) FROM tomte_memory_search s
                         WHERE s.entity_uuid=m.id AND s.tenant_id=m.tenant_id AND s.user_id=m.user_id)),
                        'NaN'::double precision) AS raw_cosine
                FROM tomte_memory_engine m
                WHERE m.tenant_id=%(tenant)s AND m.user_id=%(user)s
                  AND (NOT %(exclude)s OR m.entity_domain<>%(identity)s)
            ), ranked AS (
                SELECT *, CASE WHEN %(weighted)s THEN
                    raw_cosine * (mention_count * exp(-0.0001 * extract(epoch from (now()-last_mentioned_at))))
                    ELSE raw_cosine END AS relevance_score
                FROM scored WHERE %(threshold)s::double precision IS NULL OR raw_cosine >= %(threshold)s
            )
            SELECT id,tenant_id,user_id,entity_id,payload,raw_cosine,relevance_score FROM ranked
            ORDER BY relevance_score DESC NULLS LAST, entity_id,id LIMIT %(limit)s
        """, {"vector": vector, "tenant": tenant_id, "user": user_id, "exclude": exclude_identity,
              "identity": MemoryDomain.IDENTITY.value, "weighted": weighted, "threshold": threshold, "limit": limit})
        return cur.fetchall()


@transactional
def get_semantic_context(conn, tenant_id, user_id, vector, limit):
    return _semantic_rows(conn, tenant_id, user_id, vector, limit or settings.semantic_retrieval_limit, weighted=True)


@transactional
def get_raw_semantic_context(conn, tenant_id, user_id, vector, limit):
    return _semantic_rows(conn, tenant_id, user_id, vector, limit or settings.semantic_retrieval_limit)


@transactional
def discover_session_entities(conn, tenant_id, user_id, vector, threshold, limit):
    return _semantic_rows(conn, tenant_id, user_id, vector, limit, threshold=threshold, exclude_identity=True)


@transactional
def get_history_candidates(conn, tenant_id, user_id, entity_ids, vector, limit):
    """Return bounded positional handles; memory text comes from canonical JSON."""
    with conn.cursor() as cur:
        cur.execute("""
            WITH scored AS (
                SELECT entity_uuid,fact_id,history_index,max(1-(embedding <=> %s::vector)) AS similarity
                FROM tomte_memory_search WHERE tenant_id=%s AND user_id=%s
                    AND entity_uuid=ANY(%s::uuid[]) AND history_index>=0
                GROUP BY entity_uuid,fact_id,history_index
            ), ranked AS (
                SELECT *,row_number() OVER (PARTITION BY entity_uuid ORDER BY similarity DESC, fact_id,history_index) AS rank
                FROM scored
            )
            SELECT entity_uuid,fact_id,history_index,similarity FROM ranked WHERE rank<=%s
            ORDER BY entity_uuid,rank
        """, (vector, tenant_id, user_id, entity_ids, limit))
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
    """Atomic scoped writes with canonical UUIDs and immutable history prefixes."""
    from tomte.engine.core.extraction import validate_operation_provenance
    from tomte.engine.core.memory import public_entity
    from tomte.engine.llm.embedder import get_search_entries, changed_search_entries, search_entry_key
    from tomte.engine.workers.control import StaleSnapshot
    import math
    validate_operation_provenance(updates)
    identities = [(op["tenant_id"], op["user_id"], op["entity_id"]) for op in updates]
    if len(set(identities)) != len(identities):
        raise ValueError("Duplicate entity operation")
    for op in updates:
        if op["mode"] not in {"INSERT", "MUTATE", "REINFORCE"}:
            raise ValueError("Unknown memory operation")
        if not op["payload"]["facts"]:
            raise ValueError("Cannot persist an empty entity")
        if any(op[key] != op["payload"][key] for key in ("entity_name", "entity_type", "entity_domain")):
            raise ValueError("Row identity metadata disagrees with payload")
        if op["mode"] != "INSERT" and "_expected_payload" not in op:
            raise ValueError("Updates require an expected payload")
        old = op.get("_expected_payload")
        if old:
            current = {f["fact_id"]: f for f in op["payload"]["facts"]}
            for fact in old["facts"]:
                new = current.get(fact["fact_id"])
                if new is None or new["attribute_name"] != fact["attribute_name"]:
                    raise ValueError("Existing fact identity cannot be removed or changed")
                if new["history_values"][:len(fact["history_values"])] != fact["history_values"]:
                    raise ValueError("Existing history is immutable")
        changed = old is None or public_entity(old) != public_entity(op["payload"])
        if changed:
            expected = [{k: v for k, v in entry.items() if k != "text"} for entry in changed_search_entries(op["payload"], old)]
            entries = op.get("search_entries", [])
            if [{k: v for k, v in entry.items() if k != "embedding"} for entry in entries] != expected:
                raise ValueError("Search entries must cover every changed current and historical item")
            vectors = [op.get("embedding"), *[entry.get("embedding") for entry in entries]]
            if any(not isinstance(v, list) or not v or not any(v) or
                   any(type(x) not in (int, float) or not math.isfinite(x) for x in v) for v in vectors):
                raise ValueError("Missing or invalid memory embedding")
            if any(len(v) != len(vectors[0]) for v in vectors):
                raise ValueError("Memory embedding dimensions differ")
        op["_write_search_entries"] = changed
    with conn.cursor() as cur:
        namespaces = sorted({json.dumps([op["tenant_id"], op["user_id"]]) for op in updates if op["mode"] == "INSERT"})
        for namespace in namespaces:
            cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (namespace,))
        for op in sorted((o for o in updates if o["mode"] != "INSERT"), key=lambda o: (o["tenant_id"], o["user_id"], o["entity_id"])):
            cur.execute("SELECT payload FROM tomte_memory_engine WHERE id=%s AND tenant_id=%s AND user_id=%s FOR UPDATE",
                        (op["db_id"], op["tenant_id"], op["user_id"]))
            current = cur.fetchone()
            if current is None or current["payload"] != op["_expected_payload"]:
                raise StaleSnapshot("Stale fact snapshot; retry the frozen batch")
        for op in updates:
            facts = op["payload"]["facts"]
            mentions = sum(f["_mention_count"] for f in facts)
            last_mentioned = max(f["_last_mentioned_at"] for f in facts)
            importance = max(f["_importance"] for f in facts)
            if op["mode"] == "INSERT":
                cur.execute("""
                    INSERT INTO tomte_memory_engine
                    (id,tenant_id,user_id,entity_id,entity_name,entity_domain,entity_type,payload,embedding,importance,mention_count,last_mentioned_at)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::vector,%s,%s,%s::timestamptz)
                """, (op["reserved_id"], op["tenant_id"], op["user_id"], op["entity_id"], op["entity_name"],
                      op["entity_domain"], op["entity_type"], json.dumps(op["payload"]), op["embedding"],
                      importance, mentions, last_mentioned))
            else:
                cur.execute("""
                    UPDATE tomte_memory_engine SET payload=%s::jsonb,
                        embedding=CASE WHEN %s THEN %s::vector ELSE embedding END,
                        importance=%s,mention_count=%s,last_mentioned_at=%s::timestamptz,
                        entity_name=%s,entity_domain=%s,entity_type=%s
                    WHERE id=%s AND tenant_id=%s AND user_id=%s
                """, (json.dumps(op["payload"]), op["_write_search_entries"], op.get("embedding"),
                      importance, mentions, last_mentioned, op["entity_name"], op["entity_domain"], op["entity_type"],
                      op["db_id"], op["tenant_id"], op["user_id"]))
                if cur.rowcount != 1:
                    raise ValueError("Memory update target absent or outside namespace")
            if op["_write_search_entries"]:
                old_keys = {search_entry_key(e) for e in get_search_entries(op["_expected_payload"])} if op.get("_expected_payload") else set()
                new_keys = {search_entry_key(e) for e in get_search_entries(op["payload"])}
                for key in old_keys - new_keys:
                    cur.execute("""DELETE FROM tomte_memory_search WHERE entity_uuid=%s AND tenant_id=%s AND user_id=%s
                        AND fact_id=%s AND history_index=%s AND field_name=%s AND chunk_index=%s""",
                        (op["entity_id"], op["tenant_id"], op["user_id"], *key))
                for entry in op["search_entries"]:
                    cur.execute("""
                        INSERT INTO tomte_memory_search (entity_uuid,tenant_id,user_id,fact_id,history_index,field_name,chunk_index,embedding)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s::vector)
                        ON CONFLICT(entity_uuid,fact_id,history_index,field_name,chunk_index)
                        DO UPDATE SET embedding=EXCLUDED.embedding
                    """, (op["entity_id"], op["tenant_id"], op["user_id"], entry["fact_id"], entry["history_index"],
                          entry["field_name"], entry["chunk_index"], entry["embedding"]))
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
                    GREATEST(1 - (embedding <=> %(vector)s::vector),
                        (SELECT max(1-(s.embedding <=> %(vector)s::vector)) FROM tomte_memory_search s
                         WHERE s.entity_uuid=tomte_memory_engine.id AND s.tenant_id=tomte_memory_engine.tenant_id
                         AND s.user_id=tomte_memory_engine.user_id)) AS similarity
                FROM tomte_memory_engine WHERE tenant_id=%(tenant)s AND user_id=%(user)s
            ) SELECT *, sum((exact_match OR context_match)::int) OVER () AS identity_matches_total FROM scored
            ORDER BY exact_match DESC, (context_match AND domain_match) DESC, domain_match DESC,
                     context_match DESC, similarity DESC NULLS LAST, id
            LIMIT %(limit)s
        """, {"tenant":tenant_id,"user":user_id,"names":names,"domain":domain,
               "vector":vector,"context":context,"limit":limit})
        return cur.fetchall()
