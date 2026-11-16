"""Transactional plain-field BM25. Corpus statistics never cross a user boundary."""
import json
from tomte.engine.core.memory import value_search_text

from tomte.engine.database.postgre.transactions import transactional
from tomte.engine.exceptions import TomteConfigurationError

def value_entries(payload):
    """Whole field values: embedding chunk boundaries must not split identifiers."""
    return [{"fact_id": fact['fact_id'], "history_index": index,
             "field_name": field, "text": value_search_text(item)}
            for fact in payload['facts']
            for index, value in [(-1, fact['value']), *enumerate(h['value'] for h in fact['history_values'])]
            for field, item in value.items()]


def replace_entity_index(cur, tenant_id, user_id, entity_id, payload):
    """Caller holds the canonical write transaction/row lock; derive, never trust wire terms."""
    entries = value_entries(payload)
    cur.execute("DELETE FROM tomte_memory_lexical WHERE tenant_id=%s AND user_id=%s AND entity_uuid=%s",
                (tenant_id, user_id, entity_id))
    cur.execute("""
        WITH entries AS (
            SELECT * FROM jsonb_to_recordset(%s::jsonb) AS e(
                fact_id uuid, history_index int, field_name text, text text)
        ), frequencies AS (
            SELECT e.fact_id,e.history_index,e.field_name,t.term,count(*) AS frequency
            FROM entries e
            CROSS JOIN LATERAL ts_debug('pg_catalog.simple'::regconfig,e.text) AS parsed
            CROSS JOIN LATERAL unnest(parsed.lexemes) AS t(term)
            GROUP BY e.fact_id,e.history_index,e.field_name,t.term
        ), tokenized AS (
            SELECT fact_id,history_index,field_name,
                   jsonb_object_agg(term,frequency) AS terms,sum(frequency)::int AS document_length
            FROM frequencies GROUP BY fact_id,history_index,field_name
        )
        INSERT INTO tomte_memory_lexical
            (entity_uuid,tenant_id,user_id,fact_id,history_index,field_name,terms,document_length)
        SELECT %s,%s,%s,e.fact_id,e.history_index,e.field_name,
               COALESCE(t.terms,'{}'::jsonb),COALESCE(t.document_length,0)
        FROM entries e LEFT JOIN tokenized t USING(fact_id,history_index,field_name)
    """, (json.dumps(entries), entity_id, tenant_id, user_id))
    cur.execute("UPDATE tomte_memory_engine SET lexical_indexed=TRUE WHERE id=%s AND tenant_id=%s AND user_id=%s",
                (entity_id, tenant_id, user_id))
    if cur.rowcount != 1:
        raise ValueError("Lexical index target absent or outside namespace")


def require_index(conn, tenant_id, user_id):
    with conn.cursor() as cur:
        cur.execute("""SELECT 1 FROM tomte_memory_engine
            WHERE tenant_id=%s AND user_id=%s AND entity_domain<>'unresolved_memories'
              AND NOT lexical_indexed LIMIT 1""", (tenant_id, user_id))
        if cur.fetchone():
            raise TomteConfigurationError(
                "BM25 index is incomplete for this user; run TomteAdminClient.rebuild_lexical_index(user_id) first")


# Keep statistics over the complete scoped corpus, not just keyword matches or
# vector candidates. k1=1.2, b=.75; unique query terms, positive IDF. No stemming
# or language-specific stop list. NOT MATERIALIZED permits indexed term matching.
# Materialize aggregate statistics once: with sparse owner estimates PostgreSQL
# can otherwise rescan the corpus for every matching term, without changing scores.
SCORED_FIELDS_SQL = """
    WITH corpus AS NOT MATERIALIZED (
        SELECT l.* FROM tomte_memory_lexical l
        JOIN tomte_memory_engine m ON m.id=l.entity_uuid
            AND m.tenant_id=l.tenant_id AND m.user_id=l.user_id
        WHERE l.tenant_id=%(tenant)s AND l.user_id=%(user)s
          AND m.entity_domain<>'unresolved_memories'
    ), stats AS MATERIALIZED (
        SELECT count(*)::double precision AS n,avg(document_length)::double precision AS avg_length FROM corpus
    ), query_terms AS (
        SELECT unnest(tsvector_to_array(to_tsvector('pg_catalog.simple'::regconfig,%(query)s))) AS term
    ), matches AS (
        SELECT c.entity_uuid,c.fact_id,c.history_index,c.field_name,c.document_length,
               q.term,(c.terms->>q.term)::double precision AS tf
        FROM corpus c JOIN query_terms q ON c.terms ? q.term
        WHERE c.terms ?| ARRAY(SELECT term FROM query_terms)
    ), frequencies AS MATERIALIZED (
        SELECT term,count(*)::double precision AS df FROM matches GROUP BY term
    ), scored AS (
        SELECT m.entity_uuid,m.fact_id,m.history_index,m.field_name,
               sum(ln(1+(s.n-f.df+0.5)/(f.df+0.5)) * (m.tf*2.2)
                   / (m.tf+1.2*(0.25+0.75*m.document_length/NULLIF(s.avg_length,0)))) AS bm25_score
        FROM matches m JOIN frequencies f USING(term) CROSS JOIN stats s
        GROUP BY m.entity_uuid,m.fact_id,m.history_index,m.field_name
    )
"""


@transactional
def get_bm25_context(conn, tenant_id, user_id, query, limit):
    require_index(conn, tenant_id, user_id)
    with conn.cursor() as cur:
        cur.execute(SCORED_FIELDS_SQL + """
            , best AS (
                SELECT entity_uuid,max(bm25_score) AS bm25_score FROM scored GROUP BY entity_uuid
            )
            SELECT m.*,b.bm25_score FROM best b JOIN tomte_memory_engine m ON m.id=b.entity_uuid
            WHERE m.tenant_id=%(tenant)s AND m.user_id=%(user)s
            ORDER BY b.bm25_score DESC,m.entity_id,m.id LIMIT %(limit)s
        """, {"tenant": tenant_id, "user": user_id, "query": query, "limit": limit})
        return cur.fetchall()


@transactional
def get_bm25_history(conn, tenant_id, user_id, entity_ids, query, limit):
    require_index(conn, tenant_id, user_id)
    with conn.cursor() as cur:
        cur.execute(SCORED_FIELDS_SQL + """
            , best AS (
                SELECT entity_uuid,fact_id,history_index,max(bm25_score) AS bm25_score FROM scored
                WHERE history_index>=0 AND entity_uuid=ANY(%(ids)s::uuid[])
                GROUP BY entity_uuid,fact_id,history_index
            ), ranked AS (
                SELECT *,row_number() OVER(PARTITION BY entity_uuid
                    ORDER BY bm25_score DESC,fact_id,history_index) AS rank FROM best
            )
            SELECT * FROM ranked WHERE rank<=%(limit)s ORDER BY entity_uuid,rank
        """, {"tenant": tenant_id, "user": user_id, "query": query, "ids": entity_ids, "limit": limit})
        return cur.fetchall()


@transactional
def rebuild_user_index(conn, tenant_id, user_id):
    """Atomic/idempotent rebuild from current canonical payloads, without model calls."""
    with conn.cursor() as cur:
        cur.execute("""SELECT id,payload FROM tomte_memory_engine
            WHERE tenant_id=%s AND user_id=%s ORDER BY id FOR UPDATE""", (tenant_id, user_id))
        rows = cur.fetchall()
        for row in rows:
            replace_entity_index(cur, tenant_id, user_id, row['id'], row['payload'])
        return len(rows)
