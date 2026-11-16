"""Read-only answer lookup over canonical JSON and existing plain-value vectors."""
import json
from tomte.engine.database.postgre.transactions import transactional


SNAPSHOTS = """
    WITH facts AS (
        SELECT m.id,m.tenant_id,m.user_id,m.entity_id,m.entity_type,m.entity_domain,
               m.payload,f.fact
        FROM tomte_memory_engine m
        CROSS JOIN LATERAL jsonb_array_elements(m.payload->'facts') f(fact)
        WHERE m.tenant_id=%(tenant)s AND m.user_id=%(user)s
          AND m.entity_domain <> 'unresolved_memories'
    ), snapshots AS (
        SELECT f.*,v.value,v.history_index FROM facts f
        CROSS JOIN LATERAL (
            SELECT f.fact->'value' AS value,-1::bigint AS history_index
            UNION ALL
            SELECT h.value->'value',h.ordinality-1
            FROM jsonb_array_elements(f.fact->'history_values') WITH ORDINALITY h(value,ordinality)
        ) v
    )
"""


@transactional
def get_catalog(conn, tenant, user, limit):
    with conn.cursor() as cur:
        cur.execute(SNAPSHOTS + """
            SELECT entity_domain,entity_type,fact->>'attribute_name' AS attribute_name,
                   k.field_name,count(DISTINCT id) AS entity_count
            FROM snapshots CROSS JOIN LATERAL jsonb_object_keys(value) k(field_name)
            GROUP BY entity_domain,entity_type,fact->>'attribute_name',k.field_name
            ORDER BY entity_domain,entity_type,attribute_name,k.field_name LIMIT %(limit)s
        """, {'tenant':tenant,'user':user,'limit':limit+1})
        rows = cur.fetchall()
    return {'paths': rows[:limit], 'is_partial': len(rows)>limit}


@transactional
def get_facts(conn, tenant, user, vector, paths, limit, max_packet_bytes):
    with conn.cursor() as cur:
        cur.execute(SNAPSHOTS + """
            , requested AS (
                SELECT * FROM jsonb_to_recordset(%(paths)s::jsonb)
                AS p(entity_domain text,entity_type text,attribute_name text,field_name text,
                     entity_id text,fact_id text)
            ), matched AS (
                SELECT s.*,EXISTS (
                    SELECT 1 FROM requested p WHERE
                      (p.entity_id=s.entity_id AND p.fact_id=s.fact->>'fact_id')
                      OR (p.entity_id IS NULL AND p.entity_domain=s.entity_domain
                        AND p.entity_type=s.entity_type AND p.attribute_name=s.fact->>'attribute_name'
                        AND s.value ? p.field_name)
                ) AS path_match FROM snapshots s
            ), scoped AS (
                SELECT s.* FROM matched s
                WHERE NOT EXISTS (SELECT 1 FROM requested)
                   OR s.path_match
                   OR s.id IN (
                       SELECT m.id FROM matched m JOIN requested p
                         ON p.entity_id IS NULL AND p.entity_domain=m.entity_domain
                        AND p.entity_type=m.entity_type AND p.attribute_name=m.fact->>'attribute_name'
                        AND m.value ? p.field_name
                   )
            ), scored AS (
                SELECT s.*,(SELECT max(NULLIF(1-(r.embedding <=> %(vector)s::vector),'NaN'::float8))
                    FROM tomte_memory_retrieval r
                    WHERE r.tenant_id=s.tenant_id AND r.user_id=s.user_id AND r.entity_uuid=s.id
                      AND r.fact_id=(s.fact->>'fact_id')::uuid AND r.history_index=s.history_index
                ) AS similarity,
                jsonb_build_object(
                    'entity',jsonb_strip_nulls(jsonb_build_object(
                        'entity_id',s.entity_id,'entity_name',s.payload->'entity_name',
                        'entity_type',s.entity_type,'entity_domain',s.entity_domain,
                        'aliases',s.payload->'aliases','first_mentioned_at',s.payload->'first_mentioned_at')),
                    'fact',jsonb_build_object('fact_id',s.fact->'fact_id',
                        'attribute_name',s.fact->'attribute_name',
                        'value',(SELECT jsonb_object_agg(k,jsonb_strip_nulls(jsonb_build_object('text',v->'text',
                            'fact_confidence',v->'fact_confidence','context',v->'context'))) FROM jsonb_each(s.value) i(k,v))),
                    'value_kind',CASE WHEN s.history_index=-1 THEN 'current' ELSE 'history' END,
                    'history_index',s.history_index
                ) AS packet FROM scoped s
            ), stats AS (
                SELECT count(*) AS total_facts,
                    count(*) FILTER(WHERE octet_length(packet::text)>%(bytes)s) AS oversized_facts
                FROM scored
            ), chosen AS (
                -- Rank snapshots, not entities: a second relevant fact must not
                -- wait behind every other entity's first fact.
                SELECT * FROM scored WHERE octet_length(packet::text)<=%(bytes)s
                ORDER BY path_match DESC,similarity DESC NULLS LAST,
                    entity_id,fact->>'fact_id',history_index LIMIT %(limit)s
            )
            SELECT stats.*,c.tenant_id,c.user_id,c.packet,c.similarity,c.path_match
            FROM stats LEFT JOIN chosen c ON true
            ORDER BY c.path_match DESC,c.similarity DESC NULLS LAST,
                c.entity_id,c.fact->>'fact_id',c.history_index
        """, {'tenant':tenant,'user':user,'vector':vector,'paths':json.dumps(paths),
              'limit':limit,'bytes':max_packet_bytes})
        rows = cur.fetchall()
    return {'facts': [dict(tenant_id=r['tenant_id'],user_id=r['user_id'],packet=r['packet'])
                      for r in rows if r['packet'] is not None],
            'total_facts': rows[0]['total_facts'], 'oversized_facts': rows[0]['oversized_facts']}
