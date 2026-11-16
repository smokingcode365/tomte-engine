"""Scoped, value-free entity and fact catalogue reads."""
from tomte.engine.database.postgre.transactions import transactional


def _page(rows, limit):
    return {'rows': rows[:limit], 'is_partial': len(rows) > limit}


@transactional
def get_domains(conn, tenant, user, limit):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT tenant_id,user_id,entity_domain
            FROM tomte_memory_engine
            WHERE tenant_id=%(tenant)s AND user_id=%(user)s
              AND entity_domain <> 'unresolved_memories'
            GROUP BY tenant_id,user_id,entity_domain
            ORDER BY entity_domain LIMIT %(limit)s
        """, {'tenant': tenant, 'user': user, 'limit': limit + 1})
        return _page(cur.fetchall(), limit)


@transactional
def get_entities(conn, tenant, user, domains, limit):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT tenant_id,user_id,entity_domain,entity_id,entity_type,
                   payload->>'entity_name' AS entity_name,payload->'aliases' AS aliases
            FROM tomte_memory_engine
            WHERE tenant_id=%(tenant)s AND user_id=%(user)s
              AND entity_domain <> 'unresolved_memories'
              AND (%(domains)s::text[] IS NULL OR entity_domain = ANY(%(domains)s::text[]))
            ORDER BY entity_domain,entity_id LIMIT %(limit)s
        """, {'tenant': tenant, 'user': user, 'domains': domains, 'limit': limit + 1})
        return _page(cur.fetchall(), limit)


@transactional
def get_facts(conn, tenant, user, entity_ids, limit):
    with conn.cursor() as cur:
        cur.execute("""
            SELECT m.tenant_id,m.user_id,m.entity_id,
                   f.fact->>'fact_id' AS fact_id,
                   f.fact->>'attribute_name' AS attribute_name,
                   array_agg(DISTINCT k.field_name ORDER BY k.field_name) AS field_names
            FROM tomte_memory_engine m
            CROSS JOIN LATERAL jsonb_array_elements(m.payload->'facts') f(fact)
            CROSS JOIN LATERAL (
                SELECT f.fact->'value' AS value
                UNION ALL
                SELECT h.value->'value'
                FROM jsonb_array_elements(f.fact->'history_values') h(value)
            ) v
            CROSS JOIN LATERAL jsonb_object_keys(v.value) k(field_name)
            WHERE m.tenant_id=%(tenant)s AND m.user_id=%(user)s
              AND m.entity_domain <> 'unresolved_memories'
              AND m.entity_id = ANY(%(entity_ids)s)
            GROUP BY m.tenant_id,m.user_id,m.entity_id,
                     f.fact->>'fact_id',f.fact->>'attribute_name'
            ORDER BY m.entity_id,f.fact->>'fact_id' LIMIT %(limit)s
        """, {'tenant': tenant, 'user': user, 'entity_ids': entity_ids, 'limit': limit + 1})
        return _page(cur.fetchall(), limit)
