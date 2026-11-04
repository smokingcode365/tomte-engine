"""Synthetic fixtures for memory-format v2 evaluations; never a runtime migration.

Labels remain in evaluation gold data. Only scoped UUIDs enter persistence.
Text matching below is evaluation scoring, not a production language policy.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
from uuid import NAMESPACE_URL, uuid5

from tomte.engine.core.memory import Entity, Fact
from tomte.engine.llm.embedder import get_entity_summary, get_search_entries

MEMORY_FORMAT = 2


def fixture_uuid(tenant, user, label):
    return str(uuid5(NAMESPACE_URL, json.dumps(["tomte-eval-v2", tenant, user, label])))


def value(text, confidence="high"):
    return {"detail": {"text": text, "fact_confidence": confidence}}


def wire_fields(text, confidence="high"):
    return [{"field_name": "detail", "text": text, "fact_confidence": confidence}]


def fact(attribute, text, *, key=None, stamp=None, importance=3):
    stamp = stamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return Fact.model_validate({"fact_id": fixture_uuid("fact", attribute, key or text),
        "attribute_name": attribute,
        "value": {"detail": {**value(text)["detail"], "_recorded_at": stamp, "_updated_at": stamp}},
        "_importance": importance, "_mention_count": 1, "_last_mentioned_at": stamp}).model_dump(mode="json")


def entity(label, name, domain, kind, aliases, pairs):
    return Entity(entity_id=fixture_uuid("dataset", "entity", label), entity_name=name,
        entity_domain=domain, entity_type=kind, aliases=list(aliases),
        facts=[fact(k, v, key=f"{label}:{i}") for i, (k, v) in enumerate(pairs)]).model_dump(mode="json")


def search_texts(payload):
    return [get_entity_summary(payload), *[e["text"] for e in get_search_entries(payload)]]


def fixture_write(tenant, user, payload, vectors):
    """Prepare synthetic INSERT using genuine vectors for all indexed content."""
    payload = deepcopy(payload)
    payload["entity_id"] = fixture_uuid(tenant, user, payload["entity_id"])
    payload = Entity.model_validate(payload).model_dump(mode="json")
    return {"mode": "INSERT", "tenant_id": tenant, "user_id": user,
        **{k: payload[k] for k in ("entity_id", "entity_name", "entity_domain", "entity_type")},
        "reserved_id": payload["entity_id"], "payload": payload,
        "embedding": vectors[get_entity_summary(payload)],
        "search_entries": [{**{k: v for k, v in entry.items() if k != "text"},
                            "embedding": vectors[entry["text"]]} for entry in get_search_entries(payload)]}


def label_map(tenant, user, bank):
    return {fixture_uuid(tenant, user, payload["entity_id"]): label for label, payload in bank.items()}


def cosine(left, right):
    return sum(x*y for x, y in zip(left, right)) / (sum(x*x for x in left)*sum(y*y for y in right))**.5


def similarity(vector, payload, vectors):
    return max(cosine(vector, vectors[text]) for text in search_texts(payload))


def database_scores(db, tenant, user, vector):
    """Independent SQL oracle for max aggregate/current/history similarity."""
    return db.conn.execute("""
        SELECT m.entity_id,
            GREATEST(1-(m.embedding <=> %(vector)s::vector),
                COALESCE((SELECT MAX(1-(s.embedding <=> %(vector)s::vector))
                    FROM tomte_memory_search s WHERE s.tenant_id=m.tenant_id
                    AND s.user_id=m.user_id AND s.entity_uuid=m.id), -1)) AS cosine,
            m.mention_count*exp(-0.0001*extract(epoch FROM (now()-m.last_mentioned_at))) AS factor
        FROM tomte_memory_engine m WHERE m.tenant_id=%(tenant)s AND m.user_id=%(user)s
        """, {"vector": vector, "tenant": tenant, "user": user}).fetchall()


def current_facts(payload):
    return payload["facts"]


def historical_facts(payload):
    return [{"attribute_name": f["attribute_name"], "value": h["value"]}
            for f in payload["facts"] for h in f["history_values"]]


def item_texts(fact):
    return [item["text"] for item in fact["value"].values()]
