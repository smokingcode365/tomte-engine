"""Explicit synthetic memory builders. These are not runtime migration adapters."""
from copy import deepcopy
from uuid import UUID, NAMESPACE_URL, uuid4, uuid5
import json
from tomte.engine.core.memory import Fact, Entity
from tomte.engine.llm.embedder import changed_search_entries

STAMP = "2026-10-09T12:00:00Z"


def fixture_id(tenant, user, label):
    """Deterministic, scoped UUID for synthetic test labels only."""
    try:
        return str(UUID(label))
    except ValueError:
        return str(uuid5(NAMESPACE_URL, json.dumps([tenant, user, label])))


def value(text, confidence="high"):
    return {"detail": {"text": text, "fact_confidence": confidence}}


def field_entries(text, confidence="high"):
    return [{"field_name": "detail", "text": text, "fact_confidence": confidence}]


def fact_text(fact):
    return fact["value"]["detail"]["text"]


def simple_fact(attribute, text, *, confidence="high", importance=3, fact_id=None, mentions=1, stamp=STAMP):
    return Fact.model_validate({"fact_id": fact_id or str(uuid4()), "attribute_name": attribute,
        "value": {"detail": {"text": text, "fact_confidence": confidence, "_recorded_at": stamp, "_updated_at": stamp}},
        "_importance": importance, "_mention_count": mentions, "_last_mentioned_at": stamp}).model_dump(mode="json")


def simple_entity(name="Synthetic entity", *, entity_id=None, kind="person", domain="relationships", facts=(), aliases=()):
    return Entity(entity_id=entity_id or str(uuid4()), entity_name=name, entity_type=kind,
                  entity_domain=domain, aliases=list(aliases), facts=list(facts)).model_dump(mode="json")


def prepare_write(tenant, user, payload, *, old=None, vector=None):
    embedding = vector or [1.0] + [0.0] * 1535
    entity = deepcopy(payload)
    op = {"tenant_id": tenant, "user_id": user, "payload": entity, "mode": "INSERT" if old is None else "MUTATE",
          **{k: entity[k] for k in ("entity_id", "entity_name", "entity_domain", "entity_type")},
          "embedding": embedding, "extracted_facts": [], "importance": max((f["_importance"] for f in entity["facts"]), default=3)}
    op["reserved_id" if old is None else "db_id"] = entity["entity_id"]
    if old is not None:
        op["_expected_payload"] = deepcopy(old)
    op["search_entries"] = [{**{k: v for k, v in e.items() if k != "text"}, "embedding": embedding}
                            for e in changed_search_entries(entity, old)]
    return op
