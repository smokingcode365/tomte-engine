"""Assertion identities, scoped semantic plans and deterministic lifecycle mechanics."""
from copy import deepcopy
from enum import Enum
import json
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel, ConfigDict, Field, model_validator
from tomte.engine.dictionary.terminology import canonical_attribute


class AssertionState(str, Enum):
    ACTIVE = "ACTIVE"
    HISTORICAL = "HISTORICAL"
    RETRACTED = "RETRACTED"
    CORRECTED_INVALID = "CORRECTED_INVALID"


class EvidenceType(str, Enum):
    USER_EXPLICIT = "USER_EXPLICIT"
    RESOLVED_REFERENCE = "RESOLVED_REFERENCE"
    GROUP_EXPLICIT = "GROUP_EXPLICIT"
    USER_IMPLIED = "USER_IMPLIED"
    DERIVED = "DERIVED"
    LEGACY = "LEGACY"


class FactEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source_chat_id: str = Field(min_length=1)
    evidence_type: EvidenceType
    interpretation_confidence: float = Field(ge=0, le=1)
    recorded_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    operation: Literal["ADD", "REINFORCE", "SUPERSEDE", "RETRACT", "CORRECT"] = "ADD"


class FactMutation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    operation: Literal["ADD", "REINFORCE", "SUPERSEDE", "RETRACT", "CORRECT"]
    target_fact_id: str | None = None
    attribute_name: str | None = None
    new_value: str | None = None
    source_chat_ids: list[str] = Field(min_length=1)
    interpretation_confidence: float = Field(ge=0, le=1)
    evidence_type: Literal["USER_EXPLICIT", "RESOLVED_REFERENCE", "GROUP_EXPLICIT", "USER_IMPLIED", "DERIVED"]

    @model_validator(mode="after")
    def validate_shape(self):
        if len(set(self.source_chat_ids)) != len(self.source_chat_ids):
            raise ValueError("Duplicate mutation sources")
        if self.operation == "ADD":
            if self.target_fact_id is not None or not self.attribute_name or not self.attribute_name.strip():
                raise ValueError("ADD needs an attribute and no target fact")
        elif not self.target_fact_id or not self.target_fact_id.strip():
            raise ValueError("Mutation requires an existing target fact")
        if self.operation in {"ADD", "SUPERSEDE"} and (self.new_value is None or not self.new_value.strip()):
            raise ValueError("Mutation requires a nonempty new value")
        if self.operation in {"REINFORCE", "RETRACT"} and self.new_value is not None:
            raise ValueError("Mutation does not accept a replacement value")
        if self.new_value is not None and not self.new_value.strip():
            raise ValueError("Empty replacement value")
        return self


class TargetMutationPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")
    target_key: str
    operations: list[FactMutation] = Field(default_factory=list)


class MutationPlanResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    plans: list[TargetMutationPlan]


def stable_id(*parts):
    return str(uuid5(NAMESPACE_URL, json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)))


def adapt_assertions(facts, entity_scope="legacy"):
    """Lazy pure adaptation: never random IDs, writes or fabricated missing provenance."""
    if not isinstance(facts, list) or any(not isinstance(f, dict) for f in facts):
        raise ValueError("Stored assertions must be a list of objects")
    result = []
    seen = set()
    for original in facts:
        fact = deepcopy(original)
        fact["attribute_name"] = canonical_attribute(fact["attribute_name"])
        if not fact.get("fact_id"):
            fact["fact_id"] = stable_id("tomte-legacy-assertion-v1", entity_scope,
                fact["attribute_name"], fact["value"], fact.get("source_chat_id"), fact.get("last_mentioned_at"))
        if fact["fact_id"] in seen:
            raise ValueError("Duplicate stored assertion ID")
        seen.add(fact["fact_id"])
        state = fact.get("assertion_state") or ("HISTORICAL" if fact.get("status") == "past" else "ACTIVE")
        fact["assertion_state"] = AssertionState(state).value
        fact["created_at"] = fact.get("created_at") or fact.get("last_mentioned_at")
        fact.setdefault("effective_at", None)
        fact.setdefault("ended_at", None)
        fact.setdefault("supersedes_fact_id", None)
        fact.setdefault("corrects_fact_id", None)
        fact["origin"] = fact.get("origin") or "LEGACY"
        if not fact.get("evidence"):
            source, stamp = fact.get("source_chat_id"), fact.get("last_mentioned_at")
            fact["evidence"] = []
            if source and source != "unknown" and stamp and stamp != "1970-01-01T00:00:00Z":
                fact["evidence"] = [FactEvidence(source_chat_id=source, recorded_at=stamp,
                    evidence_type="LEGACY", interpretation_confidence=fact.get("confidence", 0)).model_dump(mode="json")]
        result.append(fact)
    return result


def project_facts(facts, view="current"):
    if view not in {"current", "history", "audit"}:
        raise ValueError("Unknown assertion projection")
    def state(f):
        return f.get("assertion_state") or ("HISTORICAL" if f.get("status") == "past" else "ACTIVE")
    if view == "audit":
        return deepcopy(facts)
    allowed = {"ACTIVE"} if view == "current" else {"HISTORICAL", "RETRACTED"}
    return [deepcopy(f) for f in facts if state(f) in allowed]


def fact_values(facts):
    """Compact identity context without erasing simultaneously active values."""
    values = {}
    for fact in project_facts(facts):
        key = canonical_attribute(fact["attribute_name"])
        if key not in values:
            values[key] = fact["value"]
        elif fact["value"] != values[key]:
            if not isinstance(values[key], list):
                values[key] = [values[key]]
            if fact["value"] not in values[key]:
                values[key].append(fact["value"])
    return values


def _add_evidence(fact, operation, sources):
    existing = {e["source_chat_id"] for e in fact["evidence"]}
    added = 0
    for source in operation.source_chat_ids:
        item = FactEvidence(source_chat_id=source, evidence_type=operation.evidence_type,
            interpretation_confidence=operation.interpretation_confidence,
            recorded_at=sources[source], operation=operation.operation).model_dump(mode="json")
        if operation.operation in {"ADD", "REINFORCE"} and source in existing:
            continue
        if item not in fact["evidence"]:
            fact["evidence"].append(item)
            added += 1
    return added


def apply_mutations(facts, operations, chat_items, entity_scope, batch_id, templates=(), offered_fact_ids=None):
    """Validate and apply a whole entity plan on copies; failure never mutates inputs."""
    from tomte.engine.core.extraction import source_timestamps
    sources = source_timestamps(chat_items)
    result = adapt_assertions(facts, entity_scope)
    by_id = {f["fact_id"]: f for f in result}
    offered = set(by_id) if offered_fact_ids is None else set(offered_fact_ids)
    for raw in operations:
        op = FactMutation.model_validate(raw)
        if any(source not in sources for source in op.source_chat_ids):
            raise ValueError("Mutation evidence must reference current frozen batch sources")
        target = None
        if op.target_fact_id is not None:
            if op.target_fact_id not in offered or op.target_fact_id not in by_id:
                raise ValueError("Target fact does not belong to offered entity assertions")
            target = by_id[op.target_fact_id]
            if op.attribute_name and canonical_attribute(op.attribute_name) != target["attribute_name"]:
                raise ValueError("Mutation attribute hint disagrees with target assertion")
            if op.operation in {"REINFORCE", "SUPERSEDE", "RETRACT"} and target["assertion_state"] != "ACTIVE":
                raise ValueError("Mutation requires an ACTIVE assertion")
            if op.operation == "CORRECT" and target["assertion_state"] == "CORRECTED_INVALID":
                raise ValueError("Assertion already invalidated")
        if op.operation == "REINFORCE":
            added = _add_evidence(target, op, sources)
            if added:
                target["mention_count"] = target.get("mention_count", 1) + added
                target["last_mentioned_at"] = max(target["last_mentioned_at"], *(sources[s] for s in op.source_chat_ids))
                target["confidence"] = op.interpretation_confidence  # Latest evidence, no probability arithmetic.
                template = next((f for f in templates if canonical_attribute(f["attribute_name"]) == target["attribute_name"]
                                 and f["value"] == target["value"]), {})
                if "importance" in template:
                    target["importance"] = template["importance"]
            continue
        if op.operation == "SUPERSEDE" and op.new_value.strip().casefold() == str(target["value"]).strip().casefold():
            raise ValueError("SUPERSEDE requires a different assertion value")
        if target is not None:
            target["assertion_state"] = {"SUPERSEDE":"HISTORICAL", "RETRACT":"RETRACTED", "CORRECT":"CORRECTED_INVALID"}[op.operation]
            # Do not manufacture real-world ended_at from recording time.
            _add_evidence(target, op, sources)
        if op.operation == "RETRACT" or (op.operation == "CORRECT" and op.new_value is None):
            continue
        attribute = canonical_attribute(op.attribute_name) if op.operation == "ADD" else target["attribute_name"]
        if any(f["assertion_state"] == "ACTIVE" and f["attribute_name"] == attribute
               and str(f["value"]).strip().casefold() == op.new_value.strip().casefold() for f in result):
            raise ValueError("Duplicate active assertion; use REINFORCE")
        fid = stable_id("tomte-assertion-v1", entity_scope, batch_id, op.operation,
                        op.target_fact_id, attribute, op.new_value, sorted(op.source_chat_ids))
        if fid in by_id:
            raise ValueError("Repeated assertion creation in one plan")
        template = next((deepcopy(f) for f in templates if canonical_attribute(f["attribute_name"]) == attribute
                         and f["value"] == op.new_value), {})
        stamp = min(sources[s] for s in op.source_chat_ids)
        new = {"attribute_name":attribute, "value":op.new_value, "fact_id":fid,
               "assertion_state":"ACTIVE", "origin":op.evidence_type, "created_at":stamp,
               "effective_at":None, "ended_at":None,
               "supersedes_fact_id":op.target_fact_id if op.operation == "SUPERSEDE" else None,
               "corrects_fact_id":op.target_fact_id if op.operation == "CORRECT" else None,
               "evidence":[], "source_chat_id":min(op.source_chat_ids, key=lambda s:(sources[s],s)),
               "last_mentioned_at":max(sources[s] for s in op.source_chat_ids), "mention_count":len(op.source_chat_ids),
               "importance":template.get("importance",3), "confidence":template.get("confidence",op.interpretation_confidence),
               "status":template.get("status","active"), "sentiment":template.get("sentiment","neutral"),
               "volatility":template.get("volatility","core"), "related_entity_ids":template.get("related_entity_ids",[]),
               "timeline":template.get("timeline",[])}
        _add_evidence(new, op, sources)
        result.append(new)
        by_id[fid] = new
    return result
