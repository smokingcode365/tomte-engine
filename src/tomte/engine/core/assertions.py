"""Authorized fact operations; no interpretation of human language in Python."""
from copy import deepcopy
from datetime import datetime, timezone
import json
from typing import Literal
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, model_validator
from tomte.engine.core.memory import Fact, MemoryModel, WireValueField, field_key, public_value, wire_value, utc_stamp
from tomte.engine.core.proposals import ProposalResponse


class FactMutation(MemoryModel):
    operation: Literal["ADD", "UPDATE", "MENTION", "APPEND_HISTORY"]
    target_fact_id: str | None = None
    attribute_name: str | None = None
    fields: list[WireValueField] = Field(default_factory=list)
    remove_fields: list[str] = Field(default_factory=list)
    update_mode: Literal["update_fields", "replace_value"] = Field(default="update_fields", description="UPDATE only: patch listed fields or replace the full value. Ignored for other operations.")
    change_type: Literal["real_change", "correction", "enrichment"] = Field(default="enrichment", description="UPDATE only: whether to archive a real-world change or update without archiving. Ignored for other operations.")
    history_index: int | None = Field(default=None, ge=0, strict=True, description=(
        "Only for APPEND_HISTORY enrichment: an offered index of the same historical episode. "
        "Append a fuller copy without changing existing items. Null means a distinct past episode."))
    importance: int | None = Field(default=None, ge=1, le=5, strict=True)
    incoming_fact_key: str | None = Field(default=None, description="Exact offered proposal key, required when acting on an incoming proposal, including group proposals.")
    source_chat_ids: list[str] = Field(min_length=1, description="Supporting CURRENT USER message IDs. Operational authorization only; never stored in memory.")

    @model_validator(mode="after")
    def shape(self):
        values = wire_value(self.fields)
        if any(not s.strip() for s in self.source_chat_ids):
            raise ValueError("Invalid mutation sources")
        self.source_chat_ids = list(dict.fromkeys(self.source_chat_ids))
        self.remove_fields = list(dict.fromkeys(self.remove_fields))
        for key in self.remove_fields:
            field_key(key)
        if set(values) & set(self.remove_fields):
            raise ValueError("Cannot set and remove the same field")
        if self.operation == "ADD":
            if self.target_fact_id is not None or not self.attribute_name or not self.attribute_name.strip() or not values:
                raise ValueError("ADD requires a new attribute/value, without a target")
        elif not self.target_fact_id:
            raise ValueError("Operation requires an offered fact UUID")
        if self.operation == "MENTION" and (values or self.remove_fields or self.importance is not None):
            raise ValueError("MENTION only changes fact mention metadata")
        if self.operation in {"ADD", "APPEND_HISTORY"} and (not values or self.remove_fields):
            raise ValueError("New values require fields and cannot remove fields")
        if self.operation == "UPDATE" and not (values or self.remove_fields):
            raise ValueError("UPDATE requires fields or removals")
        if self.history_index is not None and self.operation != "APPEND_HISTORY":
            raise ValueError("History cannot be targeted by current-value operations")
        if self.operation != "UPDATE":
            # These controls have no effect without an existing current-value update.
            self.update_mode = "update_fields"
            self.change_type = "enrichment"
        if self.update_mode == "replace_value" and (not values or self.remove_fields):
            raise ValueError("Replacement needs a nonempty complete value without removals")
        return self


class TargetMutationPlan(MemoryModel):
    target_key: str
    operations: list[FactMutation] = Field(default_factory=list)


class MutationPlanResponse(ProposalResponse):
    proposal_collections = {"plans": "operations"}
    plans: list[TargetMutationPlan]


class InvalidMutation(ValueError):
    """A proposed change is unusable; persisted-data failures are not this type."""


def stable_id(*parts):
    return str(uuid5(NAMESPACE_URL, json.dumps(parts, sort_keys=True, ensure_ascii=False, default=str)))


def validate_facts(facts):
    if not isinstance(facts, list):
        raise ValueError("Stored facts must be a list")
    result = [Fact.model_validate(f).model_dump(mode="json") for f in facts]
    if len({f["fact_id"] for f in result}) != len(result):
        raise ValueError("Duplicate stored fact UUID")
    return result


def fact_values(facts):
    """Keep separate facts even when they share an attribute name."""
    return [{"attribute_name": f["attribute_name"], "value": public_value(f["value"])} for f in facts]


def _updated_value(old, values, stamp):
    result = deepcopy(old)
    for key, item in values.items():
        previous = old.get(key)
        if previous is not None and public_value({key: previous})[key] == item:
            continue
        result[key] = {**item, "_recorded_at": previous["_recorded_at"] if previous else stamp,
                       "_updated_at": max(stamp, previous["_updated_at"]) if previous else stamp}
    return result


def apply_mutations(facts, operations, chat_items, entity_scope, batch_id, templates=(), offered_fact_ids=None,
                    *, recorded_at=None, offered_history=None, operation_indices=None):
    """Apply a whole entity plan on copies. Atomic batch markers handle replay.

    Source IDs are transient; deduplicate mentions within this plan. This function
    must be committed together with its processed-batch marker by the workflow.
    """
    from tomte.engine.core.extraction import source_timestamps
    sources = source_timestamps(chat_items)
    user_sources = {item["item_id"] for item in chat_items if item["role"] == "user"}
    stamp = utc_stamp(recorded_at or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    result = validate_facts(facts)
    by_id = {f["fact_id"]: f for f in result}
    offered = set(by_id) if offered_fact_ids is None else set(offered_fact_ids)
    original_history = {fid: deepcopy(f["history_values"]) for fid, f in by_id.items()}
    mentioned = {}
    indices = list(range(len(operations))) if operation_indices is None else list(operation_indices)
    if len(indices) != len(operations) or len(set(indices)) != len(indices) or any(type(i) is not int or i < 0 for i in indices):
        raise ValueError("Invalid original operation positions")
    for index, raw in zip(indices, operations):
        op = FactMutation.model_validate(raw)
        if not set(op.source_chat_ids) <= user_sources:
            raise InvalidMutation("Mutation evidence must reference CURRENT USER sources")
        target = by_id.get(op.target_fact_id)
        if op.target_fact_id is not None and (target is None or op.target_fact_id not in offered):
            raise InvalidMutation("Target fact does not belong to offered entity facts")
        if target is not None and op.attribute_name is not None and op.attribute_name != target["attribute_name"]:
            raise InvalidMutation("Mutation attribute disagrees with target fact")
        if op.incoming_fact_key is not None:
            matches = [f for f in templates if f.get("incoming_fact_key") == op.incoming_fact_key]
            if len(matches) != 1 or matches[0].get("source_chat_id") not in op.source_chat_ids:
                raise InvalidMutation("Mutation must select an offered current-user proposal")
        values = wire_value(op.fields)
        if op.operation == "ADD":
            fid = stable_id("tomte-fact-v2", entity_scope, batch_id, index)
            if fid in by_id:
                raise ValueError("Duplicate fact creation")
            target = {"fact_id": fid, "attribute_name": op.attribute_name,
                      "value": _updated_value({}, values, stamp), "history_values": [],
                      "_importance": op.importance or 3, "_mention_count": 0,
                      "_last_mentioned_at": max(sources[s] for s in op.source_chat_ids)}
            result.append(target)
            by_id[fid] = target
        elif op.operation == "UPDATE":
            old = target["value"]
            if any(key not in old for key in op.remove_fields):
                raise InvalidMutation("Cannot remove an absent field")
            updated = _updated_value(old, values, stamp)
            if op.update_mode == "replace_value":
                updated = {key: updated[key] for key in values}
            for key in op.remove_fields:
                del updated[key]
            if not updated:
                raise InvalidMutation("Current value cannot be empty")
            if public_value(old) != public_value(updated) and op.change_type == "real_change":
                target["history_values"].append({"_archived_at": stamp, "value": deepcopy(old)})
            target["value"] = updated
        elif op.operation == "APPEND_HISTORY":
            base = {}
            if op.history_index is not None:
                history = original_history[target["fact_id"]]
                allowed = set(range(len(history))) if offered_history is None else set(offered_history.get(target["fact_id"], []))
                if op.history_index not in allowed or op.history_index >= len(history):
                    raise InvalidMutation("History enrichment requires an offered history index")
                base = history[op.history_index]["value"]
                if any(key in base and item != public_value({key: base[key]})[key] for key, item in values.items()):
                    raise InvalidMutation("History enrichment cannot correct existing items")
            enriched = _updated_value(base, values, stamp)
            # Same-episode detail already present is only a mention. A distinct
            # episode may have identical values; Python does not merge episodes.
            if not base or public_value(base) != public_value(enriched):
                target["history_values"].append({"_archived_at": stamp, "value": enriched})
        if op.importance is not None:
            target["_importance"] = op.importance
        seen = mentioned.setdefault(target["fact_id"], set())
        target["_mention_count"] += len(set(op.source_chat_ids) - seen)
        target["_last_mentioned_at"] = max(target["_last_mentioned_at"], *(sources[s] for s in op.source_chat_ids))
        seen.update(op.source_chat_ids)
    return validate_facts(result)
