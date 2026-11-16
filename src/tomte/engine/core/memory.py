"""Stored memory contracts. Business meaning belongs to the model, not these validators."""
from copy import deepcopy
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer, model_validator

from tomte.engine.core.enums import MemoryDomain

ISO_8601_REGEX = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
Confidence = Literal["high", "medium", "low"]


class MemoryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", serialize_by_alias=True)


def nonempty_text(value: str) -> str:
    if not value.strip() or "\x00" in value:
        raise ValueError("Text must be nonempty and contain no NUL")
    return value


def field_key(value: str) -> str:
    nonempty_text(value)
    if value.startswith("_"):
        raise ValueError("Value field names cannot use the internal metadata prefix")
    return value


def utc_stamp(value: str) -> str:
    datetime.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    return value


class ValueItem(MemoryModel):
    text: str
    context: str = Field(default="", description=(
        "Only the brief source-supported background needed to interpret text, such as who 'we' refers to. "
        "No guesses, duplicated claim, source IDs or unrelated story."))
    fact_confidence: Confidence = Field(description=(
        "How strongly the user's statements support the complete claim in text: high is directly supported, "
        "medium has limited support, low has weak support. Not verified truth or likelihood of a future event. "
        "A directly reported tentative intention can have high confidence; its tentativeness stays in text. "
        "Preserve the user's uncertainty in text; never store a model guess."))
    recorded_at: str = Field(alias="_recorded_at", pattern=ISO_8601_REGEX)
    updated_at: str = Field(alias="_updated_at", pattern=ISO_8601_REGEX)

    _text = field_validator("text")(nonempty_text)
    _timestamps = field_validator("recorded_at", "updated_at")(utc_stamp)

    @field_validator("context")
    @classmethod
    def valid_context(cls, value):
        if "\x00" in value:
            raise ValueError("Context cannot contain NUL")
        return value.strip()

    @model_serializer(mode="wrap")
    def omit_empty_context(self, handler):
        result = handler(self)
        if not self.context:
            result.pop("context", None)
        return result

    @model_validator(mode="after")
    def timestamp_order(self):
        if self.updated_at < self.recorded_at:
            raise ValueError("Item update precedes recording")
        return self


class HistoryValue(MemoryModel):
    archived_at: str = Field(alias="_archived_at", pattern=ISO_8601_REGEX)
    value: dict[str, ValueItem] = Field(min_length=1)

    _timestamp = field_validator("archived_at")(utc_stamp)

    @field_validator("value")
    @classmethod
    def keys(cls, value):
        for key in value:
            field_key(key)
        return value


class Fact(MemoryModel):
    fact_id: str
    attribute_name: str
    value: dict[str, ValueItem] = Field(min_length=1)
    history_values: list[HistoryValue] = Field(default_factory=list)
    importance: int = Field(default=3, alias="_importance", ge=1, le=5, strict=True)
    mention_count: int = Field(default=0, alias="_mention_count", ge=0, strict=True)
    last_mentioned_at: str = Field(alias="_last_mentioned_at", pattern=ISO_8601_REGEX)

    _attribute = field_validator("attribute_name")(nonempty_text)
    _timestamp = field_validator("last_mentioned_at")(utc_stamp)

    @field_validator("fact_id")
    @classmethod
    def uuid(cls, value):
        return str(UUID(value))

    @field_validator("value")
    @classmethod
    def keys(cls, value):
        for key in value:
            field_key(key)
        return value


class UnresolvedState(MemoryModel):
    """Storage lifecycle, not confidence in the user's reported statement."""
    original_domain: MemoryDomain
    reported_at: str
    status: Literal["pending", "resolved"] = "pending"
    resolved_at: str | None = None
    resolved_entity_ids: list[str] = Field(default_factory=list)

    _reported = field_validator("reported_at")(utc_stamp)

    @model_validator(mode="after")
    def lifecycle(self):
        if self.original_domain == MemoryDomain.UNRESOLVED:
            raise ValueError("Unresolved memory requires its original subject domain")
        self.resolved_entity_ids = [str(UUID(value)) for value in self.resolved_entity_ids]
        if len(set(self.resolved_entity_ids)) != len(self.resolved_entity_ids):
            raise ValueError("Duplicate unresolved resolution target")
        if self.status == "pending":
            if self.resolved_at is not None or self.resolved_entity_ids:
                raise ValueError("Pending memory cannot have a completed resolution")
        elif not self.resolved_at or not self.resolved_entity_ids:
            raise ValueError("Resolved memory requires its completion and target identities")
        if self.resolved_at is not None:
            utc_stamp(self.resolved_at)
        return self


class Entity(MemoryModel):
    entity_id: str = Field(description="The canonical UUID of this user's stored entity.")
    entity_name: str
    entity_domain: MemoryDomain
    entity_type: str
    first_mentioned_at: str | None = Field(default=None, pattern=ISO_8601_REGEX, description=(
        "For thoughts_and_plans, the earliest supporting user-message time available to Tomte, in UTC. "
        "Not the time the idea arose, processing time, or confirmation that it remains active."))
    aliases: list[str] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)
    unresolved_detail: str | None = None
    unresolved: UnresolvedState | None = Field(default=None, alias="_unresolved")

    _identity_text = field_validator("entity_name", "entity_type")(nonempty_text)

    @field_validator("first_mentioned_at")
    @classmethod
    def first_mention_time(cls, value):
        return utc_stamp(value) if value is not None else None

    @model_serializer(mode="wrap")
    def omit_absent_first_mention(self, handler):
        result = handler(self)
        if self.first_mentioned_at is None:
            result.pop("first_mentioned_at", None)
        if self.unresolved is None:
            result.pop("_unresolved", None)
            result.pop("unresolved_detail", None)
        return result

    @field_validator("entity_id")
    @classmethod
    def uuid(cls, value):
        return str(UUID(value))

    @field_validator("aliases")
    @classmethod
    def valid_aliases(cls, value):
        for alias in value:
            nonempty_text(alias)
        if len(set(value)) != len(value):
            raise ValueError("Duplicate aliases")
        return value

    @model_validator(mode="after")
    def fact_ids(self):
        if self.entity_domain == MemoryDomain.UNRESOLVED:
            if self.unresolved is None or not self.unresolved_detail or len(self.facts) != 1:
                raise ValueError("Unresolved record requires one claim, an explanation and lifecycle metadata")
            nonempty_text(self.unresolved_detail)
        elif self.unresolved is not None or self.unresolved_detail is not None:
            raise ValueError("Unresolved metadata belongs only to unresolved records")
        if self.entity_domain == MemoryDomain.THOUGHTS and self.first_mentioned_at is None:
            raise ValueError("Thoughts and plans require first_mentioned_at")
        ids = [fact.fact_id for fact in self.facts]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate fact UUID within entity")
        return self


class WireValueField(MemoryModel):
    """Fixed properties work with strict structured-output providers."""
    field_name: str
    text: str
    fact_confidence: Confidence = Field(description=ValueItem.model_fields["fact_confidence"].description)
    context: str | None = Field(default=None, description=(
        "Brief supported background needed to interpret this text. Null leaves existing context unchanged "
        "on UPDATE (none for a new item); empty string explicitly clears it. Recheck context when text changes."))

    _key = field_validator("field_name")(field_key)
    _text = field_validator("text")(nonempty_text)

    @field_validator("context")
    @classmethod
    def valid_context(cls, value):
        return ValueItem.valid_context(value) if value is not None else None


def wire_value(fields) -> dict:
    result = {}
    for raw in fields:
        item = WireValueField.model_validate(raw)
        value = {"text": item.text, "fact_confidence": item.fact_confidence}
        if item.context is not None:
            value["context"] = item.context
        if item.field_name in result and result[item.field_name] != value:
            raise ValueError("Conflicting duplicate value field")
        result[item.field_name] = value
    return result


def public_value(value: dict) -> dict:
    return {key: {"text": item["text"], "fact_confidence": item["fact_confidence"],
                  **({"context": item["context"]} if item.get("context") else {})}
            for key, item in value.items()}


def value_search_text(item: dict) -> str:
    """Plain supported content, without generated labels or internal metadata."""
    return item["text"] + ("\n" + item["context"] if item.get("context") else "")


def public_fact(fact: dict, history_indices=None) -> dict:
    """An explicit allowlist; future internal metadata cannot leak by default."""
    history = fact.get("history_values", [])
    indices = list(range(len(history))) if history_indices is None else list(history_indices)
    if len(indices) != len(set(indices)) or any(type(i) is not int or i < 0 or i >= len(history) for i in indices):
        raise ValueError("Invalid selected history indices")
    return {"fact_id": fact["fact_id"], "attribute_name": fact["attribute_name"],
            "value": public_value(fact["value"]),
            "history_values": [{"value": public_value(history[i]["value"])} for i in indices],
            "history_is_partial": bool(fact.get("history_is_partial", False)) or len(indices) < len(history)}


def public_entity(entity: dict, history_selection=None) -> dict:
    result = {key: deepcopy(entity[key]) for key in
              ("entity_id", "entity_name", "entity_domain", "entity_type", "aliases")}
    if entity["entity_domain"] == MemoryDomain.THOUGHTS:
        result["first_mentioned_at"] = entity["first_mentioned_at"]
    if entity["entity_domain"] == MemoryDomain.UNRESOLVED:
        result["unresolved_detail"] = entity["unresolved_detail"]
        result["reported_at"] = entity["_unresolved"]["reported_at"]
    result["facts"] = [public_fact(fact, None if history_selection is None else
                                  history_selection.get(fact["fact_id"], []))
                       for fact in entity["facts"]]
    return result
