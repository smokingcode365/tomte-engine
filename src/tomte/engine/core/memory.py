"""Stored memory contracts. Business meaning belongs to the model, not these validators."""
from copy import deepcopy
from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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
    fact_confidence: Confidence = Field(description=(
        "How strongly the user's statements support this item: high is directly supported, "
        "medium is qualified, low is explicitly uncertain. Not verified truth or a probability. "
        "Preserve the user's uncertainty in text; never store a model guess."))
    recorded_at: str = Field(alias="_recorded_at", pattern=ISO_8601_REGEX)
    updated_at: str = Field(alias="_updated_at", pattern=ISO_8601_REGEX)

    _text = field_validator("text")(nonempty_text)
    _timestamps = field_validator("recorded_at", "updated_at")(utc_stamp)

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


class Entity(MemoryModel):
    entity_id: str = Field(description="The canonical UUID of this user's stored entity.")
    entity_name: str
    entity_domain: MemoryDomain
    entity_type: str
    aliases: list[str] = Field(default_factory=list)
    facts: list[Fact] = Field(default_factory=list)

    _identity_text = field_validator("entity_name", "entity_type")(nonempty_text)

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
        ids = [fact.fact_id for fact in self.facts]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate fact UUID within entity")
        return self


class WireValueField(MemoryModel):
    """Fixed properties work with strict structured-output providers."""
    field_name: str
    text: str
    fact_confidence: Confidence = Field(description=ValueItem.model_fields["fact_confidence"].description)

    _key = field_validator("field_name")(field_key)
    _text = field_validator("text")(nonempty_text)


def wire_value(fields) -> dict:
    result = {}
    for raw in fields:
        item = WireValueField.model_validate(raw)
        value = {"text": item.text, "fact_confidence": item.fact_confidence}
        if item.field_name in result and result[item.field_name] != value:
            raise ValueError("Conflicting duplicate value field")
        result[item.field_name] = value
    return result


def public_value(value: dict) -> dict:
    return {key: {"text": item["text"], "fact_confidence": item["fact_confidence"]}
            for key, item in value.items()}


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
    result["facts"] = [public_fact(fact, None if history_selection is None else
                                  history_selection.get(fact["fact_id"], []))
                       for fact in entity["facts"]]
    return result
