"""Bounded extraction input and deterministic provenance at the workflow boundary."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math
import re
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from tomte.engine.core.schema import Entity, Fact, TimelineEvent, TomteMemorySchema, ISO_8601_REGEX

RECENT_CONTEXT_TURNS = 4


class ExtractedTimelineEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_description: str = deepcopy(TimelineEvent.model_fields["event_description"])
    source_chat_id: Optional[str] = Field(default=None, description="Supporting CURRENT_BATCH item_id; required with multiple messages.")


def _fields(model, excluded=()):
    return {name: (field.annotation, deepcopy(field))
            for name, field in model.model_fields.items() if name not in excluded}


# Reuse semantic field definitions, but never request system timestamps from a model.
ExtractedFact = create_model(
    "ExtractedFact", __config__=ConfigDict(extra="forbid"),
    **_fields(Fact, ("source_chat_id", "last_mentioned_at", "timeline", "fact_id", "assertion_state", "origin", "created_at", "effective_at", "ended_at", "supersedes_fact_id", "corrects_fact_id", "evidence", "related_entity_uuids", "related_entity_ids")),
    related_entity_ids=(List[str], Field(default_factory=list, description="Structured relationship endpoints: exact tmp_ handles of objects in this response. No database UUIDs, names or guesses. Empty for a genuinely standalone fact.")),
    evidence_context=(Optional[str], Field(default=None, max_length=600, description="Optional untrusted interpretation of referents/time/negation. Supported by this fact's CURRENT USER source; not additional evidence.")),
    group_participant_handles=(List[str], Field(default_factory=list, description="Exact tmp_ handles of all known participants for an explicitly shared statement; not relationship endpoints.")),
    source_chat_id=(Optional[str], Field(default=None, description="Supporting CURRENT_BATCH item_id; required with multiple messages.")),
    timeline=(List[ExtractedTimelineEvent], deepcopy(Fact.model_fields["timeline"])),
)
ExtractedEntity = create_model(
    "ExtractedEntity", __config__=ConfigDict(extra="forbid"),
    **_fields(Entity, ("facts", "entity_id")),
    entity_id=(str, Field(pattern=r"^tmp_[a-z][a-z0-9_]*$", description="Ephemeral tmp_ object handle, unique across all eight lists, e.g. tmp_per_1. Never a stored identity. Related IDs reference these handles only.")),
    mention_context=(Optional[str], Field(default=None, max_length=400, description="Optional untrusted reference interpretation, grounded in the cited current user facts. Never a persistent summary.")),
    facts=(List[ExtractedFact], deepcopy(Entity.model_fields["facts"])),
)


class _ExtractionMemoryBoundary(TomteMemorySchema):
    @model_validator(mode="after")
    def unique_object_handles(self):
        ids = [entity.entity_id for name in TomteMemorySchema.model_fields
               for entity in getattr(self, name)]
        if len(set(ids)) != len(ids):
            raise ValueError("Temporary extraction IDs must be unique within a batch")
        return self


ExtractionMemorySchema = create_model(
    "ExtractionMemorySchema", __base__=_ExtractionMemoryBoundary,
    **{name: (List[ExtractedEntity], deepcopy(field))
       for name, field in TomteMemorySchema.model_fields.items()},
)

ResolvedExtractionFact = create_model("ResolvedExtractionFact", __base__=Fact,
    evidence_context=(Optional[str], None), group_participant_handles=(List[str], Field(default_factory=list)))
ResolvedExtractionEntity = create_model("ResolvedExtractionEntity", __base__=Entity,
    mention_context=(Optional[str], None), facts=(List[ResolvedExtractionFact], Field(default_factory=list)))
ResolvedExtractionMemory = create_model("ResolvedExtractionMemory", __base__=TomteMemorySchema,
    **{name: (List[ResolvedExtractionEntity], deepcopy(field)) for name, field in TomteMemorySchema.model_fields.items()})


def source_timestamps(chat_items):
    """Validate source metadata before model work or persistence; UTC second precision."""
    sources = {}
    for item in chat_items:
        source = item.get("item_id")
        stamp = item.get("timestamp")
        if not isinstance(source, str) or not source or source == "unknown" or source in sources:
            raise ValueError("Current batch requires unique non-placeholder source IDs")
        if isinstance(stamp, bool) or not isinstance(stamp, (int, float)) or not math.isfinite(stamp) or stamp <= 0:
            raise ValueError("Current batch requires valid source timestamps")
        try:
            timestamp = datetime.fromtimestamp(stamp, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, OverflowError, OSError) as exc:
            raise ValueError("Current batch source timestamp is out of range") from exc
        if timestamp == "1970-01-01T00:00:00Z":
            raise ValueError("Epoch placeholder timestamps are forbidden")
        sources[source] = timestamp
    if not sources:
        raise ValueError("Extraction requires current source messages")
    return sources


def build_extraction_input(chat_items):
    source_timestamps(chat_items)
    # The first queued message carries the history preceding the entire frozen batch.
    recent = chat_items[0].get("recent_context", [])[-RECENT_CONTEXT_TURNS:]
    return json.dumps({
        "CURRENT_BATCH": [{"item_id": item["item_id"], "role": item["role"], "content": item["content"], "timestamp": item["timestamp"]}
                          for item in chat_items],
        "RECENT_CONTEXT_REFERENCE_ONLY": [{"role": item["role"], "content": item["content"]} for item in recent],
    }, ensure_ascii=False)


def resolve_extraction(extracted, chat_items):
    """Validate model-selected source IDs and supply all deterministic metadata."""
    sources = source_timestamps(chat_items)
    user_sources = {item["item_id"] for item in chat_items if item["role"] == "user"}
    data = ExtractionMemorySchema.model_validate(extracted.model_dump(mode="json")).model_dump(mode="json")
    def resolve(source):
        if source is None and len(sources) == 1:
            source = next(iter(sources))
        if source not in sources:
            raise ValueError("Extracted provenance must reference a CURRENT_BATCH item_id")
        return source, sources[source]
    for entities in data.values():
        for entity in entities:
            user_facts = []
            for fact in entity["facts"]:
                fact["source_chat_id"], fact["last_mentioned_at"] = resolve(fact["source_chat_id"])
                user_timeline = []
                for event in fact["timeline"]:
                    event["source_chat_id"], event["timestamp"] = resolve(event["source_chat_id"])
                    if event["source_chat_id"] in user_sources:
                        user_timeline.append(event)
                fact["timeline"] = user_timeline
                if fact["source_chat_id"] in user_sources:
                    user_facts.append(fact)
            entity["facts"] = user_facts
    # Preserve ephemeral context while adding deterministic timestamps. These
    # internal models are not the extraction wire or the persistent schema.
    return ResolvedExtractionMemory.model_validate(data)


def validate_operation_provenance(operations):
    """Do not rewrite inherited placeholder metadata back into merged payloads."""
    for operation in operations:
        for fact in operation["payload"].get("facts", []):
            records = [(fact, "last_mentioned_at"),
                       *((event, "timestamp") for event in fact.get("timeline", []))]
            for record, timestamp_key in records:
                source = record.get("source_chat_id")
                timestamp = record.get(timestamp_key)
                if (not isinstance(source, str) or not source.strip() or source == "unknown" or
                        not isinstance(timestamp, str) or not re.fullmatch(ISO_8601_REGEX, timestamp) or
                        timestamp == "1970-01-01T00:00:00Z"):
                    raise ValueError("Cannot persist missing or placeholder provenance")

            for evidence in fact.get("evidence", []):
                from tomte.engine.core.assertions import FactEvidence
                item = FactEvidence.model_validate(evidence)
                if item.source_chat_id == "unknown" or item.recorded_at == "1970-01-01T00:00:00Z":
                    raise ValueError("Cannot persist placeholder assertion evidence")
