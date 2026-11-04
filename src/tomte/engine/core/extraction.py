"""Strict extraction wire values and transient current-user provenance."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math

from pydantic import Field, create_model, model_validator
from tomte.engine.core.enums import MemoryDomain
from tomte.engine.core.memory import Entity, MemoryModel, WireValueField, wire_value
from tomte.engine.core.schema import DOMAIN_DESCRIPTIONS
from tomte.engine.core.proposals import ProposalResponse

RECENT_CONTEXT_TURNS = 4


class ExtractedFact(MemoryModel):
    attribute_name: str = Field(min_length=1, description="The category of this fact; reuse offered vocabulary when its meaning fits.")
    value: list[WireValueField] = Field(min_length=1, description="Related details about one fact. Business values are text, including vague dates and qualifications.")
    importance: int = Field(default=3, ge=1, le=5, strict=True)
    evidence_context: str | None = Field(default=None, max_length=600, description="Optional untrusted interpretation, supported by the cited current USER message.")
    group_participant_handles: list[str] = Field(default_factory=list, description="Only for explicit shared statements: handles of every emitted participant. No relationship edges.")
    source_chat_id: str | None = Field(default=None, description="Supporting CURRENT_BATCH USER item_id. Required with multiple messages.")

    @model_validator(mode="after")
    def fields(self):
        wire_value(self.value)
        self.group_participant_handles = list(dict.fromkeys(self.group_participant_handles))
        return self


class ExtractedEntity(MemoryModel):
    entity_id: str = Field(pattern=r"^tmp_[a-z][a-z0-9_]*$", description="Unique ephemeral object handle; never a stored identity.")
    entity_name: str = Field(min_length=1)
    entity_domain: MemoryDomain
    entity_type: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list, description="Names or noun phrases actually used for this entity. No invented aliases.")
    mention_context: str | None = Field(default=None, max_length=400)
    facts: list[ExtractedFact] = Field(default_factory=list)


class _ExtractionMemoryBoundary(ProposalResponse):
    proposal_collections = {name: "facts" for name in DOMAIN_DESCRIPTIONS}


ExtractionMemorySchema = create_model("ExtractionMemorySchema", __base__=_ExtractionMemoryBoundary,
    **{name: (list[ExtractedEntity], Field(default_factory=list, description=description))
       for name, description in DOMAIN_DESCRIPTIONS.items()})


class ResolvedExtractionFact(MemoryModel):
    attribute_name: str
    value: dict[str, dict[str, str]]
    importance: int
    evidence_context: str | None
    group_participant_handles: list[str]
    source_chat_id: str


class ResolvedExtractionEntity(ExtractedEntity):
    facts: list[ResolvedExtractionFact] = Field(default_factory=list)


ResolvedExtractionMemory = create_model("ResolvedExtractionMemory", __base__=MemoryModel,
    **{name: (list[ResolvedExtractionEntity], Field(default_factory=list)) for name in DOMAIN_DESCRIPTIONS})



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
    """Resolve transient sources; assistant messages cannot assert stored facts."""
    sources = source_timestamps(chat_items)
    user_sources = {item["item_id"] for item in chat_items if item["role"] == "user"}
    data = ExtractionMemorySchema.model_validate(extracted.model_dump(mode="json")).model_dump(mode="json")
    from collections import Counter
    counts = Counter(e["entity_id"] for entities in data.values() for e in entities)
    rejected_handles = {raw.get("entity_id") for path, raw in extracted.rejected_items()
                        if len(path) == 2 and isinstance(raw, dict) and isinstance(raw.get("entity_id"), str)}
    # Ambiguous handles cannot be renamed or resolved by guessing. Other entities
    # and independent facts can still proceed.
    for domain, entities in data.items():
        data[domain] = [e for e in entities if counts[e["entity_id"]] == 1
                        and e["entity_id"] not in rejected_handles and e["entity_domain"] == domain]
    handles = {e["entity_id"] for entities in data.values() for e in entities}
    for entities in data.values():
        for entity in entities:
            accepted = []
            for fact in entity["facts"]:
                source = fact["source_chat_id"]
                if source is None and len(sources) == 1:
                    source = next(iter(sources))
                if source not in user_sources or not set(fact["group_participant_handles"]) <= handles:
                    continue
                fact["source_chat_id"] = source
                fact["value"] = wire_value(fact["value"])
                accepted.append(fact)
            entity["facts"] = accepted
    return ResolvedExtractionMemory.model_validate(data)


def validate_operation_provenance(operations):
    """Validate the stored boundary. Source authorization remains transient."""
    for operation in operations:
        entity = Entity.model_validate(operation["payload"])
        canonical_id = str(operation.get("db_id") or operation.get("reserved_id"))
        if entity.entity_id != canonical_id or operation["entity_id"] != canonical_id:
            raise ValueError("Stored entity identity must equal its database UUID")
