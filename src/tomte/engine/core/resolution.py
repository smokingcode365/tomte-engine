"""Target-set resolution contracts, deliberately separate from mutation classification."""
from typing import Literal
from uuid import UUID
from pydantic import BaseModel, ConfigDict, Field, model_validator

Evidence = Literal["EXPLICIT", "ALIAS", "RESOLVED_REFERENCE", "GROUP_EXPLICIT", "GROUP_INFERRED", "SEMANTIC_CANDIDATE"]
STRONG_EVIDENCE = {"EXPLICIT", "ALIAS", "RESOLVED_REFERENCE", "GROUP_EXPLICIT"}
MIN_RESOLUTION_CONFIDENCE = 0.9  # Policy threshold, not a calibrated probability.
CANDIDATE_LIMIT = 5


class ResolvedTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    db_id: UUID
    resolution_confidence: float = Field(ge=0, le=1)
    evidence: Evidence


class EntityResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    extracted_entity_id: str  # Correlation only, never evidence.
    kind: Literal["MATCH_ONE", "MATCH_MANY", "NEW", "UNRESOLVED"]
    targets: list[ResolvedTarget] = Field(default_factory=list)
    applies_equally_to_all: bool = False
    new_entity_evidence: str | None = None  # Verbatim CURRENT_BATCH introduction.
    explicitly_new: bool = False
    reasoning: str

    @model_validator(mode="after")
    def check_cardinality(self):
        if len({t.db_id for t in self.targets}) != len(self.targets):
            raise ValueError("Duplicate resolution targets")
        if self.kind == "MATCH_ONE" and len(self.targets) != 1:
            raise ValueError("MATCH_ONE requires one target")
        if self.kind == "MATCH_MANY" and len(self.targets) < 2:
            raise ValueError("MATCH_MANY requires multiple targets")
        if self.kind == "NEW" and self.targets:
            raise ValueError("NEW cannot reference an existing target")
        return self


class ResolutionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resolutions: list[EntityResolution]
