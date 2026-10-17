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
    db_id: UUID = Field(description="The offered candidate's id UUID, never its entity_id or an extraction handle.")
    resolution_confidence: float = Field(ge=0, le=1)
    evidence: Evidence


class AliasEvidence(BaseModel):
    """Source-owned quotation; the identity model judges durable alias meaning."""
    model_config = ConfigDict(extra="forbid")
    alias: str = Field(min_length=1)
    source_chat_id: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=1)


class EntityResolution(BaseModel):
    model_config = ConfigDict(extra="forbid")
    extracted_entity_id: str = Field(description="Exact temporary id from EXTRACTED_SEMANTIC_INFORMATION; correlation only, never identity evidence.")
    kind: Literal["MATCH_ONE", "MATCH_MANY", "NEW", "UNRESOLVED"]
    targets: list[ResolvedTarget] = Field(default_factory=list)
    applies_equally_to_all: bool = Field(default=False, description="For MATCH_MANY, true only if all proposed facts apply equally to every selected target.")
    new_entity_evidence: str | None = None  # Verbatim CURRENT_BATCH introduction.
    explicitly_new: bool = Field(default=False, description="For NEW, explicitly set true only for a current USER introduction, and supply new_entity_evidence as a verbatim quote. False abstains.")
    reasoning: str
    accepted_aliases: list[str] = Field(
        default_factory=list,
        description="Stable durable aliases explicitly approved for this one resolved entity from offered current-user wording; empty means no alias authorization.",
    )

    alias_evidence: list[AliasEvidence] = Field(default_factory=list)
    group_source_chat_id: str | None = Field(default=None, description="CURRENT USER source explicitly applying this claim to the complete group.")
    group_evidence_quote: str | None = Field(default=None, description="Exact quote from group_source_chat_id establishing shared applicability. Context must identify every member; similarity/type alone is insufficient.")
    group_member_handles: list[str] = Field(default_factory=list, description="When extraction names participants, exact tmp_ handles whose single-target mappings equal the complete selected group.")

    @model_validator(mode="after")
    def check_cardinality(self):
        if any(not alias.strip() for alias in self.accepted_aliases) or len(set(self.accepted_aliases)) != len(self.accepted_aliases):
            raise ValueError("Accepted aliases must be nonblank and unique")
        if self.kind in {"MATCH_MANY", "UNRESOLVED"} and (self.accepted_aliases or self.alias_evidence):
            raise ValueError("Alias authorization requires one resolved entity")
        if len({t.db_id for t in self.targets}) != len(self.targets):
            raise ValueError("Duplicate resolution targets")
        if self.kind == "MATCH_ONE" and len(self.targets) != 1:
            raise ValueError("MATCH_ONE requires one target")
        if self.kind == "MATCH_MANY" and len(self.targets) < 2:
            raise ValueError("MATCH_MANY requires multiple targets")
        if self.kind == "NEW" and self.targets:
            raise ValueError("NEW cannot reference an existing target")
        if self.kind == "UNRESOLVED" and self.targets:
            raise ValueError("UNRESOLVED cannot authorize targets")
        return self


class ResolutionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    resolutions: list[EntityResolution]
