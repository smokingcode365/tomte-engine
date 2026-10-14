"""Reference-oriented actions; these are not new extracted facts or identities."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, model_validator
from tomte.engine.core.resolution import ResolvedTarget


class ContextualMemoryAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_type: Literal["CORRECTION", "RETRACTION", "CONFIRMATION", "CLARIFICATION"]
    reference_hint: str = Field(min_length=1)
    replacement_text: str | None = None
    source_chat_ids: list[str] = Field(min_length=1)
    interpretation_confidence: float = Field(ge=0, le=1)
    target: ResolvedTarget | None = None
    target_fact_id: str | None = None

    @model_validator(mode="after")
    def validate_reference(self):
        if len(set(self.source_chat_ids)) != len(self.source_chat_ids):
            raise ValueError("Duplicate contextual sources")
        if (self.target is None) != (self.target_fact_id is None):
            raise ValueError("A resolved contextual action requires both entity and assertion")
        return self


class ContextualActionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    actions: list[ContextualMemoryAction] = Field(default_factory=list)
