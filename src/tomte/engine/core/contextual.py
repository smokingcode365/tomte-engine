"""Reference routing does not itself authorize changes to fact values."""
from typing import Literal
from pydantic import Field, model_validator
from tomte.engine.core.memory import MemoryModel
from tomte.engine.core.resolution import ResolvedTarget
from tomte.engine.core.proposals import ProposalResponse


class ContextualMemoryAction(MemoryModel):
    action_type: Literal["CORRECTION", "END", "CONFIRMATION", "CLARIFICATION", "MENTION", "HISTORY"]
    reference_hint: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=1, description="Verbatim current USER evidence, not a normalized replacement value.")
    source_chat_ids: list[str] = Field(min_length=1)
    target: ResolvedTarget | None = None
    target_fact_id: str | None = None

    @model_validator(mode="after")
    def validate_reference(self):
        self.source_chat_ids = list(dict.fromkeys(self.source_chat_ids))
        if (self.target is None) != (self.target_fact_id is None):
            raise ValueError("A resolved contextual action requires both entity and fact")
        return self


class ContextualActionResponse(ProposalResponse):
    proposal_collections = {"actions": None}
    actions: list[ContextualMemoryAction] = Field(default_factory=list)
