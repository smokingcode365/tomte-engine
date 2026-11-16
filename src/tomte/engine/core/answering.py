"""Transient model decisions for answering; never persisted as memory."""
from typing import Literal
from pydantic import Field, field_validator
from tomte.engine.core.memory import MemoryModel, nonempty_text

MAX_SEARCH_ROUNDS = 3
ANSWER_TIMEOUT_SECONDS = 120


def validate_memory_owner_name(value: str | None) -> str | None:
    """Validate optional caller context, not an identity or a database selector."""
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip() or len(value) > 256 or '\x00' in value:
        from tomte.engine.exceptions import TomteInvalidInputError
        raise TomteInvalidInputError('Memory owner name must be nonempty text of at most 256 characters')
    return value.strip()


class AnswerDecision(MemoryModel):
    action: Literal['answer', 'retrieve']
    answer: str = Field(description=(
        "Best supported answer so far, including any relevant uncertainty; required for either action."))
    suggestions: list[str] = Field(default_factory=list, description=(
        "Missing information that could improve the answer; empty when done."))

    _answer = field_validator('answer')(nonempty_text)


class RequiredInformation(MemoryModel):
    information_needed: list[str] = Field(default_factory=list, description=(
        "Short, specific information requirements; empty if no useful search remains."))


class AnswerFactSelection(MemoryModel):
    keep: list[str] = Field(description=(
        "All offered fact handles that support, qualify, contradict or might help answer the question. "
        "Do not omit a relevant instance to shorten the list. Empty is allowed."))


class CategorizedFact(MemoryModel):
    handle: str
    category: Literal['found', 'possible', 'related_only', 'unrelated']


class FactBatchJudgment(MemoryModel):
    kind: Literal['bounded', 'collection', 'ambiguous']
    found: list[int]
    possible: list[int]
    related: list[int]
    unrelated: list[int]
    enough: list[int]

    @field_validator('found', 'possible', 'related', 'unrelated', mode='before')
    @classmethod
    def preserve_valid_ids(cls, items):
        """Malformed IDs remain unknown; never shift positions or coerce strings."""
        if not isinstance(items, list):
            return []
        return [item for item in items if type(item) is int]

    @field_validator('enough', mode='before')
    @classmethod
    def invalid_citations_cannot_establish_satisfaction(cls, items):
        # An invalid citation blocks satisfaction without discarding valid facts.
        if not isinstance(items, list) or any(type(item) is not int for item in items):
            return [-1]
        return items

    @property
    def request_kind(self):
        return self.kind

    @property
    def satisfied(self):
        return bool(self.enough)

    @property
    def satisfied_by(self):
        return [f'F{i:03d}' for i in self.enough]

    @property
    def facts(self):
        # Duplicates (including cross-category duplicates) remain visible to the
        # reader's existing exact-handle validation and cannot select a fact.
        return [CategorizedFact(handle=f'F{i:03d}', category=category)
                for group, category in ((self.found, 'found'), (self.possible, 'possible'),
                                        (self.related, 'related_only'), (self.unrelated, 'unrelated'))
                for i in group]


class AnswerReaderUnavailable(RuntimeError):
    """The reader could not classify the offered evidence."""
