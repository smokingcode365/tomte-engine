"""Transient model decisions for answering; never persisted as memory."""
from typing import Literal
from pydantic import Field, field_validator
from tomte.engine.core.memory import MemoryModel, nonempty_text

MAX_SEARCH_ROUNDS = 3
MAX_SEARCHES_PER_ROUND = 3
ANSWER_TIMEOUT_SECONDS = 120


class AnswerSearchPlan(MemoryModel):
    searches: list[str] = Field(default_factory=list, description=(
        "One to three concise memory-search texts. Usually one is sufficient. "
        "Use only subjects known from the question, not guessed answers."))


class AnswerDecision(MemoryModel):
    action: Literal['answer', 'search']
    answer: str = Field(description=(
        "Always provide the best supported user-facing answer so far, in the question's language. "
        "If incomplete, explain what is unknown. This can be returned when the search budget ends."))
    missing_information: str = Field(default='', description=(
        "The specific evidence still needed; required to request another search."))
    searches: list[str] = Field(default_factory=list, description=(
        "Up to three new targeted search texts, only when action=search. Otherwise empty."))

    _answer = field_validator('answer')(nonempty_text)
