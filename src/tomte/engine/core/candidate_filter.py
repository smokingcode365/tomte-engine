"""Request-local candidate labels; no semantic identity rules or stored IDs."""
from collections import Counter

from pydantic import BaseModel, ConfigDict, Field

IDENTITY_CANDIDATE_LIMIT = 20


class CandidateFilterUnavailable(Exception):
    """An optional filter failed; its original candidates remain usable."""


class CandidateFilterResponse(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    keep: list[str] = Field(max_length=IDENTITY_CANDIDATE_LIMIT)
    drop: list[str] = Field(max_length=IDENTITY_CANDIDATE_LIMIT)


def retained_candidate_ids(labels, response):
    """Return mapped UUIDs only for a complete, exact partition, or None."""
    if not isinstance(response, CandidateFilterResponse):
        return None
    if Counter(response.keep + response.drop) != Counter(labels.keys()):
        return None
    return {labels[label] for label in response.keep}
