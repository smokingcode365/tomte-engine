"""Navigation contracts, independent of answering and evidence search."""
from dataclasses import dataclass, field

from pydantic import Field
from tomte.engine.core.memory import MemoryModel


class CatalogueSelection(MemoryModel):
    handles: list[str] = Field(default_factory=list, description=(
        "Offered handles to select; empty is allowed."))


@dataclass
class CataloguePaths:
    paths: list[dict] = field(default_factory=list)
    coverage: list[dict] = field(default_factory=list)
