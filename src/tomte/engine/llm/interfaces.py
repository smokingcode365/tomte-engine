from typing import Protocol, List, TypeVar, Type, Optional
from pydantic import BaseModel

T = TypeVar('T', bound=BaseModel)

class LLMProvider(Protocol):
    """The contract for the Tomte Engine's cognitive operations."""

    def get_vector_dimension(self) -> int:
        """Returns the size of the vectors this model produces (e.g., 1536)."""
        ...

    def embed_text(self, text: str) -> List[float]:
        """Generates a single vector for live RAG searches."""
        ...

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates multiple vectors in one API call for the Consolidation Loop."""
        ...

    def is_junk(self, chat_log: str) -> bool:
        """
        Uses a cheap, fast model to determine if processing should continue.
        Goal: Cost Control.
        """
        ...

    def parse_structured(
        self, 
        system_prompt: str, 
        user_text: str, 
        schema: Type[T],
        taxonomy: Optional[List[str]] = None,
        lexicon: Optional[List[str]] = None 
    ) -> T:
        """
        Uses a high-intelligence model to perform deterministic extraction.
        Must return an instantiated Pydantic model (TomteMemorySchema).
        """
        ...