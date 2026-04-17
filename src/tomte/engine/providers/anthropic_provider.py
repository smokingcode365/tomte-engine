from typing import List, Type, TypeVar, Optional
from pydantic import BaseModel
from pydantic_ai import Embedder
from pydantic_ai.models.anthropic import AnthropicChatModel
from tomte.engine.llm.interfaces import LLMProvider
from tomte.engine.llm.structured import tomte_extractor_agent, junk_filter_agent, TomteDeps
from tomte.engine.core.config import settings

T = TypeVar('T', bound=BaseModel)

class AnthropicLLMProvider(LLMProvider):
    def __init__(
        self, 
        anthropic_api_key: str, 
        voyage_api_key: str
    ):
        """
        Initializes the Anthropic text client and Voyage AI embedding client.
        Secrets are handled by the host application and injected here.
        """
        # Pydantic-AI handles the underlying Anthropic SDK client
        self.strong_model = AnthropicChatModel(
            settings.anthropic.strong_model, 
            api_key=anthropic_api_key
        )
        self.cheap_model = AnthropicChatModel(
            settings.anthropic.cheap_model, 
            api_key=anthropic_api_key
        )
        
        # Voyage AI is used for high-fidelity embeddings as per Anthropic best practices
        # Model identifier e.g., 'voyage-large-2'
        self.embedder = Embedder(
            settings.anthropic.embed_model, 
            api_key=voyage_api_key
        )

    def get_vector_dimension(self) -> int:
        """Returns dimensions from config (e.g., 1536) to ensure pgvector alignment."""
        return settings.anthropic.vector_dimension

    def embed_text(self, text: str) -> List[float]:
        """Generates a single vector for live RAG searches."""
        result = self.embedder.embed_query_sync(text)
        return list(result.embeddings[0])

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates multiple vectors for the Consolidation Loop (Workflow 3)."""
        if not texts:
            return []
        result = self.embedder.embed_documents_sync(texts)
        return [list(e) for e in result.embeddings]

    def is_junk(self, chat_log: str) -> bool:
        """
        Uses Claude Haiku to determine if the log contains useful facts.
        Prevents expensive processing of small talk [cite: 55-57].
        """
        result = junk_filter_agent.run_sync(
            chat_log, 
            model=self.cheap_model
        )
        return result.data

    def parse_structured(
        self, 
        system_prompt: str, 
        user_text: str, 
        schema: Type[T],
        taxonomy: List[str] = [],
        lexicon: List[str] = []
    ) -> T:
        """
        Deterministic extraction into the 8-domain Knowledge Graph [cite: 398-411].
        Injects the Tenant Dictionary for ontology alignment [cite: 106-109, 297].
        """
        # Package JIT dictionaries for the Agent's dynamic system prompt
        deps = TomteDeps(taxonomy=taxonomy, lexicon=lexicon)
        
        result = tomte_extractor_agent.run_sync(
            user_text,
            model=self.strong_model,
            deps=deps,
            # Note: system_prompt override can be passed if specific overrides are needed,
            # but our agent uses the template in structured.py + EXTRACTOR_SYSTEM_PROMPT.
        )
        return result.data