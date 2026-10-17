from typing import List, Type, TypeVar, Optional
from pydantic import BaseModel
from pydantic_ai import Agent, Embedder
from pydantic_ai.embeddings.voyageai import VoyageAIEmbeddingModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.voyageai import VoyageAIProvider
from tomte.engine.core.schema import TomteMemorySchema
from tomte.engine.llm.interfaces import LLMProvider
from tomte.engine.llm.structured import tomte_extractor_agent, junk_filter_agent, TomteDeps
from tomte.engine.core.config import settings
from tomte.engine.providers.diagnostics import observe_provider, observe_result
from tomte.engine.llm.trust import TRUST_BOUNDARY

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
        text_provider = AnthropicProvider(api_key=anthropic_api_key)
        self.strong_model = AnthropicModel(
            settings.anthropic.strong_model, 
            provider=text_provider
        )
        self.cheap_model = AnthropicModel(
            settings.anthropic.cheap_model, 
            provider=text_provider
        )
        
        # Voyage AI is used for high-fidelity embeddings as per Anthropic best practices
        # Model identifier e.g., 'voyage-large-2'
        self.embedder = Embedder(VoyageAIEmbeddingModel(
            settings.anthropic.embed_model,
            provider=VoyageAIProvider(api_key=voyage_api_key)
        ))

    def get_vector_dimension(self) -> int:
        """Returns dimensions from config (e.g., 1536) to ensure pgvector alignment."""
        return settings.anthropic.vector_dimension

    @observe_provider("embedder")
    def embed_text(self, text: str) -> List[float]:
        """Generates a single vector for live RAG searches."""
        result = self.embedder.embed_query_sync(text)
        observe_result(result)
        return list(result.embeddings[0])

    @observe_provider("embedder")
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates multiple vectors for the Consolidation Loop (Workflow 3)."""
        if not texts:
            return []
        result = self.embedder.embed_documents_sync(texts)
        observe_result(result)
        return [list(e) for e in result.embeddings]

    @observe_provider("cheap_model")
    def is_junk(self, chat_log: str) -> bool:
        """
        Uses Claude Haiku to determine if the log contains useful facts.
        Prevents expensive processing of small talk [cite: 55-57].
        """
        result = junk_filter_agent.run_sync(
            chat_log, 
            model=self.cheap_model
        )
        return result.output

    @observe_provider("strong_model")
    def parse_structured(
        self, 
        system_prompt: str, 
        user_text: str, 
        schema: Type[T],
        taxonomy: Optional[List[str]] = None,
        lexicon: Optional[List[str]] = None
    ) -> T:
        """
        Dynamically routes the extraction based on the requested schema.
        """
        # Route A: Phase 2 Generative Extraction (Requires Dictionary Deps)
        if isinstance(schema, type) and issubclass(schema, TomteMemorySchema):
            deps = TomteDeps(taxonomy=taxonomy or [], lexicon=lexicon or [])
            result = tomte_extractor_agent.run_sync(
                user_text,
                model=self.strong_model,
                output_type=schema,
                deps=deps
            )
            return result.output
            
        # Route B: Phase 3 Reconciliation or Generic Tasks (Dynamic Schema)
        else:
            # Spin up a temporary, stateless agent for this specific schema
            dynamic_agent = Agent(
                output_type=schema,
                system_prompt=TRUST_BOUNDARY + (system_prompt or "You are a helpful data extraction assistant.")
            )
            result = dynamic_agent.run_sync(
                user_text,
                model=self.strong_model
            )
            return result.output
