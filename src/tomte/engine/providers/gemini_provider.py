from typing import List, Type, TypeVar, Optional
from pydantic import BaseModel
from pydantic_ai import Agent, Embedder
from pydantic_ai.models.google import GoogleModel
from tomte.engine.core.schema import TomteMemorySchema
from tomte.engine.llm.interfaces import LLMProvider
from tomte.engine.llm.structured import tomte_extractor_agent, junk_filter_agent, TomteDeps
from tomte.engine.core.config import settings

T = TypeVar('T', bound=BaseModel)

class GeminiLLMProvider(LLMProvider):
    def __init__(self, api_key: str):
        """
        Initializes the Gemini client via Pydantic-AI.
        The API key is injected by the host application[cite: 131].
        """
        self.api_key = api_key
        
        # Initialize Gemini Chat Models
        self.strong_model = GoogleModel(
            settings.gemini.strong_model, 
            api_key=api_key
        )
        self.cheap_model = GoogleModel(
            settings.gemini.cheap_model, 
            api_key=api_key
        )
        
        # Initialize Google Native Embedder
        self.embedder = Embedder(
            settings.gemini.embed_model, 
            api_key=api_key
        )

    def get_vector_dimension(self) -> int:
        """Returns dimensions from config to prevent pgvector mismatch."""
        return settings.gemini.vector_dimension

    def embed_text(self, text: str) -> List[float]:
        """Generates a single vector for live RAG searches[cite: 248]."""
        result = self.embedder.embed_query_sync(text)
        return list(result.embeddings[0])

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generates multiple vectors for the Consolidation Loop[cite: 81]."""
        if not texts:
            return []
        result = self.embedder.embed_documents_sync(texts)
        return [list(e) for e in result.embeddings]

    def is_junk(self, chat_log: str) -> bool:
        """Uses Gemini Flash to quickly filter out irrelevant chat logs [cite: 55-57]."""
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
                deps=deps
            )
            return result.data
            
        # Route B: Phase 3 Reconciliation or Generic Tasks (Dynamic Schema)
        else:
            # Spin up a temporary, stateless agent for this specific schema
            dynamic_agent = Agent(
                output_type=schema,
                system_prompt=system_prompt or "You are a helpful data extraction assistant."
            )
            result = dynamic_agent.run_sync(
                user_text,
                model=self.strong_model
            )
            return result.data