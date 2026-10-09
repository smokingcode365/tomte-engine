from typing import List, Optional, Type, TypeVar
from pydantic import BaseModel
from pydantic_ai import Agent, Embedder
from tomte.engine.core.schema import TomteMemorySchema
from tomte.engine.llm.interfaces import LLMProvider
from tomte.engine.llm.structured import tomte_extractor_agent, junk_filter_agent,TomteDeps
from tomte.engine.core.config import settings

T = TypeVar('T', bound=BaseModel)

class OpenAILLMProvider(LLMProvider):
    """Use Responses for structured output: low effort for extraction/reconciliation,
    none for the simple junk classifier. Embeddings retain their existing vector space.
    """
    def __init__(self, api_key: str):
        # Store the key; pydantic-ai will use it from env or explicit config
        self.api_key = api_key
        
        # We use Pydantic-AI's Embedder for all vector tasks
        self.embedder = Embedder(settings.openai.embed_model)
        
        # Explicit routing avoids Pydantic-AI inferring Chat Completions from bare names.
        self.strong_model = f"openai-responses:{settings.openai.strong_model}"
        self.cheap_model = f"openai-responses:{settings.openai.cheap_model}"

    def get_vector_dimension(self) -> int:
        return settings.openai.vector_dimension

    def embed_text(self, text: str) -> List[float]:
        """Synchronously generate a single embedding vector."""
        result = self.embedder.embed_query_sync(text)
        return list(result.embeddings[0]) #

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Synchronously generate multiple embedding vectors."""
        if not texts:
            return []
        result = self.embedder.embed_documents_sync(texts)
        return [list(e) for e in result.embeddings] #

    def is_junk(self, chat_log: str) -> bool:
        """Runs the junk filter agent with the cheap model."""
        result = junk_filter_agent.run_sync(
            chat_log, 
            model=self.cheap_model,
            model_settings={"openai_reasoning_effort": "none"}
        )
        return result.output


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
                model_settings={"openai_reasoning_effort": "low"},
                deps=deps
            )
            return result.output
            
        # Route B: Phase 3 Reconciliation or Generic Tasks (Dynamic Schema)
        else:
            # Spin up a temporary, stateless agent for this specific schema
            dynamic_agent = Agent(
                output_type=schema,
                system_prompt=system_prompt or "You are a helpful data extraction assistant."
            )
            result = dynamic_agent.run_sync(
                user_text,
                model=self.strong_model,
                model_settings={"openai_reasoning_effort": "low"}
            )
            return result.output