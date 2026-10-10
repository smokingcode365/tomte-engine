import json
from typing import List, Optional, Type, TypeVar, get_args
from pydantic_core import ErrorType
from pydantic_ai.messages import ModelResponse, RetryPromptPart
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent, Embedder, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior
from tomte.engine.core.schema import TomteMemorySchema
from tomte.engine.llm.interfaces import LLMProvider
from tomte.engine.llm.structured import tomte_extractor_agent, junk_filter_agent,TomteDeps
from tomte.engine.core.config import settings
from tomte.engine.log import tomtelog

T = TypeVar('T', bound=BaseModel)

def _debug_metadata(metadata):
    # Optional diagnostics must not turn a successful run into a failure or
    # replace the original model exception when a custom logger fails.
    try:
        tomtelog.debug(json.dumps(metadata, sort_keys=True))
    except Exception:
        pass


def _validation_metadata(content, schema):
    """Keep only validation codes and schema-declared locations, never input/message/context."""
    if not isinstance(content, list):
        return []
    definition = schema.model_json_schema() if isinstance(schema, type) and issubclass(schema, BaseModel) else {}
    fields = {name for node in [definition, *definition.get("$defs", {}).values()]
              for name in node.get("properties", {})}
    return [{"type": error.get("type") if error.get("type") in get_args(ErrorType) else "custom_error",
             "location": [value if isinstance(value, int) or value in fields else "<field>"
                          for value in error.get("loc", ())]} for error in content]


def _log_structured_retries(result, schema, stage):
    """Report native retry metadata for completed runs, never prompts or error inputs."""
    messages = getattr(result, "all_messages", lambda: [])()
    retries = []
    attempt = 0
    for message in messages:
        if isinstance(message, ModelResponse):
            attempt += 1
        for part in message.parts:
            if isinstance(part, RetryPromptPart):
                retries.append((attempt, part))
    for index, (attempt, part) in enumerate(retries, 1):
        _debug_metadata({"event": "structured_output_retry", "stage": stage,
            "schema": getattr(schema, "__name__", "structured"), "attempt": attempt,
            "retry_index": index, "retry_count": len(retries),
            "reason": "pydantic_validation" if isinstance(part.content, list) else "agent_retry",
            "errors": _validation_metadata(part.content, schema)})


def _run_structured_agent(agent, schema, stage, user_text, **kwargs):
    """Use native capture on failure; propagate the original exception and retry policy."""
    with capture_run_messages() as messages:
        # Native nested capture reuses the first run's history. Never count that
        # history as this call's attempts if an enclosing capture has already run.
        owns_history = not messages
        try:
            return agent.run_sync(user_text, **kwargs)
        except UnexpectedModelBehavior as exc:
            cause = exc
            while cause is not None and not isinstance(cause, ValidationError):
                cause = cause.__cause__
            errors = cause.errors(include_url=False) if isinstance(cause, ValidationError) else None
            _debug_metadata({"event": "structured_output_failure", "stage": stage,
                "schema": getattr(schema, "__name__", "structured"),
                "attempt": sum(isinstance(m, ModelResponse) for m in messages) if owns_history else None,
                "retry_count": sum(isinstance(p, RetryPromptPart) for m in messages for p in m.parts)
                               if owns_history else None,
                "reason": "pydantic_validation" if errors is not None else "agent_failure",
                "errors": _validation_metadata(errors, schema)})
            raise


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
            result = _run_structured_agent(
                tomte_extractor_agent, schema, "extraction", user_text,
                model=self.strong_model,
                output_type=schema,
                model_settings={"openai_reasoning_effort": "low"},
                deps=deps
            )
            _log_structured_retries(result, schema, "extraction")
            return result.output
            
        # Route B: Phase 3 Reconciliation or Generic Tasks (Dynamic Schema)
        else:
            # Spin up a temporary, stateless agent for this specific schema
            dynamic_agent = Agent(
                output_type=schema,
                system_prompt=system_prompt or "You are a helpful data extraction assistant."
            )
            stage = {"MutationPlanResponse": "fact_mutation",
                     "TerminologyAlignmentResponse": "terminology_alignment"}.get(
                         getattr(schema, "__name__", ""), "structured")
            result = _run_structured_agent(
                dynamic_agent, schema, stage, user_text,
                model=self.cheap_model if stage == "terminology_alignment" else self.strong_model,
                model_settings={"openai_reasoning_effort": "low"}
            )
            _log_structured_retries(result, schema, stage)
            return result.output