import json
import os
from typing import List, Optional, Type, TypeVar, get_args
from pydantic_core import ErrorType
from pydantic_ai.messages import ModelResponse, RetryPromptPart
from pydantic import BaseModel, ValidationError
from pydantic_ai import Agent, Embedder, ToolOutput, capture_run_messages
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.exceptions import ModelHTTPError
from openai import OpenAIError
from httpx import TransportError
from tomte.engine.core.candidate_filter import CandidateFilterResponse, CandidateFilterUnavailable
from tomte.engine.core.answering import AnswerFactSelection, AnswerReaderUnavailable, FactBatchJudgment
from tomte.engine.core.extraction import ExtractionMemorySchema
from tomte.engine.core.assertions import MutationPlanResponse
from tomte.engine.llm.interfaces import LLMProvider
from tomte.engine.llm.structured import tomte_extractor_agent, junk_filter_agent,TomteDeps
from tomte.engine.core.config import settings
from tomte.engine.log import tomtelog
from tomte.engine.providers.diagnostics import observe_provider, observe_result
from tomte.engine.llm.trust import TRUST_BOUNDARY
from tomte.engine.llm.structured import make_extractor
from tomte.engine.llm.recovery import install_recovery, recoverable, run_with_recovery
from tomte.engine.llm.embedding_identity import openai_embedding_identity

T = TypeVar('T', bound=BaseModel)

def _fit_embedding_documents(texts, model):
    """Bound supported OpenAI document inputs by tokens, never by byte length.

    This changes derived embedding input only. Queries and stored values are
    untouched; current/history field chunks already fit the model's token limit.
    Unknown compatible deployments retain their own provider limit behavior.
    """
    name = model.removeprefix('openai:') if isinstance(model, str) else model.model_name
    if name not in ('text-embedding-3-large', 'text-embedding-3-small', 'text-embedding-ada-002'):
        return texts
    # OpenAI embeddings API, verified 2026-11-09: 8,192 tokens per input.
    limit = 8192
    # UTF-8 bytes upper-bound ordinary BPE token counts. Avoid loading tokenizer
    # assets for already-small inputs; this is a fast path, not a truncation cap.
    if all(len(text.encode('utf-8')) <= limit for text in texts):
        return texts
    import tiktoken
    encoding = tiktoken.encoding_for_model(name)
    result = []
    for text in texts:
        tokens = encoding.encode_ordinary(text)  # Literal special-token strings are data.
        while len(tokens) > limit:
            text = encoding.decode_bytes(tokens[:limit]).decode('utf-8', errors='ignore')
            # A token boundary can split a Unicode character. Recount the valid
            # prefix to enforce the actual input budget after decoding.
            tokens = encoding.encode_ordinary(text)
        result.append(text)
    return result


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
            return run_with_recovery(agent, schema, user_text, **kwargs)
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
    none for the simple junk classifier. Embedding dimensions are explicitly configured.
    """
    supports_candidate_filter = True

    def __init__(self, api_key: str):
        # Store the key; pydantic-ai will use it from env or explicit config
        self.api_key = api_key
        
        # We use Pydantic-AI's Embedder for all vector tasks
        self.embedder = Embedder(settings.openai.embed_model,
                                 settings={"dimensions": settings.openai.vector_dimension})
        self._embedding_cache_identity = openai_embedding_identity(
            settings.openai.embed_model, settings.openai.vector_dimension,
            os.environ.get('OPENAI_BASE_URL', 'https://api.openai.com/v1'))
        
        # Explicit routing avoids Pydantic-AI inferring Chat Completions from bare names.
        self.strong_model = f"openai-responses:{settings.openai.strong_model}"
        self.cheap_model = f"openai-responses:{settings.openai.cheap_model}"
        self.candidate_filter_model = f"openai-responses:{settings.openai.candidate_filter_model}"
        self.answer_reader_model = f"openai-responses:{settings.openai.answer_reader_model}"
        self.retrieval_planner_model = f"openai-responses:{settings.openai.retrieval_planner_model}"

    def get_vector_dimension(self) -> int:
        return settings.openai.vector_dimension

    def get_embedding_cache_identity(self) -> str | None:
        return getattr(self, '_embedding_cache_identity', None)

    @observe_provider("embedder")
    def embed_text(self, text: str) -> List[float]:
        """Synchronously generate a single embedding vector."""
        result = self.embedder.embed_query_sync(text)
        observe_result(result)
        return list(result.embeddings[0]) #

    @observe_provider("embedder")
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Synchronously generate multiple embedding vectors."""
        if not texts:
            return []
        result = self.embedder.embed_documents_sync(_fit_embedding_documents(texts, self.embedder.model))
        observe_result(result)
        return [list(e) for e in result.embeddings] #

    @observe_provider("cheap_model")
    def is_junk(self, chat_log: str) -> bool:
        """Runs the junk filter agent with the cheap model."""
        result = getattr(self, 'junk_agent', junk_filter_agent).run_sync(
            chat_log, 
            model=self.cheap_model,
            model_settings={"openai_reasoning_effort": "none"}
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
        if isinstance(schema, type) and issubclass(schema, ExtractionMemorySchema):
            deps = TomteDeps(taxonomy=taxonomy or [], lexicon=lexicon or [])
            extractor = getattr(self, 'extractor_agent', tomte_extractor_agent) if schema is ExtractionMemorySchema else make_extractor(schema)
            result = _run_structured_agent(
                extractor, schema, "extraction", user_text,
                model=self.strong_model,
                model_settings={"openai_reasoning_effort": "low"},
                deps=deps
            )
            _log_structured_retries(result, schema, "extraction")
            return result.output
            
        # Route B: Phase 3 Reconciliation or Generic Tasks (Dynamic Schema)
        else:
            # Spin up a temporary, stateless agent for this specific schema
            dynamic_agent = Agent(
                retries=0 if schema in (CandidateFilterResponse,AnswerFactSelection,FactBatchJudgment) else 1 if recoverable(schema) else getattr(self, 'agent_retries', 1),
                # Only mutation planning opts into strict generation. Local
                # provenance/lifecycle validation still owns acceptance.
                output_type=ToolOutput(schema, strict=True) if schema is MutationPlanResponse else schema,
                system_prompt=TRUST_BOUNDARY + (system_prompt or "You are a helpful data extraction assistant.")
            )
            install_recovery(dynamic_agent, schema)
            stage = {"MutationPlanResponse": "fact_mutation",
                     "CandidateFilterResponse": "candidate_filter",
                     "AnswerFactSelection": "answer_reader",
                     "FactBatchJudgment": "answer_reader",
                     "CatalogueSelection": "retrieval_planner",
                     "SearchEmbeddingQuery": "retrieval_planner",
                     "UnresolvedResolutionResponse": "unresolved_resolution",
                     "TerminologyAlignmentResponse": "terminology_alignment", "FieldAlignmentResponse": "terminology_alignment"}.get(
                         getattr(schema, "__name__", ""), "structured")
            model = self.strong_model
            if stage == "candidate_filter":
                model = self.candidate_filter_model
            elif stage == "answer_reader":
                model = self.answer_reader_model
            elif stage == "retrieval_planner":
                model = self.retrieval_planner_model
            elif stage == "terminology_alignment":
                model = self.cheap_model
            try:
                result = _run_structured_agent(
                    dynamic_agent, schema, stage, user_text,
                    model=model,
                    model_settings={"openai_reasoning_effort": "low"}
                )
            except (OpenAIError, TransportError, TimeoutError, ModelHTTPError, UnexpectedModelBehavior, ValidationError) as exc:
                if schema in (AnswerFactSelection, FactBatchJudgment):
                    raise AnswerReaderUnavailable('answer_reader_unavailable') from exc
                if schema is not CandidateFilterResponse:
                    raise
                # Do not swallow cancellation, capacity, budget or host-code errors.
                raise CandidateFilterUnavailable("candidate_filter_unavailable") from exc
            _log_structured_retries(result, schema, stage)
            return result.output
