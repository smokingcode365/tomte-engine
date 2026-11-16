from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from tomte.engine.core.memory_text import MemoryTextFormat
from tomte.engine.core.search import StoredRetrievalMode

class OpenAISettings(BaseModel):
    strong_model: str = "gpt-6.1-sol"
    cheap_model: str = "gpt-6-luna"
    candidate_filter_model: str = "gpt-6.1-terra"
    answer_reader_model: str = "gpt-6.1-terra"
    retrieval_planner_model: str = "gpt-6.1-terra"
    embed_model: str = "openai:text-embedding-3-large"
    # Request the supported shortened output to fit existing vector/IVFFlat storage.
    vector_dimension: int = 1536

class AnthropicSettings(BaseModel):
    strong_model: str = "claude-3-5-sonnet-latest"
    cheap_model: str = "claude-3-haiku-20240307"
    embed_model: str = "voyage-large-2"  # Voyage AI partner model
    vector_dimension: int = 1536       # Voyage-large-2 outputs 1536 by default

class GeminiSettings(BaseModel):
    """Default models for the Gemini Provider."""
    strong_model: str = "google-gla:gemini-1.5-pro"   # High reasoning
    cheap_model: str = "google-gla:gemini-1.5-flash" # High speed for junk filter
    embed_model: str = "google-gla:text-embedding-004"
    vector_dimension: int = 768  # Standard for Google embeddings

class PendingSearchSettings(BaseModel):
    max_messages: int = Field(default=1000, ge=1, le=10000)
    max_input_bytes: int = Field(default=262144, ge=1024, le=2097152)
    window_bytes: int = Field(default=6000, ge=512, le=8000)
    max_embedding_bytes: int = Field(default=262144, ge=512, le=2097152)
    return_windows: int = Field(default=3, ge=1, le=20)
    return_bytes: int = Field(default=8000, ge=512, le=131072)
    cache_windows: int = Field(default=256, ge=1, le=1024)
    cache_bytes: int = Field(default=8388608, ge=1024, le=67108864)
    cache_ttl_seconds: int = Field(default=86400, ge=60)
    lease_seconds: int = Field(default=45, ge=5, le=120)
    wait_seconds: float = Field(default=2, ge=0, le=10)


class AnswerRetrievalSettings(BaseModel):
    catalog_limit: int = Field(default=400, ge=1, le=2000)
    catalog_bytes: int = Field(default=32000, ge=1024, le=131072)
    # Legacy larger overrides remain readable; retrieval clamps this to 100.
    fact_limit: int = Field(default=100, ge=1, le=1000)
    lane_bytes: int = Field(default=24000, ge=1024, le=131072)
    read_all_bytes: int = Field(default=12000, ge=512, le=131072)
    memory_bytes: int = Field(default=64000, ge=1024, le=262144)
    catalog_read_all_facts: int = Field(default=6, ge=1, le=100)
    relevance_tail_size: int = Field(default=10, ge=1, le=20)


class TomteSettings(BaseSettings):
    answer_retrieval: AnswerRetrievalSettings = Field(default_factory=AnswerRetrievalSettings)
    stored_retrieval_mode: StoredRetrievalMode = Field(default='vector', description='Stored-memory search: vector, bm25, or experimental hybrid. Pending and mutation searches are unchanged.')
    pending_search: PendingSearchSettings = Field(default_factory=PendingSearchSettings)
    memory_text_format: MemoryTextFormat = Field(default='labelled_rows', description='Caller-facing context text representation; structured entities and storage are unchanged.')
    unresolved_return_limit: int = Field(default=3, ge=1, le=20)
    unresolved_candidate_limit: int = Field(default=5, ge=1, le=20)
    unresolved_recent_limit: int = Field(default=2, ge=0, le=10)
    memory_history_bytes: int = Field(default=12000, ge=0, description="Public history budget per entity; current facts are not truncated.")
    memory_history_candidate_bytes: int = Field(default=20000, ge=1000, description="Bounded history selection input, excluding current values.")
    memory_history_candidate_limit: int = Field(default=40, ge=1, le=200)
    max_model_input_bytes: int = Field(default=327680, ge=1, description="Conservative UTF-8 input capacity per model stage; not a token/context guarantee.")
    consolidation_batch_bytes: int = Field(default=65536, ge=1, description="Prepared chat-input budget for new Redis batches; whole messages are never truncated.")
    provider_diagnostics_enabled: bool = Field(default=False, description="Emit bounded metadata-only provider diagnostics; no raw payloads.")
    max_keywords_for_llm: int = Field(default=400, description="The maximum number of keywords to send to the LLM for extraction.")
    max_input_length: int = Field(default=4000, description="The maximum length of the input text.")
    enable_pii_scrubbing: bool = Field(default=False)
    consolidation_ttl_minutes: int = Field(default=30)
    history_chat_number: int = Field(default=20)
    isolate_chat_sessions: bool = Field(default=True)
    query_entities_limit: int = Field(default=3) # Limit for the number of entities to query from the database
    semantic_retrieval_limit: int = Field(default=5) # Per ranking: summary/field union returns at most twice this many entities.
    # Empirical for current-message -> Tomte summaries with text-embedding-3-small.
    session_entity_cosine_floor: float = Field(default=0.25, ge=-1, le=1)
    session_entity_activation_cap: int = Field(default=5, ge=1)
    session_entity_retention_cap: int = Field(default=10, ge=1)
    session_entity_return_cap: int = Field(default=5, ge=1)
    session_entity_ttl_seconds: int = Field(default=86400, ge=1)
    llm_threshold_high: float = Field(default=0.85)
    llm_threshold_mid: float = Field(default=0.80)
    llm_threshold_low: float = Field(default=0.75)
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings) # Added

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_nested_delimiter="__"
    )

settings: TomteSettings=TomteSettings()
