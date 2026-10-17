from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class OpenAISettings(BaseModel):
    strong_model: str = "gpt-6.1-sol"
    cheap_model: str = "gpt-6-luna"
    embed_model: str = "openai:text-embedding-3-small"
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

class TomteSettings(BaseSettings):
    max_model_input_bytes: int = Field(default=65536, ge=1, description="Conservative UTF-8 input capacity per model stage; not a token/context guarantee.")
    provider_diagnostics_enabled: bool = Field(default=False, description="Emit bounded metadata-only provider diagnostics; no raw payloads.")
    max_keywords_for_llm: int = Field(default=400, description="The maximum number of keywords to send to the LLM for extraction.")
    max_input_length: int = Field(default=4000, description="The maximum length of the input text.")
    enable_pii_scrubbing: bool = Field(default=False)
    consolidation_ttl_minutes: int = Field(default=30)
    history_chat_number: int = Field(default=20)
    isolate_chat_sessions: bool = Field(default=True)
    query_entities_limit: int = Field(default=3) # Limit for the number of entities to query from the database
    semantic_retrieval_limit: int = Field(default=5) # Limit for the number of entities to retrieve from the semantic retrieval
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
