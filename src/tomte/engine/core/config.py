from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class OpenAISettings(BaseModel):
    strong_model: str = "gpt-4o-2024-08-06"
    cheap_model: str = "gpt-4o-mini"
    embed_model: str = "text-embedding-3-small"
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
    consolidation_ttl_minutes: int = Field(default=30)
    history_chat_number: int = Field(default=20)
    isolate_chat_sessions: bool = Field(default=False)
    query_entities_limit: int = Field(default=3) # Limit for the number of entities to query from the database
    semantic_retrieval_limit: int = Field(default=5) # Limit for the number of entities to retrieve from the semantic retrieval
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    anthropic: AnthropicSettings = Field(default_factory=AnthropicSettings)
    gemini: GeminiSettings = Field(default_factory=GeminiSettings) # Added

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore", env_nested_delimiter="__"
    )

settings: TomteSettings | None = None