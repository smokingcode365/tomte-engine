# ==========================================
# 1. ENUMS (Strict Database Constraints)
# ==========================================


from enum import Enum


class FactStatus(str, Enum):
    """The temporal, objective state of a fact in reality."""

    ACTIVE = "active"  # Currently true right now.
    PAST = "past"  # Historically true, but objectively over.
    ASPIRATIONAL = "aspirational"  # Future goals, desires, or unverified plans.
    ARCHIVED = "archived"  # Soft-deleted due to an AI hallucination or user correction.


class Sentiment(str, Enum):
    """The user's subjective emotional stance toward this specific fact."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Volatility(str, Enum):
    """Signals to the database's garbage collection system how long to keep this."""

    CORE = "core"  # Permanent, identity-defining data (e.g., birthdate, severe allergies, legal names).
    EPHEMERAL = "ephemeral"  # Highly temporary states (e.g., current mood, running a quick errand, hungry).


class MemoryDomain(str, Enum):
    IDENTITY = "identity_and_demographics"
    RELATIONSHIPS = "relationships"
    ASSETS = "assets_and_possessions"
    ACTIVITIES = "activities_and_skills"
    PREFERENCES = "preferences_and_traits"
    LOGISTICS = "events_and_logistics"
    SYSTEM = "system_instructions"
    UNSTRUCTURED = "unstructured_observations"

class ChatSenderRole(str, Enum):
    ASSISTANT = "assistant"
    USER = "user"

class ProvierDictionaryKes(str, Enum):
    DB = "DB"
    CACHE = "CACHE"
    LLM = "LLM"