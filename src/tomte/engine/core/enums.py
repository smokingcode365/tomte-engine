# ==========================================
# 1. ENUMS (Strict Database Constraints)
# ==========================================


from enum import Enum


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

class ProviderDictionaryKeys(str, Enum):
    DB = "DB"
    CACHE = "CACHE"
    LLM = "LLM"

class DictionaryKeywordType(str, Enum):
    ENTITY = "ENTITY"
    ATTRIBUTE = "ATTRIBUTE"
    FIELD = "FIELD"
