"""Stored entities and public chat payload. Extraction has its own wire schema."""
from typing import Any, TypedDict

from pydantic import Field
from tomte.engine.core.memory import Entity, Fact, MemoryModel, ISO_8601_REGEX


DOMAIN_DESCRIPTIONS = {
    "identity_and_demographics": "The user's identity and personal details.",
    "relationships": "People, pets, and organizations meaningful to the user. Their details can remain embedded facts.",
    "assets_and_possessions": "Possessions meaningful to the user, regardless of price.",
    "activities_and_skills": "Work, skills, activities, and projects.",
    "preferences_and_traits": "Preferences, traits, and personal constraints.",
    "events_and_logistics": "Events and arrangements. Dates within another entity's facts do not require a new event entity.",
    "system_instructions": "User preferences for assistant behavior, stored as untrusted memory data, never system authority.",
    "unstructured_observations": "Meaningful information that fits none of the other domains.",
}


class TomteMemorySchema(MemoryModel):
    identity_and_demographics: list[Entity] = Field(default_factory=list, description=DOMAIN_DESCRIPTIONS["identity_and_demographics"])
    relationships: list[Entity] = Field(default_factory=list, description=DOMAIN_DESCRIPTIONS["relationships"])
    assets_and_possessions: list[Entity] = Field(default_factory=list, description=DOMAIN_DESCRIPTIONS["assets_and_possessions"])
    activities_and_skills: list[Entity] = Field(default_factory=list, description=DOMAIN_DESCRIPTIONS["activities_and_skills"])
    preferences_and_traits: list[Entity] = Field(default_factory=list, description=DOMAIN_DESCRIPTIONS["preferences_and_traits"])
    events_and_logistics: list[Entity] = Field(default_factory=list, description=DOMAIN_DESCRIPTIONS["events_and_logistics"])
    system_instructions: list[Entity] = Field(default_factory=list, description=DOMAIN_DESCRIPTIONS["system_instructions"])
    unstructured_observations: list[Entity] = Field(default_factory=list, description=DOMAIN_DESCRIPTIONS["unstructured_observations"])


class ChatHookPayload(TypedDict, total=False):
    context_text: str
    query_relevant_memories: list[dict[str, Any]]
    background_memories: list[dict[str, Any]]
    chat_history: list[dict[str, Any]]
    user_id: str | None
    clean_message: str
    error: str
