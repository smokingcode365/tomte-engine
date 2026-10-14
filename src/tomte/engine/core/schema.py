"""
Tomte Engine: Master Knowledge Graph Schema
Schema version: 1.0 (experimental; not a production-readiness designation)
Sub-system: Deterministic Extraction & Ontology Alignment

This schema enforces a "Bounded Hybrid" architecture. The LLM is forced to
categorize all human knowledge into 8 mutually exclusive domains, using
highly explicit Field descriptions to eliminate categorization ambiguity.
"""

from typing import Any, List, Optional, TypedDict, Dict
from pydantic import BaseModel, Field

from tomte.engine.core.assertions import AssertionState, EvidenceType, FactEvidence
from tomte.engine.core.enums import FactStatus, MemoryDomain, Sentiment, Volatility

# ==========================================
# 0. JIT PROMPT CONSTANTS (The "Anti-Drift" Rules)
# ==========================================
# Injected directly into Pydantic fields. The LLM reads these at the exact
# millisecond it calculates the token probabilities for that field.

TAXONOMY_RULE = "MUST strictly use generic snake_case (e.g., use 'vehicle' not 'Sports Car'). No spaces, no capital letters."
LEXICON_RULE = "MUST preserve the exact capitalization, spelling, and formatting established by the user or the historical dictionary."
ENTITY_ID_RULE = "MUST generate a unique 3-letter prefix indicating the type, followed by an underscore and a number (e.g., 'veh_01' for a vehicle, 'per_01' for a person, 'loc_01' for a location)."
ISO_8601_REGEX = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"


# ==========================================
#  CORE WRAPPERS (The Knowledge Graph Edges)
# ==========================================


class TimelineEvent(BaseModel):
    """Logs state changes to preserve chronological history without deleting old data."""

    event_description: str = Field(
        description="A concise verb-driven description of what changed in reality (e.g., 'purchased vehicle', 'graduated college', 'moved out')."
    )
    timestamp: str = Field(
        description="The exact UTC timestamp the event occurred or was stated.",
        pattern=ISO_8601_REGEX,
    )
    source_chat_id: str = Field(
        description="The unique database ID of the chat message that triggered this historical event, used for debugging."
    )


class Fact(BaseModel):
    """One semantic assertion; lifecycle changes preserve its attribute/value identity."""

    fact_id: Optional[str] = None
    assertion_state: Optional[AssertionState] = None
    origin: Optional[EvidenceType] = None
    created_at: Optional[str] = Field(default=None, pattern=ISO_8601_REGEX)
    effective_at: Optional[str] = Field(default=None, pattern=ISO_8601_REGEX)
    ended_at: Optional[str] = Field(default=None, pattern=ISO_8601_REGEX)
    supersedes_fact_id: Optional[str] = None
    corrects_fact_id: Optional[str] = None
    evidence: List[FactEvidence] = Field(default_factory=list)

    attribute_name: str = Field(
        description=f"The categorical key for this specific data point (e.g., 'paint_color', 'job_title', 'purchase_price'). {TAXONOMY_RULE}"
    )
    value: str = Field(
        description=f"The actual data point. MUST be a flat string. DO NOT use nested JSON objects here. {LEXICON_RULE}"
    )
    importance: int = Field(
        default=3, 
        ge=1, le=5, 
        description="5:(Eternal). 4:(Long-Term).  3:(Stable). 2:(Contextual) 1:(Transient)"
        )
    status: FactStatus = Field(
        default=FactStatus.ACTIVE,
        description="Is this fact currently true, a past memory, or a future goal?",
    )
    sentiment: Sentiment = Field(
        default=Sentiment.NEUTRAL,
        description="How does the user feel about this specific fact?",
    )
    volatility: Volatility = Field(
        default=Volatility.CORE,
        description="Is this a permanent core fact or a temporary ephemeral thought?",
    )
    related_entity_ids: List[str] = Field(
        default_factory=list,
        description="CRITICAL FOR GRAPH EDGES: If this fact connects this entity to another known entity, put the target entity_id here (e.g., ['per_01'] to link a father to a son). If it is just a standalone attribute, leave empty.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Uncalibrated stored fact-confidence signal, distinct from identity and mutation interpretation confidence.",
    )
    source_chat_id: str = Field(
        description="The unique ID of the raw chat message where this fact was found."
    )
    timeline: List[TimelineEvent] = Field(
        default_factory=list,
        description="A chronological array logging any historical changes to this specific fact over time.",
    )

    # --- COGNITIVE METRICS (For Memory Decay Math) ---
    mention_count: int = Field(
        default=1,
        description="The total number of separate times the user has brought up this exact fact. Initialize to $$1$$ for new facts.",
    )
    last_mentioned_at: str = Field(
        description="The exact UTC timestamp this fact was most recently referenced by the user.",
        pattern=ISO_8601_REGEX,
    )


# ==========================================
# 3. ENTITY DEFINITION (The Graph Nodes)
# ==========================================


class Entity(BaseModel):
    """A primary node in the Knowledge Graph. The digital anchor for a physical or abstract concept."""

    entity_id: str = Field(
        description=f"The primary key for this entity. {ENTITY_ID_RULE}"
    )
    entity_name: str = Field(
        description="The human-friendly name of the entity (e.g., 'aquarium', 'software project', 'electric vehicle')."
    )
    entity_domain: MemoryDomain = Field(
        description=f"The master category this entity belongs to. Choose the best fit from the 8 domains"
    )
    entity_type: str = Field(
        description="The granular classification of the entity (e.g., 'aquarium', 'software_project', 'electric_vehicle'). "
                "This is the specific 'what' of the entity, sitting underneath the broader entity_domain. "
                "Do NOT use the domain name as the type.")
    aliases: List[str] = Field(
        default_factory=list,
        description="A list of verbatim nouns, nicknames, or pronouns the user actually uses in dialogue to refer to this entity (e.g., ['my car', 'the Tesla', 'the beast']). DO NOT invent aliases.",
    )
    facts: List[Fact] = Field(
        default_factory=list,
        description="A list of all known Fact objects (attributes, details, and connections) that belong exclusively to this entity.",
    )


# ==========================================
# 4. MASTER SCHEMA (The 8 Mutually Exclusive Domains)
# ==========================================


class TomteMemorySchema(BaseModel):
    """
    The master extraction schema.
    You MUST sort every newly discovered entity into one of these 8 domains.
    Read the descriptions carefully to understand the boundaries of what is included and excluded.
    """

    identity_and_demographics: List[Entity] = Field(
        default_factory=list,
        description="Core foundational facts defining the user's permanent self. INCLUDES: Legal names, physical living locations, age, biological gender, contact information, and nationality. EXCLUDES: Jobs, skills, family members, or physical possessions.",
    )

    relationships: List[Entity] = Field(
        default_factory=list,
        description="Living (or previously living) beings and formal organizations the user interacts with. INCLUDES: Family members ('wife', 'son'), friends, romantic partners, pets ('dog', 'cat'), employers, and specific companies. EXCLUDES: Inanimate objects.",
    )

    assets_and_possessions: List[Entity] = Field(
        default_factory=list,
        description="High-value physical or digital property owned by the user. INCLUDES: Real estate, vehicles, expensive electronics, software licenses, and major financial investments. EXCLUDES: Cheap, everyday disposable items, groceries, or temporary purchases.",
    )

    activities_and_skills: List[Entity] = Field(
        default_factory=list,
        description="Actions, routines, and capabilities of the user. INCLUDES: Professional careers, programming languages, sports, hobbies, and active ongoing projects (e.g., 'learning Python', 'crabbing', 'marathon training').",
    )

    preferences_and_traits: List[Entity] = Field(
        default_factory=list,
        description="Subjective internal states, opinions, and biological rules. INCLUDES: Favorite brands, dietary restrictions, severe allergies, risk tolerance, communication styles, political/religious views, likes, and dislikes.",
    )

    events_and_logistics: List[Entity] = Field(
        default_factory=list,
        description="Time-bound, situational occurrences. INCLUDES: Upcoming flights, vacations, lease renewals, medical appointments, and project deadlines. RULE: If a fact is temporary and defined by a specific future or past date on a calendar, it belongs here.",
    )

    system_instructions: List[Entity] = Field(
        default_factory=list,
        description="Explicit or implicit rules the user sets governing how YOU (the AI system) should behave. INCLUDES: 'Speak formally', 'Always format math using LaTeX', 'Never mention my ex-wife', 'Keep answers under 50 words'.",
    )

    unstructured_observations: List[Entity] = Field(
        default_factory=list,
        description="THE ESCAPE HATCH. Use this domain strictly for highly specific, bizarre, or niche data that violently conflicts with or does not perfectly fit the definitions of the other 7 categories. Using this prevents the corruption of standard tables.",
    )



class ChatHookPayload(TypedDict, total=False):
    """
    Standardized return payload for the Chat Hook.
    total=False allows us to return just an 'error' key if needed.
    """
    context_text: str
    query_relevant_memories: List[Dict[str, Any]]
    background_memories: List[Dict[str, Any]]
    chat_history: List[Dict[str, Any]]
    user_id: Optional[str]
    clean_message: str
    error: str