from pydantic_ai import Agent, RunContext
from tomte.engine.core.schema import TomteMemorySchema
from tomte.engine.llm.prompts.extractor import EXTRACTOR_SYSTEM_PROMPT
from tomte.engine.llm.prompts.junk_filter import JUNK_FILTER_PROMPT
from dataclasses import dataclass
from typing import List

@dataclass
class TomteDeps:
    """Container for JIT (Just-In-Time) prompt injection."""
    taxonomy: List[str]
    lexicon: List[str]


# The templates now use the imported prompts
tomte_extractor_agent = Agent(
    output_type=TomteMemorySchema,
    system_prompt=EXTRACTOR_SYSTEM_PROMPT
)

junk_filter_agent = Agent(
    output_type=bool,
    system_prompt=JUNK_FILTER_PROMPT
)


@tomte_extractor_agent.system_prompt
def add_dictionaries(ctx: RunContext[TomteDeps]) -> str:
    """
    Appends the dynamic Redis dictionaries to the base system prompt.
    """
    # Start with your static prompt from the file
    prompt = EXTRACTOR_SYSTEM_PROMPT 
    
    # Append the JIT Dictionary data [cite: 361-362]
    prompt += f"\n\n### CURRENT TENANT DICTIONARY"
    prompt += f"\n- SCHEMA TAXONOMY (Keys): {ctx.deps.taxonomy}"
    prompt += f"\n- VALUE LEXICON (Values): {ctx.deps.lexicon}"
    
    return prompt