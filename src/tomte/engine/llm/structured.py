from pydantic_ai import Agent, RunContext
from tomte.engine.core.schema import TomteMemorySchema
from tomte.engine.llm.prompts.extractor import EXTRACTOR_SYSTEM_PROMPT, DICTIONARY_EXTRACTION_GUIDANCE
from tomte.engine.llm.prompts.junk_filter import JUNK_FILTER_PROMPT
from dataclasses import dataclass
from typing import List
from tomte.engine.llm.trust import TRUST_BOUNDARY

@dataclass
class TomteDeps:
    """Container for JIT (Just-In-Time) prompt injection."""
    taxonomy: List[str]
    lexicon: List[str]


# The templates now use the imported prompts
tomte_extractor_agent = Agent(
    output_type=TomteMemorySchema,
    system_prompt=TRUST_BOUNDARY + EXTRACTOR_SYSTEM_PROMPT
)

junk_filter_agent = Agent(
    output_type=bool,
    system_prompt=TRUST_BOUNDARY + JUNK_FILTER_PROMPT
)


@tomte_extractor_agent.system_prompt
def add_dictionaries(ctx: RunContext[TomteDeps]) -> str:
    """
    Appends established entity-type and attribute-name vocabulary to the prompt.
    """
    if not ctx.deps.taxonomy and not ctx.deps.lexicon:
        return ""  # Raw extraction: no vocabulary, no duplicate static prompt.
    prompt = DICTIONARY_EXTRACTION_GUIDANCE
    
    # Append the JIT Dictionary data [cite: 361-362]
    prompt += f"\n\n### ESTABLISHED DICTIONARY VOCABULARY"
    prompt += f"\n- ESTABLISHED ENTITY-TYPE VOCABULARY: {ctx.deps.taxonomy}"
    prompt += f"\n- ESTABLISHED ATTRIBUTE VOCABULARY (Names/Keys): {ctx.deps.lexicon}"
    
    return prompt


def make_agents():
    """Operation-owned agents; worker retries own the bounded retry policy."""
    extractor = Agent(output_type=TomteMemorySchema,
                      system_prompt=TRUST_BOUNDARY + EXTRACTOR_SYSTEM_PROMPT, retries=0)
    extractor.system_prompt(add_dictionaries)
    junk = Agent(output_type=bool, system_prompt=TRUST_BOUNDARY + JUNK_FILTER_PROMPT, retries=0)
    return extractor, junk
