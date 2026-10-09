# engine/utils/cognitive.py (Recommended location)
from typing import Dict, Any, List
from tomte.engine.core.assertions import project_facts

from tomte.engine.llm.interfaces import LLMProvider

def get_entity_summary(entity_data: Dict[str, Any]) -> str:
    """
    Converts a raw Entity dictionary into a high-fidelity Semantic Signature.
    This string is used for generating vectors that anchor the entity in 
    both the Knowledge Graph and the Semantic Dictionary.
    """
    # 1. Core Identity (Highest Weight)
    # We include the Domain and Type to ensure the vector lands in the right 'neighborhood'
    domain = entity_data.get("entity_domain", "unknown").replace("_", " ")
    e_type = entity_data.get("entity_type", "unknown").replace("_", " ")
    name = entity_data.get("entity_name", "Unknown")
    
    summary_parts = [
        f"DOMAIN: {domain}",
        f"TYPE: {e_type}",
        f"NAME: {name}"
    ]

    # 2. Add Aliases (Identity nuances)
    aliases = entity_data.get("aliases", [])
    if aliases:
        summary_parts.append(f"ALSO KNOWN AS: {', '.join(aliases)}")

    # 3. Filtered Facts (The 'Substance')
    # We only include Stable or Permanent facts (Importance >= 3) to prevent 
    # transient comments (e.g., 'I am hungry') from drifting the anchor vector.
    facts = project_facts(entity_data.get("facts", []))
    stable_facts = [
        f"{f['attribute_name'].replace('_', ' ')}: {f['value']}"
        for f in facts 
        if f.get("importance", 3) >= 3
    ]
    
    if stable_facts:
        summary_parts.append("FACTS: " + " | ".join(stable_facts))

    # Join with a clear separator to help the transformer model 
    # distinguish between different logical components[cite: 114].
    return " | ".join(summary_parts)


def embed_entity(llm: LLMProvider, entity_data: Dict[str, Any]) -> List[float]:
    """
    Embeds a raw Entity dictionary into a high-fidelity Semantic Signature.
    This string is used for generating vectors that anchor the entity in 
    both the Knowledge Graph and the Semantic Dictionary.
    """
    return llm.embed_text(get_entity_summary(entity_data))
