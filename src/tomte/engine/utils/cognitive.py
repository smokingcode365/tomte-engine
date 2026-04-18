from typing import List, Dict, Any

from tomte.engine.core.schema import Entity

class CognitiveProcessor:
    """
    The Tomte Engine Memory Router.
    Evaluates synaptic weights, enforces relevance thresholds, and formats context.
    """
    
    def __init__(self, relevance_threshold: float = 0.65, max_context_chars: int = 12000):
        """
        :param relevance_threshold: The minimum score (0.0 to 1.0) a fact needs to be included.
        :param max_context_chars: Hard limit on context size to prevent LLM context window overflow.
        """
        self.relevance_threshold = relevance_threshold
        self.max_context_chars = max_context_chars

    def filter_memories(self, raw_db_results: List[Dict[str, Any]], sort_mode: str = "relevance") -> List[Dict[str, Any]]:
        """
        Extracts payloads and filters by threshold.
        If sort_mode is 'importance', it prioritizes facts inside the payload.
        """
        # 1. Extract and Filter by Threshold
        valid_payloads = []
        for row in raw_db_results:
            score = row.get('relevance_score', 1.0)
            if score >= self.relevance_threshold:
                valid_payloads.append(row['payload'])

        # 2. Internal Fact Sorting
        if sort_mode == "importance":
            for p in valid_payloads:
                if 'facts' in p:
                    # Sort the facts array inside each entity by importance (High to Low)
                    p['facts'].sort(key=lambda x: x.get('importance', 3), reverse=True)
        
        return valid_payloads

    def _format_entity(self, payload: Dict[str, Any]) -> str:
        """
        Converts a single JSONB Entity payload into a human/LLM-readable string.
        """
        entity_type = payload.get('entity_type', 'unknown').upper()
        entity_id = payload.get('entity_id', 'N/A')
        
        lines = [f"[{entity_type} | ID: {entity_id}]"]
        
        # Add Aliases if they exist
        aliases = payload.get('aliases', [])
        if aliases:
            lines.append(f"  Known as: {', '.join(aliases)}")
            
        # Format the Facts
        facts = payload.get('facts', [])
        for fact in facts:
            attr = fact.get('attribute_name', 'unknown')
            val = fact.get('value', 'unknown')
            lines.append(f"  - {attr}: {val}")
            
        return "\n".join(lines)

    def assemble_jit_context(self, payloads: List[Dict[str, Any]], title: str = "RECALLED MEMORIES") -> str:
        """
        Assembles the filtered payloads into a clean text block for prompt injection.
        Enforces the max_context_chars limit to protect token budgets.
        """
        if not payloads:
            return ""

        context_blocks = [f"--- {title} ---"]
        current_length = len(context_blocks[0])

        for payload in payloads:
            entity_text = self._format_entity(payload)
            
            # Check if adding this entity exceeds our context window budget
            if current_length + len(entity_text) > self.max_context_chars:
                context_blocks.append("\n[Notice: Further memories truncated due to cognitive limits.]")
                break
                
            context_blocks.append(entity_text)
            current_length += len(entity_text)

        context_blocks.append("-" * (len(title) + 8))
        return "\n\n".join(context_blocks)

    def prepare_baseline_profile(self, baseline_payloads: List[Dict[str, Any]]) -> str:
        """
        Specifically formats the 'Identity' and 'System Instruction' domains
        that are injected into every single LLM call.
        """
        return self.assemble_jit_context(baseline_payloads, title="CORE USER PROFILE")
        
    def prepare_rag_context(self, semantic_payloads: List[Dict[str, Any]], recent_payloads: List[Dict[str, Any]]) -> str:
        """
        Combines semantic matches (meaning) with recent matches (time) 
        into a single context block, removing duplicates.
        """
        # Deduplicate based on entity_id
        seen_entities = set()
        combined_payloads = []
        
        for payload in recent_payloads + semantic_payloads:
            e_id = payload.get('entity_id')
            if e_id not in seen_entities:
                seen_entities.add(e_id)
                combined_payloads.append(payload)
                
        return self.assemble_jit_context(combined_payloads, title="RELEVANT CONTEXT")


    def merge_entities(self, *lists: List[Entity]) -> List[Entity]:
        """
        Merges multiple lists of Entities.
        If an entity_id appears in more than one list, the entity from the later
        list (higher priority) replaces the one from earlier lists.
        """
        if not lists:
            return []

        master_map: Dict[str, Entity] = {e.entity_id: e for e in lists[0]}

        for lst in lists[1:]:
            for entity in lst:
                master_map[entity.entity_id] = entity

        return list(master_map.values())

    def format_entities(self, entities: List[Entity]) -> str:
        """
        Converts a list of Entities into the structured string 
        representation for the LLM context.
        """
        if not entities:
            return ""

        formatted_blocks = []
        for entity in entities:
            # Sort facts by importance (5 -> 1) before formatting
            sorted_facts = sorted(
                entity.facts, 
                key=lambda x: x.importance, 
                reverse=True
            )

            fact_lines = [
                f"- {f.attribute_name}: {f.value} (imp: {f.importance})"
                for f in sorted_facts
            ]

            block = (
                f"[[ENTITY: {entity.entity_id} | TYPE: {entity.entity_type.value}]]\n"
                f"{chr(10).join(fact_lines)}"
            )
            formatted_blocks.append(block)

        return "\n\n".join(formatted_blocks)