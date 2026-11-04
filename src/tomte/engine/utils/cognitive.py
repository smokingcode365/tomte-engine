from typing import List, Dict, Any
import json
from tomte.engine.core.memory import public_entity


class CognitiveProcessor:
    """
    The Tomte Engine Memory Router.
    Evaluates synaptic weights, enforces relevance thresholds, and formats context.
    """
    
    def merge_entities(self, *lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        High-performance merge using raw dictionaries.
        Later lists overwrite earlier ones (Priority: Right > Left).
        """
        master_map = {}
        
        for entity_list in lists:
            if not entity_list:
                continue
            for entity in entity_list:
                # O(1) dictionary lookup/overwrite
                eid = entity.get("entity_id")
                if eid:
                    # DB identity wins: corrupt logical-ID collisions must not drop rows.
                    key = ("db", entity.get("tenant_id"), entity.get("user_id"), str(entity["id"])) if entity.get("id") else ("entity", eid)
                    master_map[key] = entity
                    
        return list(master_map.values())

    def format_entities(self, entities: List[Dict[str, Any]]) -> str:
        """Serialize the shared public allowlist, including selected history."""
        return json.dumps([public_entity(entity, entity.get("_history_selection")) for entity in entities], ensure_ascii=False) if entities else ""
