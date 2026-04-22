# tomte/engine/dictionary/service.py

from typing import List, Dict, Any
from tomte.engine.core.config import settings
from tomte.engine.core.enums import DictionaryKeywordType
from tomte.engine.dictionary import logic
from tomte.engine.database.postgre import dictionary_ops

class DictionaryService:
    def __init__(self, db_conn):
        self.conn = db_conn

    def bootstrap(self, chat_vectors: List[List[float]]) -> Dict[DictionaryKeywordType, List[str]]:
        """
        Phase 1: Topic grouping -> Concept Lookup -> Tiered Bucketing -> Final Truncation.
        """
        centroids = logic.calculate_topic_centroids(chat_vectors, eps=0.5)
        
        # We bucket the keywords to ensure Semantic Relevance beats Historical Hits
        tiered_keywords = {
            "high": {},
            "mid": {},
            "low": {}
        }

        for c in centroids:
            # Query ONCE down to the lowest threshold to save database round-trips
            res = dictionary_ops.get_semantic_keywords(
                self.conn, 
                c, 
                settings.llm_threshold_low, # Fetch all acceptable matches
                id_limit=20
            )
            
            for item in res:
                unique_key = (item['keyword'], item['keyword_type'])
                hits = item['hits']
                score = item['score']
                
                # Assign to the appropriate semantic tier
                if score >= settings.llm_threshold_high:
                    tier = "high"
                elif score >= settings.llm_threshold_mid:
                    tier = "mid"
                else:
                    tier = "low"
                    
                # Deduplication: If this keyword was already found in a HIGHER tier, ignore it
                if unique_key in tiered_keywords["high"] and tier != "high":
                    continue
                if unique_key in tiered_keywords["mid"] and tier == "low":
                    continue

                # Deduplicate within the current tier: Keep the version with the most hits
                if unique_key not in tiered_keywords[tier] or hits > tiered_keywords[tier][unique_key]:
                    tiered_keywords[tier][unique_key] = hits

        final_entities = []
        final_attributes = []
        safe_limit = settings.max_keywords_for_llm

        # Iterate through the tiers in strict order of importance
        for tier in ["high", "mid", "low"]:
            # Sort THIS specific tier by hits descending
            sorted_tier = sorted(tiered_keywords[tier].items(), key=lambda x: x[1], reverse=True)
            
            for k, hits in sorted_tier:
                keyword_str, keyword_type = k
                
                if keyword_type == DictionaryKeywordType.ENTITY.value:
                    if len(final_entities) < safe_limit:
                        final_entities.append(keyword_str)
                
                elif keyword_type == DictionaryKeywordType.ATTRIBUTE.value:
                    if len(final_attributes) < safe_limit:
                        final_attributes.append(keyword_str)

        return {
            DictionaryKeywordType.ENTITY: final_entities, 
            DictionaryKeywordType.ATTRIBUTE: final_attributes
        }
    
    # tomte/engine/dictionary/service.py

    def apply_feedback(self, refined_ops: List[Dict[str, Any]]):
        """
        Phase 4: Updates hit counts and nudges vectors based on refined results.
        Fixes the 0.75/0.85 overlap paradox by making Scenarios A and B mutually exclusive.
        """
        try:
            for op in refined_ops:
                vector = op['embedding']
                
                # Extract keywords from the operation and the nested payload
                keywords = [(op['entity_type'], DictionaryKeywordType.ENTITY)]
                for fact in op.get('extracted_facts', []): 
                    keywords.append((fact['attribute_name'], DictionaryKeywordType.ATTRIBUTE))
                # 1. Identify anchors to update (Threshold: > 0.75)
                matches = dictionary_ops.find_matches_above_threshold(self.conn, vector, settings.llm_threshold_mid)
                
                if matches:
                    # SCENARIO A: Update ONLY the single best existing anchor
                    best_match = matches[0] # Grab the highest scoring anchor
                    d_id = best_match['dictionary_id']
                    
                    # Calculate nudged vector once
                    details = dictionary_ops.get_anchor_details(self.conn, d_id)
                    if not details:
                        continue
                    new_v = logic.nudge_vector(details['embedding'], vector, details['total_hits'])
                    
                    # Update vector ONCE
                    dictionary_ops.update_anchor_vector(self.conn, d_id, new_v)
                    
                    # Upsert all keywords to this specific concept
                    for kw, kw_type in keywords:
                        dictionary_ops.upsert_keyword(self.conn, d_id, kw, kw_type.value)
                else:
                    # SCENARIO B: Create new anchor
                    new_id = dictionary_ops.create_new_anchor(self.conn, vector)
                    for kw, kw_type in keywords:
                        dictionary_ops.upsert_keyword(self.conn, new_id, kw, kw_type.value)
            self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e