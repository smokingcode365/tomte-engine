# tomte/engine/dictionary/service.py

from typing import List, Dict, Any
from tomte.engine.core.config import settings
from tomte.engine.core.enums import DictionaryKeywordType
from tomte.engine.dictionary import logic
from tomte.engine.database.postgre import dictionary_ops

class DictionaryService:
    def __init__(self, db_conn, config: Dict[str, Any]):
        self.conn = db_conn
        self.config = config

    def bootstrap(self, chat_vectors: List[List[float]]) -> Dict[DictionaryKeywordType, List[str]]:
        """
        Phase 1: Topic grouping -> Concept Lookup -> Deduplication -> Final Truncation.
        """
        centroids = logic.calculate_topic_centroids(
            chat_vectors, 
            eps=self.config.get('dbscan_eps', 0.5)
        )
        
        # Use a map to track max hits for each unique (keyword, type)
        best_keywords: Dict[tuple, int] = {}

        for c in centroids:
            # Query based on Concept ID limit (dictionary_top_k)
            res = dictionary_ops.get_semantic_keywords(
                self.conn, 
                c, 
                self.config.get('match_threshold', 0.75), 
                id_limit=self.config.get('dictionary_top_k', 20)
            )
            
            for item in res:
                unique_key = (item['keyword'], item['keyword_type'])
                hits = item['hits']
                
                # Deduplication logic: Keep the version with the most hits
                if unique_key not in best_keywords or hits > best_keywords[unique_key]:
                    best_keywords[unique_key] = hits

        # Sort the entire global pool by hits descending
        sorted_pool = sorted(best_keywords.items(), key=lambda x: x[1], reverse=True)

        # Truncation logic: Split and cut to "Safe Length"
        safe_limit = self.config.get('max_keywords_per_list', 200)
        
        entities = [
            k[0] for k, hits in sorted_pool 
            if k[1] == DictionaryKeywordType.ENTITY
        ][:safe_limit] # Truncate

        attributes = [
            k[0] for k, hits in sorted_pool 
            if k[1] == DictionaryKeywordType.ATTRIBUTE
        ][:safe_limit] # Truncate

        return {
            DictionaryKeywordType.ENTITY: entities, 
            DictionaryKeywordType.ATTRIBUTE: attributes
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
                payload = op.get('payload', {}) # CRITICAL FIX: Access the nested data
                
                # Extract keywords from the operation and the nested payload
                keywords = [(op['entity_type'], DictionaryKeywordType.ENTITY)]
                for fact in payload.get('facts', []): # CRITICAL FIX: Get facts from payload
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