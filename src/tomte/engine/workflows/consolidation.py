# tomte/engine/workflows/consolidation.py

from typing import List, Dict, Any, Tuple
from tomte.engine.database.db_interface import DatabaseProvider
from tomte.engine.llm.embedder import embed_entity
from tomte.engine.llm.interfaces import LLMProvider
from tomte.engine.dictionary.service import DictionaryService
from tomte.engine.core.schema import TomteMemorySchema
from tomte.engine.core.enums import DictionaryKeywordType
from tomte.engine import log
from tomte.engine.llm.prompts.consolidation_golden_record import RECONCILER_SYSTEM_PROMPT

class ConsolidationWorkflow:
    def __init__(
        self, 
        db: DatabaseProvider, 
        llm: LLMProvider, 
        dict_service: DictionaryService
    ):
        self.db = db
        self.llm = llm
        self.dict_service = dict_service

    def run_consolidation_cycle(self, tenant_id: str, user_id: str, chat_items: List[Dict[str, Any]]) -> bool:
        """
        Executes the full 4-Phase consolidation cycle[cite: 137, 152, 157].
        """
        # 1. Preparation: Extract text and vectors [cite: 139-140]
        chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_items])
        chat_vectors = [m['vector'] for m in chat_items if m.get('vector')]

        if not chat_vectors:
            log.warn(f"No vectors found for consolidation session: {user_id}")
            return False

        # --- PHASE 1: Semantic Bootstrapping ---
        # Clusters chat vectors to find relevant dictionary keywords [cite: 137-151]
        log.debug(f"Phase 1: Bootstrapping dictionary for {user_id}")
        bootstrap_dict = self.dict_service.bootstrap(chat_vectors)
        
        # --- PHASE 2: Generative Extraction ---
        # First LLM Call: Extracts entities using the bootstrap dictionary 
        log.debug(f"Phase 2: Executing primary extraction")
        extracted_schema: TomteMemorySchema = self.llm.parse_structured(
            system_prompt=None, 
            user_text=chat_text,
            schema=TomteMemorySchema,
            taxonomy=bootstrap_dict[DictionaryKeywordType.ENTITY],
            lexicon=bootstrap_dict[DictionaryKeywordType.ATTRIBUTE]
        )

        # --- PHASE 3: Golden Record Reconciliation ---
        # Second LLM Call: Matches extraction against the DB skeleton [cite: 157-159]
        log.debug(f"Phase 3: Reconciling with Golden Records")
        skeleton = self.db.get_user_graph_skeleton(tenant_id, user_id)
        
        # We flatten the Pydantic schema into a list of entities for the reconciler
        raw_entities = self._flatten_schema(extracted_schema)
        
        # This helper determines if we INSERT, MUTATE, or REINFORCE [cite: 164-169]
        refined_ops = self._reconcile_with_golden_records(
            tenant_id, 
            user_id, 
            raw_entities, 
            skeleton
        )

        # Before committing, we must generate vectors for any new or changed entities.
        # These vectors anchor the facts in the Knowledge Graph [cite: 160-171].
        log.debug(f"Consolidation: Generating embeddings for {len(refined_ops)} operations")
        for op in refined_ops:
            # Use your new utility to generate the concept-based vector
            # This ensures the vector is 'clean' (no JSON noise/timeline noise)
            op['embedding'] = embed_entity(self.llm, op['payload']) # [cite: 114]

        # --- ATOMIC COMMIT ---
        # Now the ops contain the 'embedding' vector required by the SQL logic [cite: 145-177]
        if not self.db.commit_graph_updates(refined_ops):
            log.error(f"Consolidation commit failed for user {user_id}")
            return False

        # --- PHASE 4: Dictionary Feedback Loop ---
        # We pass the same refined_ops (now with vectors) to the dictionary service
        # to nudge the anchors and update keyword hits [cite: 157-171].
        log.debug(f"Phase 4: Applying dictionary feedback")
        self.dict_service.apply_feedback(refined_ops)

        log.info(f"Consolidation cycle complete for {user_id}")
        return True

    # ==========================================
    # INTERNAL HELPERS
    # ==========================================

    def _flatten_schema(self, schema: TomteMemorySchema) -> List[Dict[str, Any]]:
        """Converts the multi-domain Pydantic schema into a flat list of entities."""
        all_entities = []
        # Access each domain defined in the TomteMemorySchema
        domains = [
            schema.identity_and_demographics,
            schema.relationships,
            schema.assets_and_possessions,
            schema.activities_and_skills,
            schema.preferences_and_traits,
            schema.events_and_logistics,
            schema.system_instructions,
            schema.unstructured_observations
        ]
        for domain_list in domains:
            for entity in domain_list:
                all_entities.append(entity.model_dump(mode='json'))
        return all_entities

    def _reconcile_with_golden_records(
        self, 
        tenant_id: str, 
        user_id: str, 
        extracted: List[Dict[str, Any]], 
        skeleton: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Uses the second LLM call to frame new entities against existing ones[cite: 159].
        Determines the 'mode' (INSERT/MUTATE/REINFORCE) for each operation [cite: 101, 164-169].
        """
        if not extracted:
            return []

        # 1. Define Internal Models for the Reconciler
        from pydantic import BaseModel
        from typing import Literal, Optional
        import json

        class EntityMapping(BaseModel):
            extracted_entity_id: str
            mode: Literal["INSERT", "MUTATE", "REINFORCE"]
            matched_db_id: Optional[str] = None # The UUID from the skeleton
            reasoning: str

        class ReconciliationResponse(BaseModel):
            mappings: List[EntityMapping]

        # 2. Prepare context for the LLM
        # We provide a simplified view of the extracted entities to focus the LLM on mapping [cite: 154]
        extracted_summary = [
            {
                "id": e["entity_id"],
                "name": e.get("entity_name", "N/A"),
                "type": e["entity_type"],
                "aliases": e.get("aliases", []),
                "fact_keys": [f["attribute_name"] for f in e.get("facts", [])]
            }
            for e in extracted
        ]
        
        reconciliation_context = (
            f"EXISTING_GOLDEN_RECORDS (SKELETON):\n{json.dumps(skeleton, indent=2)}\n\n"
            f"NEW_EXTRACTED_ENTITIES:\n{json.dumps(extracted_summary, indent=2)}"
        )

        # 3. Call the Reconciler LLM [cite: 159]
        # This is the 'Second Time' call to reconcile with Golden Records
        response: ReconciliationResponse = self.llm.parse_structured(
            system_prompt=RECONCILER_SYSTEM_PROMPT,
            user_text=reconciliation_context,
            schema=ReconciliationResponse
        )

        # 4. Map LLM decisions back to database operations [cite: 164-169]
        mapping_lookup = {m.extracted_entity_id: m for m in response.mappings}
        refined_ops = []

        for entity in extracted:
            decision = mapping_lookup.get(entity["entity_id"])
            
            # Default to INSERT if the LLM failed to provide a mapping for an entity
            mode = decision.mode if decision else "INSERT"
            db_id = decision.matched_db_id if decision else None

            # Calculate entity-level importance based on the most important fact extracted
            max_imp = max([f.get("importance", 3) for f in entity.get("facts", [])]) if entity.get("facts") else 3

            op = {
                "mode": mode,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "entity_id": entity["entity_id"],
                "entity_name": entity.get("entity_name") or entity["entity_id"],
                "entity_domain": entity["entity_domain"],
                "entity_type": entity["entity_type"],
                "payload": entity,
                "importance": max_imp
            }
            
            # If we found a match, add the DB UUID to the operation [cite: 153-154]
            if db_id:
                op["db_id"] = db_id
                
            refined_ops.append(op)

        return refined_ops