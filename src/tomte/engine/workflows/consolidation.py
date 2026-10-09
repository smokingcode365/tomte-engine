# tomte/engine/workflows/consolidation.py

from typing import List, Dict, Any
from tomte.engine.llm.embedder import  get_entity_summary
from tomte.engine.dictionary.service import get_dictionary_service
from tomte.engine.core.schema import TomteMemorySchema
from tomte.engine.core.enums import DictionaryKeywordType
from tomte.engine import tomtelog, Tomte
from tomte.engine.llm.prompts.consolidation_golden_record import RECONCILER_SYSTEM_PROMPT

class ConsolidationWorkflow:
    def __init__(
        self
    ):
        self.db = Tomte.get_db()
        self.llm = Tomte.get_llm()
        self.dict_service = get_dictionary_service()
    # ==========================================
    # CONSOLIDATION CYCLE
    # ==========================================

    def run_consolidation_cycle(self, tenant_id: str, user_id: str, chat_items: List[Dict[str, Any]]) -> bool:
        """
        Executes the full 4-Phase consolidation cycle[cite: 137, 152, 157].
        """
        # 1. Preparation: Extract text and vectors [cite: 139-140]
        chat_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_items])
        chat_vectors = [m['vector'] for m in chat_items if m.get('vector')]

        if not chat_vectors:
            tomtelog.warn(f"No vectors found for consolidation session: {user_id}")
            return False
        # CRITICAL FIX: Inject the Junk Filter here
        if self.llm.is_junk(chat_text):
            tomtelog.info(f"Consolidation aborted: Chat log is junk/small-talk for {user_id}.")
            return False
        # --- PHASE 1: Semantic Bootstrapping ---
        # Clusters chat vectors to find relevant dictionary keywords [cite: 137-151]
        tomtelog.debug(f"Phase 1: Bootstrapping dictionary for {user_id}")
        bootstrap_dict = self.dict_service.bootstrap(chat_vectors)
        
        # --- PHASE 2: Generative Extraction ---
        # First LLM Call: Extracts entities using the bootstrap dictionary 
        tomtelog.debug(f"Phase 2: Executing primary extraction")
        extracted_schema: TomteMemorySchema = self.llm.parse_structured(
            system_prompt=None, 
            user_text=chat_text,
            schema=TomteMemorySchema,
            taxonomy=bootstrap_dict[DictionaryKeywordType.ENTITY],
            lexicon=bootstrap_dict[DictionaryKeywordType.ATTRIBUTE]
        )

        # --- PHASE 3: Golden Record Reconciliation ---
        # Second LLM Call: Matches extraction against the DB skeleton [cite: 157-159]
        tomtelog.debug(f"Phase 3: Reconciling with Golden Records")
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
        tomtelog.debug(f"Consolidation: Generating embeddings for {len(refined_ops)} operations")
        if refined_ops:
            summaries_to_embed = [get_entity_summary(op['payload']) for op in refined_ops]
            batch_vectors = self.llm.embed_batch(summaries_to_embed)
            for i, op in enumerate(refined_ops):
                op['embedding'] = batch_vectors[i]
                
        # Prepared memory writes and dictionary feedback form one PostgreSQL unit.
        self._commit_refined_operations(refined_ops)

        tomtelog.info(f"Consolidation cycle complete for {user_id}")
        return True

    def _commit_refined_operations(self, refined_ops: List[Dict[str, Any]]) -> None:
        """Persist prepared memory changes and dictionary feedback atomically."""
        with self.db.transaction():
            if not self.db.commit_graph_updates(refined_ops):
                raise RuntimeError("Consolidation memory write failed")
            self.dict_service.apply_feedback(refined_ops)

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

    # ==========================================
    # RECONCILIATION HELPER
    # ==========================================
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
        
        # PREPARE LLM CONTEXT (Fix 2: Token Saver & UUID Crash Fix)
        # We summarize the skeleton so we don't blow up the LLM token limit with massive payloads,
        # and we cast the UUID to a string so json.dumps doesn't crash.
        llm_skeleton_view = [
            {
                "id": str(s["id"]), 
                "entity_id": s["entity_id"],
                "name": s["entity_name"],
                "type": s["entity_type"],
                # We only show the LLM the keys of existing facts so it knows what to MUTATE
                "existing_fact_keys": [f["attribute_name"] for f in s.get("payload", {}).get("facts", [])] 
            }
            for s in skeleton
        ]
        
        reconciliation_context = (
            f"EXISTING_GOLDEN_RECORDS (SKELETON):\n{json.dumps(llm_skeleton_view, indent=2)}\n\n"
            f"NEW_EXTRACTED_ENTITIES:\n{json.dumps(extracted_summary, indent=2)}"
        )

        # 3. Call the Reconciler LLM
        response: ReconciliationResponse = self.llm.parse_structured(
            system_prompt=RECONCILER_SYSTEM_PROMPT,
            user_text=reconciliation_context,
            schema=ReconciliationResponse
        )

        # 4. Map LLM decisions back to database operations
        mapping_lookup = {m.extracted_entity_id: m for m in response.mappings}
        refined_ops = []

        for entity in extracted:
            decision = mapping_lookup.get(entity["entity_id"])
            
            # Default to INSERT if the LLM failed to provide a mapping for an entity
            mode = decision.mode if decision else "INSERT"
            db_id = decision.matched_db_id if decision else None
            
            # ==========================================
            # FALLBACK GUARD (Fix 3: Prevent Rollbacks)
            # ==========================================
            if mode in ["MUTATE", "REINFORCE"] and not db_id:
                tomtelog.warn(f"LLM flagged {mode} but provided no DB ID for {entity['entity_id']}. Falling back to INSERT.")
                mode = "INSERT"

            # ==========================================
            # MERGE LOGIC (Fix 1 & 3: Prevent Alias Loss & Vector Dilution)
            # ==========================================
            # Isolate the newly extracted facts BEFORE merging so we don't contaminate the dictionary
            extracted_new_facts = entity.get("facts", []) 
            
            if mode in ["MUTATE", "REINFORCE"] and db_id:
                old_record = next((s for s in skeleton if str(s["id"]) == str(db_id)), None)
                
                if old_record and "payload" in old_record:
                    # A. Merge Facts safely
                    old_facts = {f["attribute_name"]: f for f in old_record["payload"].get("facts", [])}
                    new_facts = {f["attribute_name"]: f for f in extracted_new_facts}
                    old_facts.update(new_facts) 
                    entity["facts"] = list(old_facts.values())

                    # B. Merge Aliases safely
                    old_aliases = set(old_record["payload"].get("aliases", []))
                    new_aliases = set(entity.get("aliases", []))
                    entity["aliases"] = list(old_aliases | new_aliases)

            max_imp = max([f.get("importance", 3) for f in entity.get("facts", [])]) if entity.get("facts") else 3

            op = {
                "mode": mode,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "entity_id": entity["entity_id"],
                "entity_name": entity.get("entity_name") or entity["entity_id"],
                "entity_domain": entity["entity_domain"],
                "entity_type": entity["entity_type"],
                "payload": entity, # Full merged entity for the Database and Vector Embedder
                "extracted_facts": extracted_new_facts, # Isolated new facts for the Dictionary Hits
                "importance": max_imp
            }
            
            if db_id:
                op["db_id"] = str(db_id)
                
            refined_ops.append(op)

        return refined_ops
