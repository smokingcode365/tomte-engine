# tomte/engine/workflows/consolidation.py

import hashlib
import json
import re
from copy import deepcopy

from typing import List, Dict, Any
from tomte.engine.llm.embedder import  get_entity_summary
from tomte.engine.dictionary.service import get_dictionary_service
from tomte.engine.core.schema import TomteMemorySchema
from tomte.engine.core.contextual import ContextualActionResponse
from tomte.engine.llm.prompts.contextual_action import CONTEXTUAL_ACTION_PROMPT
from tomte.engine.core.assertions import MutationPlanResponse, adapt_assertions, apply_mutations, fact_values, project_facts
from tomte.engine.llm.prompts.fact_mutation import MUTATION_SYSTEM_PROMPT
from tomte.engine.core.resolution import ResolutionResponse, CANDIDATE_LIMIT, MIN_RESOLUTION_CONFIDENCE, STRONG_EVIDENCE
from tomte.engine.dictionary.terminology import canonical_attribute, ATTRIBUTE_ALIASES
from tomte.engine.core.extraction import RECENT_CONTEXT_TURNS, ExtractionMemorySchema, build_extraction_input, resolve_extraction, validate_operation_provenance
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
        if not chat_items:
            return False
        batch_id = self._batch_id(tenant_id, user_id, chat_items)
        if self.db.is_consolidation_processed(tenant_id, user_id, batch_id):
            return True  # Idempotent success: the worker may acknowledge Redis.

        extraction_input = build_extraction_input(chat_items)

        # 1. Preparation: Extract text and vectors [cite: 139-140]
        chat_vectors = [m['vector'] for m in chat_items if m.get('vector')]

        if not chat_vectors:
            tomtelog.warn(f"No vectors found for consolidation session: {user_id}")
            return False
        # CRITICAL FIX: Inject the Junk Filter here
        if self.llm.is_junk(extraction_input):
            tomtelog.info(f"Consolidation aborted: Chat log is junk/small-talk for {user_id}.")
            return False
        # --- PHASE 1: Semantic Bootstrapping ---
        # Clusters chat vectors to find relevant dictionary keywords [cite: 137-151]
        tomtelog.debug(f"Phase 1: Bootstrapping dictionary for {user_id}")
        bootstrap_dict = self.dict_service.bootstrap(chat_vectors)
        
        # --- PHASE 2: Generative Extraction ---
        # First LLM Call: Extracts entities using the bootstrap dictionary 
        tomtelog.debug(f"Phase 2: Executing primary extraction")
        extracted_result = self.llm.parse_structured(
            system_prompt=None, 
            user_text=extraction_input,
            schema=ExtractionMemorySchema,
            taxonomy=bootstrap_dict[DictionaryKeywordType.ENTITY],
            lexicon=bootstrap_dict[DictionaryKeywordType.ATTRIBUTE]
        )

        extracted_schema = resolve_extraction(extracted_result, chat_items)

        # --- PHASE 3: Golden Record Reconciliation ---
        # Second LLM Call: Matches extraction against the DB skeleton [cite: 157-159]
        tomtelog.debug(f"Phase 3: Reconciling with Golden Records")
        # Reuse current-message vectors; no extra embedding/model request is needed.
        raw_entities = self._flatten_schema(extracted_schema)
        input_data = json.loads(extraction_input)
        reference = input_data["RECENT_CONTEXT_REFERENCE_ONLY"]
        search_context = "\n".join(item["content"] for item in input_data["CURRENT_BATCH"] + reference)
        vector = [sum(values) / len(chat_vectors) for values in zip(*chat_vectors)]
        if raw_entities:
            candidate_sets = {}
            for entity in raw_entities:
                names = [re.sub(r"\s+", " ", name.strip()).lower()
                         for name in [entity["entity_name"], *entity.get("aliases", [])] if name.strip()]
                candidate_sets[entity["entity_id"]] = self.db.get_resolution_candidates(
                    tenant_id, user_id, names, entity["entity_domain"], vector, search_context, CANDIDATE_LIMIT)
            skeleton = list({str(row["id"]): row for rows in candidate_sets.values() for row in rows}.values())
        
            # This helper determines if we INSERT, MUTATE, or REINFORCE [cite: 164-169]
            refined_ops = self._reconcile_with_golden_records(
                tenant_id,
                user_id,
                raw_entities,
                skeleton,
                reference_context=reference,
                current_batch=input_data["CURRENT_BATCH"], candidate_sets=candidate_sets
            )

        else:
            refined_ops = self._resolve_contextual_actions(tenant_id, user_id, input_data, vector)
            if refined_ops is None:
                return False  # Action unresolved: retain frozen batch, no processed claim.

        refined_ops = self._plan_fact_mutations(refined_ops, input_data, batch_id)
        validate_operation_provenance(refined_ops)

        # Before committing, we must generate vectors for any new or changed entities.
        # These vectors anchor the facts in the Knowledge Graph [cite: 160-171].
        tomtelog.debug(f"Consolidation: Generating embeddings for {len(refined_ops)} operations")
        if refined_ops:
            summaries_to_embed = [get_entity_summary(op['payload']) for op in refined_ops]
            batch_vectors = self.llm.embed_batch(summaries_to_embed)
            for i, op in enumerate(refined_ops):
                op['embedding'] = batch_vectors[i]
                
        # Prepared memory writes and dictionary feedback form one PostgreSQL unit.
        self._commit_refined_operations(refined_ops, tenant_id, user_id, batch_id)

        tomtelog.info(f"Consolidation cycle complete for {user_id}")
        return True

    def _resolve_contextual_actions(self, tenant_id, user_id, input_data, vector):
        """Narrow reference resolver for empty extraction; never creates new entities."""
        current = input_data["CURRENT_BATCH"]
        reference = input_data["RECENT_CONTEXT_REFERENCE_ONLY"]
        context = "\n".join(item["content"] for item in current + reference)
        candidates = self.db.get_resolution_candidates(
            tenant_id, user_id, [], "", vector, context, CANDIDATE_LIMIT)
        offered = {}
        view = []
        for row in candidates[:CANDIDATE_LIMIT]:
            key = str(row["id"])
            assertions = adapt_assertions(row["payload"].get("facts", []), key)
            facts = (project_facts(assertions) + [f for f in assertions if f["assertion_state"] != "ACTIVE"][-15:])[:50]
            offered[key] = (row, {f["fact_id"] for f in facts})
            view.append({"db_id":key, "entity_name":row["entity_name"],
                "entity_domain":row["entity_domain"], "entity_type":row["entity_type"],
                "aliases":row["payload"].get("aliases", []),
                "assertions":[{name:f.get(name) for name in ("fact_id", "attribute_name", "value", "assertion_state")} for f in facts]})
        response = self.llm.parse_structured(system_prompt=CONTEXTUAL_ACTION_PROMPT,
            user_text=json.dumps({"CURRENT_BATCH":current, "REFERENCE_ONLY_CONTEXT":reference,
                "CANDIDATES":view}, ensure_ascii=False), schema=ContextualActionResponse)
        self.last_contextual_route = response.model_dump(mode="json")
        sources = {item["item_id"] for item in current}
        operations = {}
        unresolved = False
        for action in response.actions:
            if not set(action.source_chat_ids) <= sources:
                raise ValueError("Contextual evidence must reference current frozen batch sources")
            if action.replacement_text is not None and not any(action.replacement_text in item["content"] for item in current):
                raise ValueError("Contextual replacement text must quote current evidence")
            if action.target is None:
                unresolved = True
                continue
            key = str(action.target.db_id)
            if key not in offered or action.target_fact_id not in offered[key][1]:
                raise ValueError("Contextual target must name an offered scoped entity/assertion")
            record = offered[key][0]
            anchors = [record["entity_name"], *record["payload"].get("aliases", [])]
            mentioned = any(len(name.strip()) > 2 and re.search(r"\b" + re.escape(name.lower().strip()) + r"\b", context.lower()) for name in anchors)
            if (action.target.resolution_confidence < MIN_RESOLUTION_CONFIDENCE or
                    action.target.evidence not in STRONG_EVIDENCE or not mentioned):
                unresolved = True
                continue
            if key not in operations:
                # Reference preparation asserts no new facts or aliases; stable metadata remains authoritative.
                entity = {**deepcopy(record["payload"]), "facts":[]}
                operations[key] = self._prepare_operation(tenant_id, user_id, entity, record)
                operations[key]["_contextual_fact_ids"] = []
                operations[key]["_contextual_actions"] = []
            operations[key]["_contextual_fact_ids"].append(action.target_fact_id)
            operations[key]["_contextual_actions"].append(action.model_dump(mode="json"))
        return None if unresolved else list(operations.values())

    def _plan_fact_mutations(self, operations, input_data, batch_id):
        """One bounded semantic call after identity acceptance, before any SQL writes."""
        if not operations:
            return []
        prepared = deepcopy(operations)
        offered = {}
        targets = []
        for index, operation in enumerate(prepared):
            key = f"target_{index}"
            scope = operation.get("db_id") or json.dumps([operation["tenant_id"], operation["user_id"], operation["entity_id"]])
            assertions = adapt_assertions(operation["payload"].get("facts", []), scope)
            operation["payload"]["facts"] = assertions
            active = project_facts(assertions)
            inactive = [f for f in assertions if f["assertion_state"] != "ACTIVE"]
            # Bound model history; omitted fact IDs cannot be targeted by a plan.
            selected = (active + inactive[-15:])[:50]
            if "_contextual_fact_ids" in operation:
                selected = [f for f in selected if f["fact_id"] in operation["_contextual_fact_ids"]]
            offered[key] = (operation, scope, {f["fact_id"] for f in selected})
            targets.append({"target_key":key, "identity":{name:operation[name] for name in
                            ("entity_id", "entity_name", "entity_domain", "entity_type")},
                            "incoming_semantics":[{field:f.get(field) for field in ("attribute_name", "value", "status", "source_chat_id", "confidence")}
                                                  for f in operation["extracted_facts"]],
                            "assertions":[{name:f.get(name) for name in ("fact_id", "attribute_name", "value",
                                "assertion_state", "effective_at", "supersedes_fact_id", "corrects_fact_id")}
                                for f in selected], "assertions_truncated":len(selected)<len(assertions)})
            if "_contextual_actions" in operation:
                targets[-1]["contextual_actions"] = operation["_contextual_actions"]
        response = self.llm.parse_structured(system_prompt=MUTATION_SYSTEM_PROMPT,
            user_text=json.dumps({"CURRENT_BATCH":input_data["CURRENT_BATCH"],
                "REFERENCE_ONLY_CONTEXT":input_data["RECENT_CONTEXT_REFERENCE_ONLY"],
                "CANONICAL_ATTRIBUTE_GUIDANCE":{alias:key for key,aliases in ATTRIBUTE_ALIASES.items() for alias in sorted(aliases)},
                "TARGETS":targets}, ensure_ascii=False), schema=MutationPlanResponse)
        plans = {}
        for plan in response.plans:
            if plan.target_key not in offered or plan.target_key in plans:
                raise ValueError("Mutation plans must name unique offered targets")
            plans[plan.target_key] = plan.operations
        self.last_mutation_plan = response.model_dump(mode="json")
        for key, (operation, scope, ids) in offered.items():
            mutations = plans.get(key, [])  # Omitted targets abstain; never fall back to ADD.
            if "_contextual_fact_ids" in operation and not mutations:
                raise ValueError("Resolved contextual action has no mutation; retain frozen batch")
            operation["payload"]["facts"] = apply_mutations(operation["payload"]["facts"], mutations,
                input_data["CURRENT_BATCH"], scope, batch_id, operation["extracted_facts"], ids)
            operation["importance"] = max((f.get("importance",3) for f in project_facts(operation["payload"]["facts"])), default=3)
            if operation.get("db_id"):
                operation["mode"] = "REINFORCE" if mutations and all(m.operation == "REINFORCE" for m in mutations) else "MUTATE"
            operation["extracted_facts"] = [
                {"attribute_name":canonical_attribute(m.attribute_name) if m.attribute_name else
                    next(f["attribute_name"] for f in operation["payload"]["facts"] if f["fact_id"] == m.target_fact_id)}
                for m in mutations]
        # A NEW target with no accepted facts/aliases must not create an empty guessed memory.
        return [op for op in prepared if op.get("db_id") or op["payload"]["facts"]]

    @staticmethod
    def _batch_id(tenant_id: str, user_id: str, chat_items: List[Dict[str, Any]]) -> str:
        """Hash identities, not content; similar later messages have new IDs."""
        session_id = chat_items[0].get("session_id")
        ids = []
        for item in chat_items:
            item_id = item.get("item_id")
            if (not isinstance(item_id, str) or not item_id or
                    not isinstance(session_id, str) or not session_id or
                    item.get("session_id") != session_id or
                    item.get("tenant_id") != tenant_id or item.get("user_id") != user_id):
                raise ValueError("Consolidation requires stable, consistently scoped queue item identities")
            ids.append(item_id)
        if len(set(ids)) != len(ids):
            raise ValueError("Consolidation batch contains duplicate queue item identities")
        identity = json.dumps([tenant_id, user_id, session_id, ids], separators=(",", ":"))
        return hashlib.sha256(identity.encode("utf-8")).hexdigest()

    def _commit_refined_operations(self, refined_ops: List[Dict[str, Any]], tenant_id: str, user_id: str, batch_id: str) -> None:
        """Persist batch claim, prepared memory changes and feedback atomically."""
        validate_operation_provenance(refined_ops)
        with self.db.transaction():
            if not self.db.claim_consolidation_batch(tenant_id, user_id, batch_id):
                return  # Another attempt committed while model work was prepared.
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
    def _reconcile_with_golden_records(self, tenant_id, user_id, extracted, skeleton,
                                      reference_context=None, current_batch=None, candidate_sets=None):
        """Resolve target sets once, then conservatively prepare existing-style operations."""
        if not extracted:
            return []
        ids = [entity["entity_id"] for entity in extracted]
        if len(set(ids)) != len(ids):
            raise ValueError("Temporary extraction IDs must be unique within a batch")
        entities = deepcopy(extracted)
        for entity in entities:
            # Canonicalize without collapsing simultaneous assertions by attribute.
            for fact in entity.get("facts", []):
                fact["attribute_name"] = canonical_attribute(fact["attribute_name"])
        if candidate_sets is None:
            # Direct callers may supply an already bounded set; never send a whole graph.
            candidate_sets = {eid: skeleton[:CANDIDATE_LIMIT] for eid in ids}
        records = {str(row["id"]): deepcopy(row) for row in skeleton}
        candidate_view = {}
        for eid in ids:
            rows = candidate_sets.get(eid, [])[:CANDIDATE_LIMIT]
            candidate_view[eid] = [{
                "id": str(row["id"]), "entity_id": row["entity_id"],
                "name": row["entity_name"], "type": row["entity_type"],
                "domain": row.get("entity_domain"), "aliases": row.get("payload", {}).get("aliases", []),
                "facts": fact_values(row.get("payload", {}).get("facts", [])),
                "candidate_only_similarity": row.get("similarity"),
            } for row in rows]
        reference = [{"role": item["role"], "content": item["content"]}
                     for item in (reference_context or [])[-RECENT_CONTEXT_TURNS:]]
        current = [{key: item[key] for key in ("role", "content", "item_id", "timestamp") if key in item}
                   for item in (current_batch or [])]
        prompt = {"CURRENT_BATCH": current, "REFERENCE_ONLY_CONTEXT": reference,
                  "EXTRACTED_SEMANTIC_INFORMATION": [{
                      "id": e["entity_id"], "name": e.get("entity_name"), "type": e["entity_type"],
                      "domain": e["entity_domain"], "aliases": e.get("aliases", []),
                      "facts": fact_values(e["facts"]),
                  } for e in entities], "CANDIDATES_BY_OBJECT": candidate_view,
                  "CANDIDATE_SET_TRUNCATED": {eid: any(row.get("identity_matches_total", 0) > CANDIDATE_LIMIT
                                                      for row in candidate_sets.get(eid, [])) for eid in ids}}
        response = self.llm.parse_structured(system_prompt=RECONCILER_SYSTEM_PROMPT,
                                            user_text=json.dumps(prompt, ensure_ascii=False), schema=ResolutionResponse)
        self.last_resolution = response.model_dump(mode="json")
        lookup = {}
        for result in response.resolutions:
            if result.extracted_entity_id not in ids or result.extracted_entity_id in lookup:
                raise ValueError("Resolution must reference each extraction object at most once")
            lookup[result.extracted_entity_id] = result
        operations = []
        prepared = {}
        for entity in entities:
            result = lookup.get(entity["entity_id"])
            if result is None or result.kind == "UNRESOLVED":
                tomtelog.info("Identity unresolved: no permanent memory operation")
                continue
            if result.kind == "NEW":
                quote = result.new_entity_evidence
                if not result.explicitly_new or not quote or not any(quote in item["content"] for item in current):
                    tomtelog.info("New identity lacks current introduction evidence: no permanent operation")
                    continue
                operations.append(self._prepare_operation(tenant_id, user_id, entity, None))
                continue
            offered = {str(row["id"]) for row in candidate_sets.get(entity["entity_id"], [])[:CANDIDATE_LIMIT]}
            for target in result.targets:
                if str(target.db_id) not in offered or str(target.db_id) not in records:
                    raise ValueError("Matched Golden Record not found in offered scoped candidates")
            targets = []
            for target in result.targets:
                record = records[str(target.db_id)]
                anchors = [record["entity_name"], *record.get("payload", {}).get("aliases", [])]
                active_text = " ".join(item["content"] for item in current + reference).lower()
                active = any(len(name.strip()) > 2 and re.search(r"\b" + re.escape(name.lower().strip()) + r"\b", active_text)
                             for name in anchors)
                if (target.resolution_confidence >= MIN_RESOLUTION_CONFIDENCE and target.evidence in STRONG_EVIDENCE
                        and (target.evidence not in {"GROUP_EXPLICIT", "RESOLVED_REFERENCE"} or active)):
                    targets.append(target)
            if result.kind == "MATCH_MANY" and any(re.search(r"\bone of\b", item["content"], re.I) for item in current):
                targets = []  # An unspecified individual cannot be expanded into a set.
            if result.kind == "MATCH_MANY" and (not result.applies_equally_to_all or len(targets) < 2
                                               or prompt["CANDIDATE_SET_TRUNCATED"][entity["entity_id"]]):
                tomtelog.info("Target set/update is uncertain: no permanent memory operation")
                continue
            if not targets:
                tomtelog.info("Identity evidence below policy threshold: no permanent operation")
            for target in targets:
                db_id = str(target.db_id)
                record = deepcopy(records[db_id])
                if db_id in prepared:
                    record["payload"] = prepared[db_id]["payload"]
                update = deepcopy(entity)
                if result.kind == "MATCH_MANY":
                    # Apply facts, not the group label/type/aliases, to individual identities.
                    update.update(entity_name=record["entity_name"], entity_type=record["entity_type"],
                                  entity_domain=record["entity_domain"], aliases=record["payload"].get("aliases", []))
                op = self._prepare_operation(tenant_id, user_id, update, record)
                if db_id in prepared:
                    previous = prepared[db_id]
                    previous["payload"] = op["payload"]
                    previous["extracted_facts"] += op["extracted_facts"]
                    if op["mode"] == "MUTATE":
                        previous["mode"] = "MUTATE"
                    previous.update(entity_name=op["entity_name"], entity_type=op["entity_type"],
                                    entity_domain=op["entity_domain"], importance=op["importance"])
                else:
                    operations.append(op)
                    prepared[db_id] = op
        return operations

    @staticmethod
    def _prepare_operation(tenant_id, user_id, entity, record):
        """Existing fact merge/history semantics; identity decisions never choose a mode."""
        payload = deepcopy(entity)
        incoming = deepcopy(entity.get("facts", []))
        mode = "INSERT"
        if record is not None:
            old = deepcopy(record.get("payload", {}).get("facts", []))
            values = {f["attribute_name"]: (f["value"], f.get("status", "active")) for f in old}
            mode = "REINFORCE" if all(values.get(f["attribute_name"]) == (f["value"], f.get("status", "active"))
                                      for f in incoming) else "MUTATE"
            # Normal matched updates preserve identity/classification. Explicit
            # corrections would require a separate, deliberate operation.
            for key in ("entity_id", "entity_name", "entity_domain", "entity_type"):
                payload[key] = record[key]
            # Identity preparation does not decide assertion truth transitions.
            payload["facts"] = old
            aliases = set(record.get("payload", {}).get("aliases", [])) | set(entity.get("aliases", []))
            incoming_name = entity.get("entity_name")
            if incoming_name and incoming_name != record["entity_name"]:
                aliases.add(incoming_name)
            payload["aliases"] = sorted(aliases)
        op = {"mode": mode, "tenant_id": tenant_id, "user_id": user_id,
              "entity_id": payload["entity_id"], "entity_name": payload.get("entity_name") or payload["entity_id"],
              "entity_type": payload["entity_type"], "entity_domain": payload["entity_domain"],
              "payload": payload, "extracted_facts": incoming,
              "importance": max((f.get("importance", 3) for f in payload.get("facts", [])), default=3)}
        if record is not None:
            op["db_id"] = str(record["id"])
            op["_expected_payload"] = deepcopy(record["payload"])
        else:
            op["payload"]["facts"] = []  # NEW still requires validated ADD proposals.
        return op
