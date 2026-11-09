# tomte/engine/workflows/consolidation.py

import hashlib
import json
import re
from collections import Counter
from copy import deepcopy
from uuid import uuid4

from typing import List, Dict, Any
from tomte.engine.llm.embedder import get_entity_summary, get_resolution_query, changed_search_entries, changed_retrieval_entries, embed_texts
from tomte.engine.core.memory import public_fact, public_entity
from tomte.engine.core.enums import MemoryDomain
from tomte.engine.core.schema import DOMAIN_DESCRIPTIONS
from tomte.engine.dictionary.service import get_dictionary_service
from tomte.engine.dictionary.alignment import align_terms
from tomte.engine.core.contextual import ContextualActionResponse
from tomte.engine.llm.prompts.contextual_action import CONTEXTUAL_ACTION_PROMPT
from tomte.engine.core.assertions import MutationPlanResponse, InvalidMutation, validate_facts, apply_mutations, fact_values
from tomte.engine.llm.prompts.fact_mutation import MUTATION_SYSTEM_PROMPT
from tomte.engine.core.resolution import ResolutionResponse, CANDIDATE_LIMIT, compatible_identity_domains
from tomte.engine.core.candidate_filter import (
    CandidateFilterResponse, CandidateFilterUnavailable, IDENTITY_CANDIDATE_LIMIT, retained_candidate_ids,
)
from tomte.engine.llm.prompts.candidate_filter import CANDIDATE_FILTER_PROMPT
from tomte.engine.core.extraction import RECENT_CONTEXT_TURNS, ExtractionMemorySchema, ResolvedExtractionMemory, build_extraction_input, resolve_extraction, validate_operation_provenance, source_timestamps
from tomte.engine import tomtelog, Tomte
from tomte.engine.llm.prompts.consolidation_golden_record import RECONCILER_SYSTEM_PROMPT

class ConsolidationWorkflow:
    def __init__(
        self, db=None, llm=None, control=None
    ):
        self.db = db if db is not None else Tomte.get_db()
        self.llm = llm if llm is not None else Tomte.get_llm()
        self.control = control
        from tomte.engine.dictionary.service import DictionaryService
        self.dict_service = DictionaryService(self.db, self.llm)

    def _parse_proposals(self, **kwargs):
        """A failed model format cannot invalidate another independent stage."""
        from pydantic_ai.exceptions import UnexpectedModelBehavior
        schema = kwargs["schema"]
        unusable = False
        try:
            response = self.llm.parse_structured(**kwargs)
        except UnexpectedModelBehavior:
            # Providers have already spent their single output-repair allowance.
            # No transport, persistence, cancellation or capacity exceptions here.
            response = schema.model_validate({name: [] for name in schema.proposal_collections})
            unusable = True
        if not hasattr(self, "last_proposal_recovery"):
            self.last_proposal_recovery = {}
        self.last_proposal_recovery[schema.__name__] = {**response.recovery_summary(), "unusable_response": unusable}
        return response
    # ==========================================
    # CONSOLIDATION CYCLE
    # ==========================================

    def run_consolidation_cycle(self, tenant_id: str, user_id: str, chat_items: List[Dict[str, Any]]) -> bool:
        """
        Extract, align vocabulary, resolve identity, plan and commit scoped memory.
        """
        if not chat_items:
            return False
        batch_id = self._batch_id(tenant_id, user_id, chat_items)
        unresolved_proposals = []
        if self.db.is_consolidation_processed(tenant_id, user_id, batch_id):
            return True  # Idempotent success: the worker may acknowledge Redis.

        extraction_input = build_extraction_input(chat_items)
        from tomte.engine.core.capacity import check_capacity
        check_capacity(extraction_input)

        # Assistant context is not user evidence and needs no embedding.
        user_items = [m for m in chat_items if m["role"] == "user"]
        if not user_items:
            self._commit_refined_operations([], tenant_id, user_id, batch_id)
            return True
        # Storage-only ingestion intentionally queues no vectors. Generate only
        # missing USER embeddings here; leave the frozen queue snapshot untouched
        # so retries and Redis acknowledgement still compare the original bytes.
        missing = [m for m in user_items if not m.get("vector")]
        generated = iter(embed_texts(self.llm, [m["content"] for m in missing]) if missing else [])
        chat_vectors = [m.get("vector") or next(generated) for m in user_items]
        # Every user batch reaches extraction; an empty result is valid abstention.
        # Raw extraction is authoritative; dictionary alignment happens only after
        # deterministic current-batch provenance validation.
        tomtelog.debug(f"Phase 2: Executing primary extraction")
        extracted_result = self._parse_proposals(
            system_prompt=None, 
            user_text=extraction_input,
            schema=ExtractionMemorySchema,
        )

        extracted_schema = resolve_extraction(extracted_result, chat_items)

        # --- PHASE 3: Golden Record Reconciliation ---
        # Identity resolution selects only offered scoped candidates.
        tomtelog.debug(f"Phase 3: Reconciling with Golden Records")
        # Resolve each extracted subject using its own meaning, not other batch topics.
        raw_entities = self._flatten_schema(extracted_schema)
        input_data = json.loads(extraction_input)
        if raw_entities:
            alignment = align_terms(raw_entities, input_data["CURRENT_BATCH"], self.llm, self.db)
            self.last_terminology_metrics = alignment.metrics
            raw_entities = alignment.entities
            for entity, evidence in zip(raw_entities, alignment.evidence):
                entity["_dictionary_evidence"] = evidence
        reference = input_data["RECENT_CONTEXT_REFERENCE_ONLY"]
        search_context = "\n".join(item["content"] for item in input_data["CURRENT_BATCH"] + reference)
        vector = [sum(values) / len(chat_vectors) for values in zip(*chat_vectors)]
        correction_context = []
        if raw_entities:
            entity_vectors = embed_texts(self.llm, [get_resolution_query(entity) for entity in raw_entities])
            candidate_sets = {}
            for entity, entity_vector in zip(raw_entities, entity_vectors):
                names = [re.sub(r"\s+", " ", name.strip()).lower()
                         for name in [entity["entity_name"], *entity.get("aliases", [])] if name.strip()]
                candidate_sets[entity["entity_id"]] = self.db.get_resolution_candidates(
                    tenant_id, user_id, names, entity["entity_domain"], entity_vector, search_context, self._identity_candidate_limit())
            skeleton = list({str(row["id"]): row for rows in candidate_sets.values() for row in rows}.values())
            correction_context = skeleton
        
            # Prepare targets; the mutation planner decides their fact changes.
            refined_ops = self._reconcile_with_golden_records(
                tenant_id,
                user_id,
                raw_entities,
                skeleton,
                reference_context=reference,
                current_batch=input_data["CURRENT_BATCH"], candidate_sets=candidate_sets,
                unresolved_proposals=unresolved_proposals
            )

        else:
            refined_ops = []
        from tomte.engine.workflows.unresolved import prepare_unresolved_bindings
        refined_ops, pending = prepare_unresolved_bindings(
            self, tenant_id, user_id, input_data, vector, refined_ops)
        contextual = self._resolve_contextual_actions(tenant_id, user_id, input_data, vector, correction_context, refined_ops)
        by_id = {op.get("db_id"): op for op in refined_ops if op.get("db_id")}
        for op in contextual:
            if op["db_id"] in by_id:
                by_id[op["db_id"]].setdefault("_contextual_actions", []).extend(op["_contextual_actions"])
            else:
                refined_ops.append(op)

        refined_ops = self._plan_fact_mutations(refined_ops, input_data, batch_id, correction_context)
        from tomte.engine.core.unresolved import retained_operations, resolution_operations
        refined_ops += resolution_operations(refined_ops, pending)
        refined_ops += retained_operations(unresolved_proposals, input_data["CURRENT_BATCH"],
                                           tenant_id, user_id, batch_id)
        validate_operation_provenance(refined_ops)

        # Before committing, we must generate vectors for any new or changed entities.
        # Every current/history item receives its own searchable chunks.
        tomtelog.debug(f"Consolidation: Generating embeddings for {len(refined_ops)} operations")
        for op in refined_ops:
            if not op.get("_semantic_changed", True):
                continue
            entries = changed_search_entries(op["payload"], op.get("_expected_payload"))
            retrieval_entries = changed_retrieval_entries(op["payload"], op.get("_expected_payload"))
            vectors = embed_texts(self.llm, [get_entity_summary(op["payload"]),
                *[e["text"] for e in entries], *[e["text"] for e in retrieval_entries]])
            op["embedding"] = vectors[0]
            op["search_entries"] = [{**{k: v for k, v in entry.items() if k != "text"}, "embedding": vector}
                                    for entry, vector in zip(entries, vectors[1:1 + len(entries)])]
            op["retrieval_entries"] = [{**{k: v for k, v in entry.items() if k != "text"}, "embedding": vector}
                                    for entry, vector in zip(retrieval_entries, vectors[1 + len(entries):])]

        # Memory and its replay marker commit before optional dictionary feedback.
        self._commit_refined_operations(refined_ops, tenant_id, user_id, batch_id)

        tomtelog.info(f"Consolidation cycle complete for {user_id}")
        return True

    def _resolve_contextual_actions(self, tenant_id, user_id, input_data, vector, correction_context=None, prepared=()):
        """Resolve remaining references/mentions, including mixed-message batches."""
        current = input_data["CURRENT_BATCH"]
        reference = input_data["RECENT_CONTEXT_REFERENCE_ONLY"]
        context = "\n".join(item["content"] for item in current + reference)
        candidates = self.db.get_resolution_candidates(tenant_id, user_id, [], "", vector, context, CANDIDATE_LIMIT)
        if correction_context is not None:
            correction_context.extend(candidates[:CANDIDATE_LIMIT])
        offered, view = {}, []
        for row in candidates[:CANDIDATE_LIMIT]:
            if row['payload']['entity_domain'] == MemoryDomain.UNRESOLVED:
                continue
            key = str(row["id"])
            facts = validate_facts(row["payload"].get("facts", []))[:50]
            offered[key] = (row, {f["fact_id"] for f in facts})
            view.append({"db_id": key, "entity_name": row["entity_name"], "entity_domain": row["entity_domain"],
                         "entity_type": row["entity_type"], "aliases": row["payload"].get("aliases", []),
                         "facts": [public_fact(f, []) for f in facts]})
            if row["payload"].get("first_mentioned_at") is not None:
                view[-1]["first_mentioned_at"] = row["payload"]["first_mentioned_at"]
        if not view:
            return []
        response = self._parse_proposals(system_prompt=CONTEXTUAL_ACTION_PROMPT,
            user_text=json.dumps({"CURRENT_BATCH": current, "REFERENCE_ONLY_CONTEXT": reference,
                "ALREADY_RESOLVED_PROPOSALS": [{"db_id": op.get("db_id"), "incoming": op["extracted_facts"]} for op in prepared],
                "CANDIDATES": view}, ensure_ascii=False), schema=ContextualActionResponse)
        self.last_contextual_route = response.model_dump(mode="json")
        sources = {item["item_id"] for item in current}
        user_sources = {item["item_id"]: item["content"] for item in current if item["role"] == "user"}
        operations = {}
        for action in response.actions:
            if not set(action.source_chat_ids) <= sources:
                continue
            if action.target is None:
                continue
            key = str(action.target.db_id)
            if key not in offered or action.target_fact_id not in offered[key][1]:
                continue
            if not set(action.source_chat_ids) <= user_sources.keys():
                continue
            if not any(action.evidence_quote in user_sources[s] for s in action.source_chat_ids):
                continue
            if key not in operations:
                record = offered[key][0]
                entity = {**deepcopy(record["payload"]), "facts": []}
                operations[key] = self._prepare_operation(tenant_id, user_id, entity, record)
                operations[key]["_contextual_actions"] = []
            operations[key]["_contextual_actions"].append(action.model_dump(mode="json"))
        return list(operations.values())

    def _plan_fact_mutations(self, operations, input_data, batch_id, correction_context=()):
        """One semantic plan; deterministic mechanics never turn abstention into writes."""
        if not operations:
            return []
        offered, targets = {}, []
        for index, operation in enumerate(deepcopy(operations)):
            key = f"target_{index}"
            scope = operation.get("db_id") or operation["reserved_id"]
            facts = validate_facts(operation["payload"].get("facts", []))
            operation["payload"]["facts"] = facts
            operation["_contextual_actions"] = [dict(action, contextual_action_key=f"action_{i}")
                for i, action in enumerate(operation.get("_contextual_actions", []))]
            requested = {a["target_fact_id"] for a in operation.get("_contextual_actions", [])}
            selected = sorted(facts, key=lambda f: f["fact_id"] not in requested)[:50]
            history = {f["fact_id"]: list(range(max(0, len(f["history_values"]) - 15), len(f["history_values"]))) for f in selected}
            fact_view = []
            for fact in selected:
                item = public_fact(fact, history[fact["fact_id"]])
                for entry, source_index in zip(item["history_values"], history[fact["fact_id"]]):
                    entry["history_index"] = source_index
                fact_view.append(item)
            offered[key] = (operation, scope, {f["fact_id"] for f in selected}, history)
            operation["_required_groups"] = {f["_group_key"]: f["_group_targets"] for f in operation["extracted_facts"] if f.get("_group_key")}
            targets.append({"target_key": key, "identity": {name: operation[name] for name in
                ("entity_id", "entity_name", "entity_domain", "entity_type")},
                "incoming_semantics": operation["extracted_facts"], "facts": fact_view,
                "facts_truncated": len(selected) < len(facts),
                "contextual_actions": operation.get("_contextual_actions", [])})
            if operation["payload"].get("first_mentioned_at") is not None:
                targets[-1]["identity"]["first_mentioned_at"] = operation["payload"]["first_mentioned_at"]
        arguments = dict(system_prompt=MUTATION_SYSTEM_PROMPT,
            user_text=json.dumps({"CURRENT_BATCH": input_data["CURRENT_BATCH"],
                "REFERENCE_ONLY_CONTEXT": input_data["RECENT_CONTEXT_REFERENCE_ONLY"], "TARGETS": targets}, ensure_ascii=False),
            schema=MutationPlanResponse)
        if any(f.get('stored_unresolved_id') for op in operations for f in op['extracted_facts']):
            from tomte.engine.core.capacity import CapacityExceeded, check_provider_capacity
            try:
                check_provider_capacity('parse_structured', arguments)
            except CapacityExceeded:
                remaining = deepcopy(operations)
                for op in remaining:
                    op['extracted_facts'] = [f for f in op['extracted_facts'] if not f.get('stored_unresolved_id')]
                remaining = [op for op in remaining if op['extracted_facts'] or op.get('_contextual_actions')]
                tomtelog.info('Deferred unresolved resolution to preserve current-batch mutation capacity')
                return self._plan_fact_mutations(remaining, input_data, batch_id, correction_context)
        response = self._parse_proposals(**arguments)
        from datetime import datetime, timezone
        from tomte.engine.core.proposals import at_path
        from tomte.engine.core.relationships import incomplete_groups

        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        message_times = source_timestamps(input_data["CURRENT_BATCH"])
        counts = Counter(plan.target_key for plan in response.plans)
        plans = {plan.target_key: (i, plan.operations) for i, plan in enumerate(response.plans)
                 if plan.target_key in offered and counts[plan.target_key] == 1}
        self.last_mutation_plan = response.model_dump(mode="json")
        rejected_plans, blocked = set(), {}

        def tokens(raw, position):
            data = raw.model_dump(mode="json") if hasattr(raw, "model_dump") else raw
            if not isinstance(data, dict):
                return {("slot", position)}
            keys = {(name, data[name]) for name in ("target_fact_id", "incoming_fact_key", "contextual_action_key")
                    if isinstance(data.get(name), str) and data[name]}
            return keys or {("slot", position)}

        for path, raw in response.rejected_items():
            parent = at_path(response._raw, path[:2])
            key = parent.get("target_key") if isinstance(parent, dict) else None
            if not isinstance(key, str) or key not in offered:
                continue
            if len(path) == 2:
                rejected_plans.add(key)
            else:
                blocked.setdefault(key, set()).update(tokens(raw, path[-1]))

        units_by_target = {}
        skipped_units = 0
        for key, (operation, scope, ids, history) in offered.items():
            if key in rejected_plans:
                continue
            plan_index, mutations = plans.get(key, (0, []))
            proposals = {f["incoming_fact_key"]: f for f in operation["extracted_facts"]}
            has_group = any(f.get("_group_key") for f in proposals.values())
            units = []
            for i, mutation in enumerate(mutations):
                position = response.original_index("plans", plan_index, "operations", i)
                keys = tokens(mutation, position)
                group = proposals.get(mutation.incoming_fact_key, {}).get("_group_key")
                unit = {"tokens": keys, "items": [(position, mutation)], "groups": {group} if group else set()}
                # Same fact or same incoming proposal forms one dependent unit.
                overlaps = [u for u in units if u["tokens"] & keys]
                for other in overlaps:
                    unit["tokens"] |= other["tokens"]
                    unit["items"] += other["items"]
                    unit["groups"] |= other["groups"]
                    units.remove(other)
                units.append(unit)
            usable = []
            for unit in units:
                unit["items"].sort(key=lambda pair: pair[0])
                # A missing key must not bypass a shared proposal. An explicitly
                # routed contextual action can be independent, even in that chat.
                needs_key = has_group and any(m.incoming_fact_key is None and m.contextual_action_key is None
                                              for _, m in unit["items"])
                if unit["tokens"] & blocked.get(key, set()) or needs_key:
                    skipped_units += 1
                    continue
                try:
                    apply_mutations(operation["payload"]["facts"], [m for _, m in unit["items"]],
                        input_data["CURRENT_BATCH"], scope, batch_id, operation["extracted_facts"], ids,
                        offered_history=history, recorded_at=stamp,
                        contextual_actions=operation["_contextual_actions"],
                        operation_indices=[i for i, _ in unit["items"]])
                except InvalidMutation:
                    skipped_units += 1
                    continue
                usable.append(unit)
            units_by_target[key] = usable

        def build_operations():
            accepted = []
            for key, units in units_by_target.items():
                if not units:
                    continue
                original, scope, ids, history = offered[key]
                operation = deepcopy(original)
                old = operation["payload"]["facts"]
                items = sorted((item for unit in units for item in unit["items"]), key=lambda pair: pair[0])
                mutations = [m for _, m in items]
                acknowledged = [f for f in operation['extracted_facts'] if f.get('stored_unresolved_id')
                                and any(m.incoming_fact_key == f['incoming_fact_key'] for m in mutations)]
                operation['_accepted_unresolved_ids'] = sorted({f['stored_unresolved_id'] for f in acknowledged})
                operation["payload"]["facts"] = apply_mutations(old, mutations, input_data["CURRENT_BATCH"],
                    scope, batch_id, operation["extracted_facts"], ids, offered_history=history,
                    contextual_actions=operation["_contextual_actions"],
                    recorded_at=stamp, operation_indices=[i for i, _ in items])
                if operation["payload"]["entity_domain"] == MemoryDomain.THOUGHTS:
                    # Only accepted USER-backed operations establish this date.
                    # Rebuilding after group rejection excludes discarded evidence.
                    first = min(message_times[source] for mutation in mutations for source in mutation.source_chat_ids)
                    first = min([first, *[f['reported_at'] for f in acknowledged]])
                    previous = operation["payload"].get("first_mentioned_at")
                    operation["payload"]["first_mentioned_at"] = min(previous, first) if previous else first
                groups = {group for unit in units for group in unit["groups"]}
                operation["_required_groups"] = {f["_group_key"]: f["_group_targets"]
                    for f in operation["extracted_facts"] if f.get("_group_key") in groups}
                # Successful keyed operations acknowledge application of the
                # proposal. The model owns meaning; text equality cannot judge
                # paraphrases, already-known information or historical updates.
                operation["_accepted_group_keys"] = sorted(groups)
                by_id = {f["fact_id"]: f for f in operation["payload"]["facts"]}
                operation["importance"] = max((f["_importance"] for f in operation["payload"]["facts"]), default=3)
                operation["_semantic_changed"] = (not operation.get("_expected_payload") or
                    public_entity(operation["_expected_payload"]) != public_entity(operation["payload"]))
                if operation.get("db_id"):
                    operation["mode"] = "MUTATE" if operation["_semantic_changed"] else "REINFORCE"
                attributes = {m.attribute_name or by_id[m.target_fact_id]["attribute_name"]
                              for m in mutations if m.operation != "MENTION"}
                operation["extracted_facts"] = [{"attribute_name": name} for name in attributes]
                operation["_dictionary_evidence"] = [e for e in operation.get("_dictionary_evidence", [])
                    if attributes and (e["keyword_type"] == "ENTITY" or
                        (e["keyword_type"] == "ATTRIBUTE" and e["keyword"] in attributes) or
                        (e["keyword_type"] == "FIELD" and any(f["attribute_name"] == e.get("attribute_name")
                            and e["keyword"] in f["value"] for f in operation["payload"]["facts"])))]
                accepted.append(operation)
            return accepted

        skipped_groups = set()
        while True:
            accepted = build_operations()
            failed = incomplete_groups(accepted)
            if not failed:
                break
            skipped_groups |= failed
            removed = 0
            for key, units in units_by_target.items():
                keep = [unit for unit in units if not unit["groups"] & failed]
                removed += len(units) - len(keep)
                units_by_target[key] = keep
            if not removed:
                raise RuntimeError("Incomplete group did not identify a dependent mutation unit")
            skipped_units += removed
        self.last_mutation_recovery = {**response.recovery_summary(), "skipped_units": skipped_units,
            "skipped_groups": len(skipped_groups), "accepted_entities": len(accepted)}
        return accepted


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
        """Commit authoritative memory/marker, then best-effort optional feedback."""
        validate_operation_provenance(refined_ops)
        from tomte.engine.core.unresolved import validate_transitions
        validate_transitions(refined_ops)
        if getattr(self, 'control', None) is not None:
            self.control.check()
        with self.db.transaction():
            if not self.db.claim_consolidation_batch(tenant_id, user_id, batch_id):
                return  # Another attempt committed while model work was prepared.
            if not self.db.commit_graph_updates(refined_ops):
                raise RuntimeError("Consolidation memory write failed")
            if getattr(self, 'control', None) is not None:
                self.control.check()
        # Its own transaction and bounded dictionary lock protect optional writes.
        # A failed feedback transaction must never replay already committed memory.
        try:
            if getattr(self, 'control', None) is not None:
                self.control.check()
            self.dict_service.apply_feedback(refined_ops)
            self.last_dictionary_feedback = 'applied'
        except Exception:
            self.last_dictionary_feedback = 'failed'
            try:
                tomtelog.warn('Optional dictionary feedback failed; memory remains committed')
            except Exception:
                pass  # A failing log sink cannot turn an optional failure into replay.

    # ==========================================
    # INTERNAL HELPERS
    # ==========================================

    def _flatten_schema(self, schema: ResolvedExtractionMemory) -> List[Dict[str, Any]]:
        """Converts the multi-domain Pydantic schema into a flat list of entities."""
        all_entities = []
        # Domain containers are structural keys, not a business vocabulary.
        for domain in DOMAIN_DESCRIPTIONS:
            domain_list = getattr(schema, domain)
            for entity in domain_list:
                all_entities.append(entity.model_dump(mode='json'))
        return all_entities

    # ==========================================
    # RECONCILIATION HELPER
    # ==========================================
    def _identity_candidate_limit(self):
        return IDENTITY_CANDIDATE_LIMIT if getattr(self.llm, "supports_candidate_filter", False) is True else CANDIDATE_LIMIT

    @staticmethod
    def _identity_candidate_view(record):
        view = {
            "id": str(record["id"]), "entity_id": record["entity_id"],
            "name": record["entity_name"], "type": record["entity_type"],
            "domain": record.get("entity_domain"), "aliases": record.get("payload", {}).get("aliases", []),
            "facts": fact_values(record.get("payload", {}).get("facts", [])),
        }
        if record.get("payload", {}).get("first_mentioned_at") is not None:
            view["first_mentioned_at"] = record["payload"]["first_mentioned_at"]
        return view

    def _filter_identity_candidates(self, entities, candidate_sets, records, current, reference):
        """Optional semantic pruning; Python checks labels, never word meaning."""
        from pydantic import ValidationError
        from pydantic_ai.exceptions import UnexpectedModelBehavior
        self.last_candidate_filter = {}
        if getattr(self.llm, "supports_candidate_filter", False) is not True:
            return candidate_sets
        filtered = {}
        for entity in entities:
            eid = entity["entity_id"]
            rows = candidate_sets[eid]
            filtered[eid] = rows
            outcome = "empty"
            if rows:
                labels = {f"c{i}": str(row["id"]) for i, row in enumerate(rows, 1)}
                candidates = []
                for label, row in zip(labels, rows):
                    view = self._identity_candidate_view(records.get(str(row["id"]), row))
                    view.pop("id")
                    view.pop("entity_id")
                    candidates.append({"candidate_label": label, **view})
                body = {"CURRENT_BATCH": current, "REFERENCE_ONLY_CONTEXT": reference,
                    "EXTRACTED_OBJECT": {"name": entity.get("entity_name"), "type": entity["entity_type"],
                        "domain": entity["entity_domain"], "aliases": entity.get("aliases", []),
                        "facts": fact_values(entity["facts"]), "mention_context": entity.get("mention_context")},
                    "CANDIDATES": candidates}
                try:
                    response = self.llm.parse_structured(system_prompt=CANDIDATE_FILTER_PROMPT,
                        user_text=json.dumps(body, ensure_ascii=False), schema=CandidateFilterResponse)
                    kept = retained_candidate_ids(labels, response)
                    outcome = "invalid_response" if kept is None else "filtered"
                    if kept is not None:
                        filtered[eid] = [row for row in rows if str(row["id"]) in kept]
                except (CandidateFilterUnavailable, ValidationError, UnexpectedModelBehavior, TimeoutError):
                    outcome = "unavailable"
                # Metadata only: no prompts, values, labels or UUIDs in logs.
                if outcome != "filtered":
                    tomtelog.warn(f"Candidate filter {outcome}; retaining original candidate set")
            self.last_candidate_filter[eid] = {
                "offered": len(rows), "retained": len(filtered[eid]), "outcome": outcome,
            }
        return filtered

    def _reconcile_with_golden_records(self, tenant_id, user_id, extracted, skeleton,
                                      reference_context=None, current_batch=None, candidate_sets=None,
                                      unresolved_proposals=None):
        """Resolve target sets once, then conservatively prepare existing-style operations."""
        if not extracted:
            return []
        ids = [entity["entity_id"] for entity in extracted]
        if len(set(ids)) != len(ids):
            raise ValueError("Temporary extraction IDs must be unique within a batch")
        entities = deepcopy(extracted)
        if candidate_sets is None:
            # Direct callers may supply an already bounded set; never send a whole graph.
            candidate_sets = {eid: skeleton for eid in ids}
        records = {str(row["id"]): deepcopy(row) for row in skeleton}
        limit = self._identity_candidate_limit()
        candidate_sets = {entity["entity_id"]: [row for row in candidate_sets.get(entity["entity_id"], [])
            if compatible_identity_domains(entity["entity_domain"], row["payload"]["entity_domain"])][:limit]
            for entity in entities}
        # Preserve pre-filter retrieval metadata even if every row is dropped.
        truncated = {eid: any(row.get("identity_matches_total", 0) > limit
                             for row in candidate_sets[eid]) for eid in ids}
        counts_before = {eid: len(candidate_sets[eid]) for eid in ids}
        reference = [{"role": item["role"], "content": item["content"]}
                     for item in (reference_context or [])[-RECENT_CONTEXT_TURNS:]]
        current = [{key: item[key] for key in ("role", "content", "item_id", "timestamp", "message_time_utc") if key in item}
                   for item in (current_batch or [])]
        candidate_sets = self._filter_identity_candidates(entities, candidate_sets, records, current, reference)
        candidate_view = {}
        candidates_by_id = {}
        for eid in ids:
            rows = candidate_sets.get(eid, [])
            candidate_view[eid] = []
            for row in rows:
                key = str(row["id"])
                candidate_view[eid].append({"id": key, "candidate_only_similarity": row.get("similarity")})
                if key not in candidates_by_id:
                    # Reuse the snapshot used to prepare a matched operation.
                    record = records.get(key, row)
                    candidates_by_id[key] = self._identity_candidate_view(record)
        prompt = {"CURRENT_BATCH": current, "REFERENCE_ONLY_CONTEXT": reference,
                  "EXTRACTED_SEMANTIC_INFORMATION": [{
                      "id": e["entity_id"], "name": e.get("entity_name"), "type": e["entity_type"],
                      "domain": e["entity_domain"], "aliases": e.get("aliases", []),
                      "facts": fact_values(e["facts"]), "fact_proposals": e["facts"],
                      "mention_context": e.get("mention_context"),
                  } for e in entities], "CANDIDATES_BY_ID": candidates_by_id,
                  "CANDIDATES_BY_OBJECT": candidate_view,
                  "CANDIDATE_SET_TRUNCATED": truncated,
                  "CANDIDATES_FILTERED": {eid: len(candidate_sets[eid]) < counts_before[eid] for eid in ids}}
        response = self._parse_proposals(system_prompt=RECONCILER_SYSTEM_PROMPT,
                                            user_text=json.dumps(prompt, ensure_ascii=False), schema=ResolutionResponse)
        self.last_resolution = response.model_dump(mode="json")
        lookup = {}
        counts = Counter(result.extracted_entity_id for result in response.resolutions)
        rejected_ids = {raw.get("extracted_entity_id") for _, raw in response.rejected_items()
                        if isinstance(raw, dict) and isinstance(raw.get("extracted_entity_id"), str)}
        for result in response.resolutions:
            if (result.extracted_entity_id not in ids or counts[result.extracted_entity_id] != 1
                    or result.extracted_entity_id in rejected_ids):
                continue
            lookup[result.extracted_entity_id] = result
        operations = []
        prepared = {}
        for entity in entities:
            result = lookup.get(entity["entity_id"])
            if result is None:
                continue
            if result.kind == "UNRESOLVED":
                if unresolved_proposals is not None:
                    unresolved_proposals.append({"entity": deepcopy(entity), "reason": result.reasoning})
                tomtelog.info("Identity unresolved: retain supported claim separately from entity updates")
                continue
            if result.kind == "NEW":
                quote = result.new_entity_evidence
                if not result.explicitly_new or not quote or not any(
                        quote in item["content"] for item in current if item["role"] == "user"):
                    tomtelog.info("New identity lacks current introduction evidence: no permanent operation")
                    continue
                aliases = self._approved_aliases(result.accepted_aliases, entity, current, result.alias_evidence)
                op = self._prepare_operation(tenant_id, user_id, entity, None, aliases)
                op["_single_handles"] = [entity["entity_id"]]
                operations.append(op)
                continue
            offered = {str(row["id"]) for row in candidate_sets.get(entity["entity_id"], [])}
            if any(str(target.db_id) not in offered or str(target.db_id) not in records for target in result.targets):
                continue
            if any(not compatible_identity_domains(entity["entity_domain"], records[str(target.db_id)]["payload"]["entity_domain"])
                   for target in result.targets):
                continue
            # The model selects identity; Python checks references and write scope.
            targets = result.targets
            if result.kind == "MATCH_MANY" and (not result.applies_equally_to_all
                                               or not result.group_evidence_quote
                                               or not any(item.get("item_id") == result.group_source_chat_id and item["role"] == "user"
                                                          and result.group_evidence_quote in item["content"] for item in current)):
                tomtelog.info("Group update lacks shared applicability or current user provenance")
                continue
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
                aliases = self._approved_aliases(result.accepted_aliases, entity, current, result.alias_evidence) if result.kind == "MATCH_ONE" else []
                op = self._prepare_operation(tenant_id, user_id, update, record, aliases)
                op["_single_handles"] = [entity["entity_id"]] if result.kind == "MATCH_ONE" else []
                if result.kind == "MATCH_MANY":
                    for fact in op["extracted_facts"]:
                        fact["_group_targets"] = [str(t.db_id) for t in targets]
                        fact["_group_handles"] = result.group_member_handles
                if db_id in prepared:
                    previous = prepared[db_id]
                    previous["payload"] = op["payload"]
                    previous["extracted_facts"] += op["extracted_facts"]
                    previous.setdefault("_single_handles", []).extend(op["_single_handles"])
                    previous.setdefault("_dictionary_evidence", []).extend(op.get("_dictionary_evidence", []))
                    if op["mode"] == "MUTATE":
                        previous["mode"] = "MUTATE"
                    previous.update(entity_name=op["entity_name"], entity_type=op["entity_type"],
                                    entity_domain=op["entity_domain"], importance=op["importance"])
                else:
                    operations.append(op)
                    prepared[db_id] = op
        from tomte.engine.core.relationships import prepare_proposals
        operations, self.last_identity_map = prepare_proposals(operations)
        return operations

    @staticmethod
    def _approved_aliases(accepted, entity, current, alias_evidence=()):
        """Verify offered spelling and exact USER quote provenance, not alias meaning."""
        if not accepted:
            return []
        offered = [entity.get("entity_name"), *entity.get("aliases", [])]
        user_sources = {item["item_id"]: item["content"] for item in current if item["role"] == "user"}
        approved = []
        for alias in accepted:
            citations = [citation for citation in alias_evidence if citation.alias == alias]
            if not alias.strip() or alias not in offered or len(citations) != 1:
                tomtelog.info("Alias citation or offered spelling not authorized: retain supported fact operations")
                continue
            citation = citations[0]
            text = user_sources.get(citation.source_chat_id)
            if text is not None and citation.evidence_quote.strip() and citation.evidence_quote in text and alias not in approved:
                approved.append(alias)
            else:
                tomtelog.info("Alias source quotation not authorized: retain supported fact operations")
        return approved

    @staticmethod
    def _prepare_operation(tenant_id, user_id, entity, record, approved_aliases=()):
        """Identity acceptance retains existing metadata and does not decide fact truth."""
        canonical_id = str(record["id"]) if record is not None else str(uuid4())
        payload = {name: deepcopy(record["payload"][name] if record is not None else entity[name])
                   for name in ("entity_name", "entity_type", "entity_domain")}
        payload["entity_id"] = canonical_id
        if record is not None and record["payload"].get("first_mentioned_at") is not None:
            payload["first_mentioned_at"] = record["payload"]["first_mentioned_at"]
        aliases = deepcopy(record["payload"].get("aliases", [])) if record is not None else []
        for alias in approved_aliases:
            if alias != payload["entity_name"] and alias not in aliases:
                aliases.append(alias)
        payload["aliases"] = aliases
        payload["facts"] = deepcopy(record["payload"].get("facts", [])) if record is not None else []
        op = {"mode": "MUTATE" if record is not None else "INSERT", "tenant_id": tenant_id, "user_id": user_id,
              **{key: payload[key] for key in ("entity_id", "entity_name", "entity_type", "entity_domain")},
              "payload": payload, "extracted_facts": deepcopy(entity.get("facts", [])),
              "_mention_context": entity.get("mention_context"),
              "_dictionary_evidence": deepcopy(entity.get("_dictionary_evidence", [])),
              "importance": max((f.get("_importance", 3) for f in payload["facts"]), default=3)}
        if record is not None:
            op["db_id"] = canonical_id
            op["_expected_payload"] = deepcopy(record["payload"])
        else:
            op["reserved_id"] = canonical_id
        return op
