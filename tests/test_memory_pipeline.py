"""Offline positive/abstention evidence for embedded facts and atomic replay."""
from contextlib import contextmanager
from copy import deepcopy
import json

from tomte.engine.core.extraction import ExtractionMemorySchema, validate_operation_provenance
from tomte.engine.core.assertions import MutationPlanResponse
from tomte.engine.core.resolution import ResolutionResponse
from tomte.engine.dictionary.alignment import TerminologyAlignmentResponse, FieldAlignmentResponse
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


class MemoryDB:
    def __init__(self):
        self.rows = []
        self.processed = set()

    @contextmanager
    def transaction(self):
        snapshot = deepcopy((self.rows, self.processed))
        try:
            yield
        except Exception:
            self.rows, self.processed = snapshot
            raise

    def is_consolidation_processed(self, tenant, user, batch):
        return (tenant, user, batch) in self.processed

    def claim_consolidation_batch(self, tenant, user, batch):
        key = tenant, user, batch
        if key in self.processed:
            return False
        self.processed.add(key)
        return True

    def get_semantic_keywords(self, *args, **kwargs):
        return []

    def get_resolution_candidates(self, *args, **kwargs):
        return []

    def commit_graph_updates(self, operations):
        validate_operation_provenance(operations)
        self.rows.extend(deepcopy(operations))
        return True


class LLM:
    def __init__(self, abstain=False):
        self.abstain = abstain
        self.calls = []

    def get_vector_dimension(self):
        return 2

    def embed_batch(self, texts):
        return [[1.0, 0.0] for _ in texts]

    def is_junk(self, text):
        return False

    def parse_structured(self, system_prompt, user_text, schema):
        self.calls.append(schema)
        if schema is ExtractionMemorySchema:
            return schema.model_validate({"relationships": [{"entity_id": "tmp_per_1", "entity_name": "Bob",
                "entity_domain": "relationships", "entity_type": "person", "facts": [
                    {"attribute_name": "relationship", "source_chat_id": "m1", "value": [
                        {"field_name": "relationship_to_user", "text": "friend", "fact_confidence": "high"}]},
                    {"attribute_name": "employment", "source_chat_id": "m1", "value": [
                        {"field_name": "employer_name", "text": "Cedar Labs", "fact_confidence": "high"}]}]}]})
        if schema in (TerminologyAlignmentResponse, FieldAlignmentResponse):
            return schema()
        if schema is ResolutionResponse:
            return schema.model_validate({"resolutions": [{"extracted_entity_id": "tmp_per_1", "kind": "NEW",
                "reasoning": "Current user explicitly introduces Bob.",
                "explicitly_new": True, "new_entity_evidence": "My friend Bob works at Cedar Labs."}]})
        if schema is MutationPlanResponse:
            if self.abstain:
                return schema(plans=[])
            target = json.loads(user_text)["TARGETS"][0]
            return schema.model_validate({"plans": [{"target_key": target["target_key"], "operations": [
                {"operation": "ADD", "attribute_name": f["attribute_name"], "source_chat_ids": ["m1"],
                 "incoming_fact_key": f["incoming_fact_key"],
                 "fields": [{"field_name": k, **v} for k, v in f["value"].items()]}
                for f in target["incoming_semantics"]]}]})
        raise AssertionError(schema)


def chat():
    return [{"tenant_id": "tenant", "user_id": "user", "session_id": "session", "item_id": "m1",
             "role": "user", "timestamp": 1793613600, "content": "My friend Bob works at Cedar Labs.", "vector": [1.0, 0.0]}]


def test_bob_employer_persists_without_a_separate_organization_or_relationship_ids():
    db, llm = MemoryDB(), LLM()
    workflow = ConsolidationWorkflow(db, llm)
    assert workflow.run_consolidation_cycle("tenant", "user", chat())
    assert len(db.rows) == 1
    op = db.rows[0]
    assert op["reserved_id"] == op["entity_id"] == op["payload"]["entity_id"]
    facts = op["payload"]["facts"]
    assert len(facts) == 2
    assert facts[1]["value"]["employer_name"]["text"] == "Cedar Labs"
    serialized = json.dumps(op["payload"])
    assert "source_chat" not in serialized and "related_entity" not in serialized and "tmp_" not in serialized
    assert len(op["search_entries"]) == 2
    before = len(llm.calls)
    assert workflow.run_consolidation_cycle("tenant", "user", chat())
    assert len(db.rows) == 1 and len(llm.calls) == before


def test_empty_mutation_plan_abstains_without_forced_writes_or_retry():
    db, llm = MemoryDB(), LLM(abstain=True)
    assert ConsolidationWorkflow(db, llm).run_consolidation_cycle("tenant", "user", chat())
    assert db.rows == []
    assert len(db.processed) == 1
    assert llm.calls.count(MutationPlanResponse) == 1
