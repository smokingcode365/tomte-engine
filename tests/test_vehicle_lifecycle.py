from memory_fixtures import field_entries, fact_text
from uuid import UUID
from resolution_helpers import resolution_response, mutation_response
"""Deterministic vehicle persistence/context checks on local PostgreSQL/Redis.

Reuse the existing opt-in mechanics fixture and its scoped cleanup. These tests
prescribe model outputs and vectors; they do not evaluate natural-language
extraction, entity resolution, or real embedding quality. No daemon/model API runs.
"""

from collections import Counter
import json

import psycopg
from psycopg.pq import TransactionStatus
import pytest

from test_consolidation_mechanics import (
    mechanics, memory_rows, dictionary_state, processed_count, acknowledge, extraction_response,
)
from tomte.engine import Tomte
from tomte.engine.core.enums import ChatSenderRole, MemoryDomain
from tomte.engine.core.extraction import ExtractedEntity, ExtractedFact
from tomte.engine.core.extraction import ExtractionMemorySchema
from tomte.engine.workflows.chat_hook import ChatHookWorkflow
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


MESSAGES = (
    "I bought a Tesla Model Y.", "I painted my car red.",
    "I named it Little Dog.", "I sold Little Dog.",
)
QUERIES = (
    "What car do I own?", "What color is my Tesla?", "What is Little Dog?",
    "Do I still own Little Dog?", "What happened to my red Tesla?",
    "What vehicle did I name Little Dog?", "I'm thinking about my car.",
)


class VehicleLLM:
    def __init__(self, db, index):
        self.db, self.index, self.calls = db, index, []
        self.source = None

    def get_vector_dimension(self):
        return 1536

    def prepare(self, stage, db_id=None):
        self.stage, self.db_id, self.message = stage, db_id, MESSAGES[stage]
        self.mode = "INSERT" if stage == 0 else "MUTATE"
        self.calls, self.extra_entity, self.vector_index = [], None, self.index

    def record(self, method):
        assert self.db.conn.info.transaction_status == TransactionStatus.IDLE
        self.calls.append(method)

    def vector(self):
        vector = [float(i == self.vector_index) for i in range(1536)]
        if self.vector_index == self.index:
            vector[(self.index + 1) % 1536] = self.stage * 0.04
        return vector

    def embed_text(self, text):
        assert text == self.message or text in QUERIES
        self.record("embed_text")
        return self.vector()

    def is_junk(self, text):
        assert json.loads(text)["CURRENT_BATCH"][0]["content"] == self.message
        self.record("is_junk")
        return False

    def fact(self, key, value):
        return ExtractedFact(attribute_name=key,value=field_entries(value),importance=4,source_chat_id=self.source)

    def extracted(self):
        if self.extra_entity is not None:
            return self.extra_entity
        facts = (
            [self.fact("make", "Tesla"), self.fact("model", "Model Y"),
             self.fact("ownership_status", "owned"),
             self.fact("purchase_event", "Purchased Tesla Model Y")],
            [self.fact("paint_color", "red")],
            [self.fact("nickname", "Little Dog")],
            [self.fact("ownership_status", "sold"),
             self.fact("sale_event", "Sold Little Dog")],
        )[self.stage]
        return ExtractedEntity(
            entity_id="tmp_veh_01" if self.stage == 0 else f"tmp_veh_9{self.stage}",
            entity_name="Tesla Model Y", entity_type="vehicle", entity_domain=MemoryDomain.ASSETS,
            aliases=(["Tesla Model Y"], ["my car"], ["it", "Little Dog"], ["Little Dog"])[self.stage],
            facts=facts,
        )

    def parse_structured(self, system_prompt, user_text, schema, taxonomy=None, lexicon=None):
        if schema.__name__ in {"TerminologyAlignmentResponse","FieldAlignmentResponse","ContextualActionResponse"}:
            self.record({"TerminologyAlignmentResponse":"align","FieldAlignmentResponse":"field_align","ContextualActionResponse":"contextual"}[schema.__name__])
            return schema()
        if schema.__name__ == "MutationPlanResponse":
            self.record("mutate")
            return mutation_response(schema,user_text)
        entity = self.extracted()
        if schema is ExtractionMemorySchema:
            assert json.loads(user_text)["CURRENT_BATCH"][0]["content"] == self.message
            assert taxonomy is None and lexicon is None
            self.record("extract")
            return extraction_response(schema, entity, user_text)[0]
        assert schema.__name__ == "ResolutionResponse"
        if self.db_id:
            assert str(self.db_id) in user_text
        self.record("reconcile")
        return resolution_response(schema, {"mappings": [{
            "extracted_entity_id": entity.entity_id, "mode": self.mode,
            "matched_db_id": str(self.db_id) if self.db_id else None,
            "reasoning": "Prescribed synthetic match, not a model-quality claim",
            "alias_evidence": [{"alias": "Little Dog", "source_chat_id": json.loads(user_text)["CURRENT_BATCH"][0]["item_id"], "evidence_quote": self.message}] if self.stage == 2 else [],
            "accepted_aliases": ["Little Dog"] if self.stage == 2 else [],
        }]}, user_text)

    def embed_batch(self, texts):
        if texts and texts[0].startswith("domain: "):
            self.record("dictionary_embed")
        elif texts and " field: " in texts[0]:
            self.record("field_embed")
        else:
            self.record("embed_batch")
            if self.extra_entity is None:
                assert all("Tesla" in text and "Model Y" in text for text in texts)
        return [self.vector() for _ in texts]


@pytest.fixture
def vehicle_case(mechanics, monkeypatch):
    case = mechanics
    case.llm = VehicleLLM(case.db, case.llm.index)
    case.llm.prepare(0)
    assert Tomte.shutdown()
    Tomte.init(case.db, case.cache, case.llm, start_daemon_worker=False)
    assert memory_rows(case) == [] and processed_count(case) == 0 and case.anchors == []
    assert case.db.find_dictionary_matches_above_threshold(case.llm.vector(), 0.75) == []
    assert list(case.cache.client.scan_iter(match="*" + case.session + "*")) == []
    commit = case.db.commit_graph_updates
    def check_identity(operations):
        for op in operations:
            canonical = next((r["entity_id"] for r in memory_rows(case) if str(r["id"]) == op.get("db_id")), op["entity_id"])
            assert str(UUID(canonical))==canonical
            assert op["entity_id"] == op["payload"]["entity_id"] == canonical
            if op["mode"] != "INSERT":
                assert op["db_id"] == str(case.llm.db_id)
        return commit(operations)
    monkeypatch.setattr(case.db, "commit_graph_updates", check_identity)
    return case


def facts(row):
    return {fact["attribute_name"]: fact for fact in row["payload"]["facts"]}


def queue_input(case):
    ChatHookWorkflow().on_chat(case.tenant, case.session, case.llm.message, ChatSenderRole.USER, case.user)
    queue = case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session)
    assert len(queue) == 1 and queue[0]["content"] == case.llm.message
    case.llm.source = queue[0]["item_id"]
    return queue


def consolidate_stage(case, stage, db_id=None):
    case.llm.prepare(stage, db_id)
    queue = queue_input(case)
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue)
    assert case.llm.calls == ["embed_text", "is_junk", "extract", "dictionary_embed", "align", "field_embed", "field_align", "reconcile"] + (["contextual"] if stage else []) + ["mutate", "embed_batch"]
    row, = memory_rows(case)
    assert row["entity_id"] == row["payload"]["entity_id"]
    assert row["entity_id"]==str(row["id"])
    assert row["payload"]["entity_type"] == "vehicle"
    assert row["payload"]["entity_domain"] == "assets_and_possessions"
    if db_id:
        assert row["id"] == db_id
    case.last_queue = queue
    acknowledge(case)
    return row


def context(case, query):
    case.llm.vector_index = case.llm.index
    result = ChatHookWorkflow().on_chat(case.tenant, case.session, query, ChatSenderRole.USER, case.user)
    acknowledge(case)  # This synthetic retrieval question is not consolidated.
    memory=result["query_relevant_memories"]+result["background_memories"]
    assert all(json.dumps(entity,ensure_ascii=False) in result["context_text"] for entity in memory)
    vehicle_id=next(r["entity_id"] for r in memory_rows(case) if r["payload"]["entity_type"]=="vehicle")
    assert sum(e["entity_id"]==vehicle_id for e in memory)==1
    assert all(not e["entity_id"].startswith("tmp_") for e in memory)
    return memory


def context_values(entities):
    return {f["attribute_name"]:fact_text(f) for e in entities for f in e["facts"]}


def test_full_vehicle_lifecycle_preserves_identity_state_history_and_replay(vehicle_case):
    case = vehicle_case
    snapshots = []
    for stage in range(4):
        previous = snapshots[-1] if snapshots else None
        row = consolidate_stage(case, stage, snapshots[0]["id"] if snapshots else None)
        snapshots.append(row)
        assert row["mention_count"] == (4,5,6,8)[stage] and row["importance"] == 4
        expected_aliases = set()
        if stage >= 2:
            expected_aliases.add("Little Dog")
        assert set(row["payload"]["aliases"]) == expected_aliases
        if previous:
            assert row["created_at"] == previous["created_at"]
            assert row["last_mentioned_at"] >= previous["last_mentioned_at"]
            assert row["vector"] != previous["vector"]
            for key, fact in facts(previous).items():
                if not (stage == 3 and key == "ownership_status"):
                    assert facts(row)[key] == fact
        query = QUERIES[stage]
        text = context(case, query)
        assert context_values(text)["make"]=="Tesla" and context_values(text)["model"]=="Model Y"
        if stage >= 1:
            assert context_values(text)["paint_color"]=="red"
        if stage >= 2:
            assert context_values(text)["nickname"]=="Little Dog"
        if stage == 3:
            assert context_values(text)["ownership_status"]=="sold"
            assert context_values(context(case, QUERIES[4]))["sale_event"]=="Sold Little Dog"
    final = snapshots[-1]
    assert set(final["payload"]["aliases"]) == {"Little Dog"}
    assert len({row["id"] for row in snapshots}) == 1
    assert len(memory_rows(case)) == 1 and processed_count(case) == 4
    original=facts(snapshots[0])["ownership_status"]
    current=facts(final)["ownership_status"]
    assert current["fact_id"]==original["fact_id"] and current["history_values"][0]["value"]==original["value"]
    assert fact_text(facts(final)["purchase_event"])=="Purchased Tesla Model Y"
    assert all("timeline" not in f and "source_chat_id" not in f for f in final["payload"]["facts"])
    assert len(case.anchors)==20  # Entity + attribute + parent-scoped field anchors.
    hits = Counter()
    for anchor in case.anchors:
        for item in case.db.get_anchor_keywords(str(anchor)):
            hits[item["keyword"]] += item["hits"]
    assert hits == {"vehicle": 4, "make": 1, "model": 1, "ownership_status": 2,
                    "purchase_event": 1, "paint_color": 1, "nickname": 1, "sale_event": 1, "detail": 8}
    assert not {"car", "Tesla", "Model Y", "Little Dog"}.intersection(hits)
    before, dictionary_before = memory_rows(case), dictionary_state(case)
    case.llm.calls = []
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, case.last_queue)
    assert memory_rows(case) == before and dictionary_state(case) == dictionary_before
    assert processed_count(case) == 4 and case.llm.calls == []
    print("Vehicle snapshots: " + json.dumps([{
        "UUID": str(r["id"]), "entity_id": r["entity_id"], "aliases": sorted(r["payload"]["aliases"]),
        "facts": {key: fact["value"] for key, fact in facts(r).items()}, "mention_count": r["mention_count"],
    } for r in snapshots]))
    print("Dictionary keyword hits: " + json.dumps(hits))
    assert case.db.conn.info.transaction_status == TransactionStatus.IDLE


@pytest.mark.parametrize("stage", [1, 2, 3], ids=["purchase-to-paint", "purchase-to-nickname", "purchase-to-sale"])
def test_purchase_then_one_vehicle_update_preserves_unrelated_facts(vehicle_case, stage):
    case = vehicle_case
    purchased = consolidate_stage(case, 0)
    updated = consolidate_stage(case, stage, purchased["id"])
    assert updated["mention_count"] == (6 if stage==3 else 5) and facts(updated)["make"] == facts(purchased)["make"]
    assert facts(updated)["model"] == facts(purchased)["model"]
    key, expected = {1: ("paint_color", "red"), 2: ("nickname", "Little Dog"), 3: ("ownership_status", "sold")}[stage]
    assert fact_text(facts(updated)[key]) == expected
    text = context(case, QUERIES[stage])
    assert context_values(text)[key]==expected


def test_vehicle_update_failure_rolls_back_and_same_queue_retry_is_idempotent(vehicle_case, monkeypatch):
    case = vehicle_case
    purchased = consolidate_stage(case, 0)
    case.llm.prepare(1, purchased["id"])
    queue = queue_input(case)
    before, dictionary_before = memory_rows(case), dictionary_state(case)
    original = case.db.upsert_dictionary_keyword
    def fail_attribute(anchor, keyword, kind, keyword_context=""):
        return original(anchor, None if kind == "ATTRIBUTE" else keyword, kind, keyword_context)
    monkeypatch.setattr(case.db, "upsert_dictionary_keyword", fail_attribute)
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue)
    assert memory_rows(case) != before and dictionary_state(case) == dictionary_before
    assert processed_count(case) == 2
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == queue
    assert case.db.conn.info.transaction_status == TransactionStatus.IDLE
    monkeypatch.setattr(case.db, "upsert_dictionary_keyword", original)
    case.llm.calls = []
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue)
    painted, = memory_rows(case)
    assert painted["id"] == purchased["id"] and painted["mention_count"] == 5
    assert fact_text(facts(painted)["paint_color"]) == "red"
    assert processed_count(case) == 2
    before, dictionary_before = memory_rows(case), dictionary_state(case)
    case.llm.calls = []
    for _ in range(2):
        assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue)
    assert memory_rows(case) == before and dictionary_state(case) == dictionary_before
    assert processed_count(case) == 2 and case.llm.calls == []
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == queue



def test_vehicle_semantic_rank_and_profile_behavior_with_unrelated_memories(vehicle_case):
    case = vehicle_case
    vehicle = None
    for stage in range(4):
        vehicle = consolidate_stage(case, stage, vehicle["id"] if vehicle else None)
    vehicle_vector = case.llm.vector()
    # Seed unrelated memories through the same real Chat Hook/consolidation path.
    extras = (
        ("per_01", "Synthetic boss", MemoryDomain.RELATIONSHIPS, "person", "relationship", "boss", 3),
        ("act_01", "Gardening", MemoryDomain.ACTIVITIES, "hobby", "activity", "gardening", 3),
        ("usr_01", "Synthetic driver", MemoryDomain.IDENTITY, "person", "display_name", "Synthetic Driver", 5),
    )
    indices = []
    for offset, (eid, name, domain, kind, key, value, importance) in enumerate(extras):
        index = next(i for i in range(1536) if i not in [case.llm.index, (case.llm.index + 1) % 1536, *indices]
                     and not case.db.find_dictionary_matches_above_threshold([float(j == i) for j in range(1536)], 0.75))
        indices.append(index)
        case.llm.prepare(0)
        case.llm.message = "Synthetic unrelated fixture: " + name
        case.llm.vector_index = index
        case.llm.extra_entity = ExtractedEntity(
            entity_id="tmp_"+eid, entity_name=name, entity_domain=domain, entity_type=kind,
            aliases=[name], facts=[ExtractedFact(attribute_name=key,value=field_entries(value),importance=importance,source_chat_id="synthetic-unrelated-fixture")],
        )
        queue = queue_input(case)
        assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, queue)
        acknowledge(case)
    rows = memory_rows(case)
    assert len(rows) == 4 and sum(r["payload"]["entity_type"] == "vehicle" for r in rows) == 1
    semantic = case.db.semantic_retrieval(case.tenant, case.user, vehicle_vector, limit=4)
    assert semantic[0]["entity_id"] == vehicle["entity_id"] and semantic[0]["relevance_score"] > 0
    assert all(entity["relevance_score"] == pytest.approx(0) for entity in semantic[1:])
    assert case.db.semantic_retrieval(case.tenant, case.user, vehicle_vector, limit=1)[0]["entity_id"] == vehicle["entity_id"]
    profile = case.db.get_important_entities(case.tenant, case.user, domain=MemoryDomain.IDENTITY)
    profile_id = next(r["entity_id"] for r in rows if r["payload"]["entity_name"] == "Synthetic driver")
    assert [entity["entity_id"] for entity in profile] == [profile_id]
    case.llm.stage = 3
    text = context(case, "I'm thinking about my car.")
    assert context_values(text)["ownership_status"]=="sold" and context_values(text)["display_name"]=="Synthetic Driver"
    # Semantic ranking is correct, but final context includes irrelevant recent/frequent memories.
    assert {"Synthetic boss","Gardening"} <= {e["entity_name"] for e in text}
    assert [e["entity_id"] for e in text].index(vehicle["entity_id"]) < [e["entity_id"] for e in text].index(profile_id)  # Current query-first context contract.
    assert case.cache.get_persona_entities(case.tenant, case.user) == []  # Current path reads scoped profile directly.
    print("Retrieval: vehicle ranked first; unrelated semantic scores zero; profile injected; boss/hobby still included through recent/frequent context")
