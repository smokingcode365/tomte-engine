"""Vocabulary safety contracts. Stub decisions test mechanics, not model quality."""
from contextlib import nullcontext
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from pydantic import ValidationError
import pytest

from tomte.engine.core.extraction import ExtractionMemorySchema, resolve_extraction
from tomte.engine.dictionary.alignment import (
    TerminologyAlignmentResponse, align_terms, apply_decisions, query_text,
)
from tomte.engine.dictionary.service import DictionaryService
from tomte.engine.llm.prompts.dictionary_maintenance import CROSS_ANCHOR_MERGE_PROMPT
from tomte.engine.llm.prompts.terminology_alignment import TERMINOLOGY_ALIGNMENT_PROMPT
from tomte.engine.llm.structured import TomteDeps, add_dictionaries
from test_postgres_transactions import local_db
from test_consolidation_mechanics import mechanics


def example(attribute="exterior_vehicle_colour", value="dark blue", entity_type="automobile"):
    batch = [{"item_id": "source-1", "timestamp": 1791547200,
              "content": "I painted the exterior of my Tesla dark blue.", "role": "user"},
             {"item_id": "source-2", "timestamp": 1791547210,
              "content": "Unrelated synthetic context must not be included.", "role": "user"}]
    wire = ExtractionMemorySchema.model_validate({"assets_and_possessions": [{
        "entity_id": "veh_99", "entity_name": "Tesla", "entity_domain": "assets_and_possessions",
        "entity_type": entity_type, "aliases": ["my Tesla"],
        "facts": [{"attribute_name": attribute, "value": value, "confidence": .9,
                   "source_chat_id": "source-1"}]}]})
    raw = [e.model_dump(mode="json") for e in resolve_extraction(wire, batch).assets_and_possessions]
    return raw, batch


def setup_stage(response=None):
    raw, batch = example()
    entity_id, attribute_id = uuid4(), uuid4()
    def retrieve(vector, threshold, id_limit, keyword_type, keyword_limit):
        return [{"dictionary_id": entity_id if keyword_type == "ENTITY" else attribute_id,
                 "keyword": "car" if keyword_type == "ENTITY" else "paint_color",
                 "keyword_type": keyword_type, "score": .99, "hits": 1}]
    db = SimpleNamespace(get_semantic_keywords=Mock(side_effect=retrieve))
    llm = SimpleNamespace(embed_batch=Mock(return_value=[[1., 0.], [0., 1.]]),
                          get_vector_dimension=lambda: 2,
                          parse_structured=Mock(return_value=response or TerminologyAlignmentResponse()))
    return raw, batch, db, llm


def reuse_response():
    return {"entity_type_decisions": [{"entity_handle": "entity_0", "decision": "REUSE",
                                      "candidate_handle": "candidate_0_0"}],
            "attribute_decisions": [{"entity_handle": "entity_0", "fact_handle": "fact_0_0",
                                      "decision": "REUSE", "candidate_handle": "candidate_1_0"}]}


def test_raw_extraction_does_not_inject_dictionary_guidance():
    assert add_dictionaries(SimpleNamespace(deps=TomteDeps([], []))) == ""
    # The workflow invocation itself is verified by real consolidation fixtures,
    # which require taxonomy/lexicon=None instead of silently ignoring vocabulary.


def test_query_contract_and_batched_embeddings_exclude_values_names_and_sources():
    raw, batch, db, llm = setup_stage()
    result = align_terms(raw, batch, llm, db)
    texts = llm.embed_batch.call_args.args[0]
    assert texts == ["domain: assets_and_possessions\nentity_type: automobile",
                     "domain: assets_and_possessions\nentity_type: automobile\nattribute: exterior_vehicle_colour"]
    assert all("dark blue" not in t and "Tesla" not in t for t in texts)
    assert result.metrics["embedding_texts"] == 2 and result.metrics["embedding_calls"] == 1
    assert db.get_semantic_keywords.call_count == 2
    llm.parse_structured.assert_called_once()
    view = json.loads(llm.parse_structured.call_args.kwargs["user_text"])["entities"][0]
    assert view["facts"][0]["value_read_only"] == "dark blue"
    assert view["facts"][0]["source_text"] == [batch[0]["content"]]
    assert batch[1]["content"] not in llm.parse_structured.call_args.kwargs["user_text"]
    assert set(view["candidates"][0]) == {"candidate_handle", "keyword"}
    assert set(result.evidence[0][0]) == {"keyword", "keyword_type", "vector", "dictionary_id"}


@pytest.mark.parametrize("kind", ["ENTITY", "ATTRIBUTE"])
def test_candidates_are_typed_and_bounded_even_if_provider_returns_wrong_type(kind):
    raw, batch, db, llm = setup_stage()
    def retrieve(vector, threshold, id_limit, keyword_type, keyword_limit):
        other = "ATTRIBUTE" if keyword_type == "ENTITY" else "ENTITY"
        return [{"dictionary_id": uuid4(), "keyword": "wrong", "keyword_type": other,
                 "score": .99}] + [{"dictionary_id": uuid4(), "keyword": f"term_{i}",
                    "keyword_type": keyword_type, "score": .99} for i in range(10)]
    db.get_semantic_keywords.side_effect = retrieve
    result = align_terms(raw, batch, llm, db)
    assert not result.metrics["raw_fallback"]
    view = json.loads(llm.parse_structured.call_args.kwargs["user_text"])["entities"][0]
    candidates = view["candidates"] if kind == "ENTITY" else view["facts"][0]["candidates"]
    assert len(candidates) == 5 and all(c["keyword"] != "wrong" for c in candidates)


def test_valid_reuse_changes_only_the_two_vocabulary_fields():
    raw, batch, db, llm = setup_stage(TerminologyAlignmentResponse.model_validate(reuse_response()))
    before = deepcopy(raw)
    result = align_terms(raw, batch, llm, db)
    assert result.entities[0]["entity_type"] == "car"
    assert result.entities[0]["facts"][0]["attribute_name"] == "paint_color"
    expected = deepcopy(before)
    expected[0]["entity_type"] = "car"
    expected[0]["facts"][0]["attribute_name"] = "paint_color"
    assert result.entities == expected and raw == before
    assert all(e["dictionary_id"] for e in result.evidence[0])


@pytest.mark.parametrize("value", ["dark blue", "Senior Software Engineer", "jointly owned", "leased"])
def test_values_provenance_metadata_names_and_counts_are_immutable(value):
    raw, batch, db, llm = setup_stage(TerminologyAlignmentResponse.model_validate(reuse_response()))
    raw[0]["facts"][0]["value"] = value
    result = align_terms(raw, batch, llm, db)
    assert result.entities[0]["facts"][0]["value"] == value
    restored = deepcopy(result.entities)
    restored[0]["entity_type"] = raw[0]["entity_type"]
    restored[0]["facts"][0]["attribute_name"] = raw[0]["facts"][0]["attribute_name"]
    assert restored == raw  # Includes all counts, aliases, confidence, IDs/timeline/source.


@pytest.mark.parametrize("response", [{}, {"entity_type_decisions": [{"entity_handle": "entity_0",
    "decision": "KEEP_RAW"}], "attribute_decisions": [{"entity_handle": "entity_0",
    "fact_handle": "fact_0_0", "decision": "KEEP_RAW"}]}])
def test_missing_or_explicit_keep_raw_preserves_exact_strings(response):
    raw, batch, db, llm = setup_stage(TerminologyAlignmentResponse.model_validate(response))
    raw[0]["entity_type"] = "unusual_exact_type"
    raw[0]["facts"][0]["attribute_name"] = "unusual_exact_attribute"
    result = align_terms(raw, batch, llm, db)
    assert result.entities == raw
    assert all(e["dictionary_id"] is None for e in result.evidence[0])


@pytest.mark.parametrize("fault", ["invented", "other_target", "unknown_entity", "unknown_fact",
                                   "duplicate", "wrong_kind", "value", "name", "facts", "source"])
def test_invalid_alignment_rejects_whole_response_without_partial_reuse(fault):
    response = reuse_response()
    decision = response["attribute_decisions"][0]
    if fault == "invented": decision["candidate_handle"] = "invented"
    elif fault == "other_target": decision["candidate_handle"] = "candidate_0_0"
    elif fault == "unknown_entity": decision["entity_handle"] = "entity_100"
    elif fault == "unknown_fact": decision["fact_handle"] = "fact_0_100"
    elif fault == "duplicate": response["attribute_decisions"].append(deepcopy(decision))
    elif fault == "wrong_kind": response["entity_type_decisions"][0]["candidate_handle"] = "candidate_1_0"
    elif fault == "value": decision["value"] = "blue"
    elif fault == "name": response["entity_type_decisions"][0]["entity_name"] = "replacement"
    elif fault == "facts": response["facts"] = []
    elif fault == "source": decision["source_chat_id"] = "invented-source"
    raw, batch, db, llm = setup_stage()
    llm.parse_structured.return_value = response  # Defend against non-conforming custom providers too.
    result = align_terms(raw, batch, llm, db)
    assert result.metrics["raw_fallback"] and result.entities == raw
    assert result.evidence == [[]]


def test_wrong_internal_candidate_type_rejects_response():
    raw, _ = example()
    targets = {("entity_0", None): {"keyword_type": "ENTITY", "candidates": [
        {"candidate_handle": "bad", "keyword_type": "ATTRIBUTE"}]}}
    with pytest.raises(ValueError):
        apply_decisions(raw, targets, {"entity_type_decisions": [
            {"entity_handle": "entity_0", "decision": "REUSE", "candidate_handle": "bad"}]})


@pytest.mark.parametrize("failure", ["embedding", "retrieval", "model"])
def test_provider_failure_retains_all_raw_memory_and_logs_no_private_values(monkeypatch, failure):
    raw, batch, db, llm = setup_stage()
    diagnostic = Mock()
    monkeypatch.setattr("tomte.engine.dictionary.alignment.tomtelog.warn", diagnostic)
    error = RuntimeError("do-not-log-private-value-or-credential")
    if failure == "embedding": llm.embed_batch.side_effect = error
    if failure == "retrieval": db.get_semantic_keywords.side_effect = error
    if failure == "model": llm.parse_structured.side_effect = error
    result = align_terms(raw, batch, llm, db)
    assert result.entities == raw and result.metrics["raw_fallback"]
    assert "do-not-log" not in str(diagnostic.call_args)


@pytest.mark.parametrize("raw_term,candidate", [
    ("car", "vehicle"), ("sedan", "car"), ("interior_color", "paint_color"),
    ("sale_status", "ownership_status"), ("current_location", "current_home_address"),
    ("professional_field", "job_title")])
def test_distinction_keep_raw_is_accepted_not_forced_to_vector_nearest(raw_term, candidate):
    # These test abstention/application mechanics. Whether the real model obeys
    # equivalence is assessed separately, never proved by prescribed stub output.
    raw, batch, db, llm = setup_stage()
    raw[0]["entity_type"] = raw_term if raw_term in ("car", "sedan") else "automobile"
    if raw_term not in ("car", "sedan"): raw[0]["facts"][0]["attribute_name"] = raw_term
    db.get_semantic_keywords.side_effect = None
    db.get_semantic_keywords.return_value = [
        {"dictionary_id": uuid4(), "keyword": candidate, "keyword_type":
         "ENTITY" if raw_term in ("car", "sedan") else "ATTRIBUTE", "score": 1.0}]
    assert align_terms(raw, batch, llm, db).entities == raw
    assert raw_term in TERMINOLOGY_ALIGNMENT_PROMPT.lower() and candidate in TERMINOLOGY_ALIGNMENT_PROMPT.lower()


def test_maintenance_prompt_explicitly_rejects_hierarchy():
    prompt = CROSS_ANCHOR_MERGE_PROMPT
    assert "hierarchy is NOT synonymy" in prompt
    assert "'car' / 'automobile'" in prompt
    assert "'car' / 'vehicle' are broader/narrower concepts, NOT exact synonyms" in prompt
    assert "'sedan' / 'car' are subtype/supertype concepts, NOT exact synonyms" in prompt


def test_feedback_reuse_updates_exact_anchor_keep_raw_creates_separate_anchor():
    accepted = str(uuid4())
    db = Mock()
    db.transaction.side_effect = nullcontext
    db.get_dictionary_anchor_details.return_value = {"embedding": [1., 0.], "total_hits": 2}
    db.create_dictionary_anchor.return_value = uuid4()
    service = DictionaryService.__new__(DictionaryService)
    service.db = db
    vector = [0., 1.]
    service.apply_feedback([{"extracted_facts": [{"attribute_name": "interior_color"}],
        "_dictionary_evidence": [{"keyword": "car", "keyword_type": "ENTITY",
            "vector": vector, "dictionary_id": accepted},
            {"keyword": "interior_color", "keyword_type": "ATTRIBUTE", "vector": vector,
             "dictionary_id": None}]}])
    db.get_dictionary_anchor_details.assert_called_once_with(accepted)
    assert db.update_dictionary_anchor_vector.call_args.args[0] == accepted
    db.create_dictionary_anchor.assert_called_once_with(vector)
    db.find_dictionary_matches_above_threshold.assert_not_called()
    assert db.upsert_dictionary_keyword.call_args_list[0].args == (accepted, "car", "ENTITY")


def test_deleted_reuse_anchor_raises_instead_of_redirecting_feedback():
    db = Mock()
    db.transaction.side_effect = nullcontext
    db.get_dictionary_anchor_details.return_value = None
    service = DictionaryService.__new__(DictionaryService)
    service.db = db
    with pytest.raises(ValueError, match="no longer exists"):
        service.apply_feedback([{"_dictionary_evidence": [{"keyword": "car", "keyword_type": "ENTITY",
            "vector": [1., 0.], "dictionary_id": str(uuid4())}]}])
    db.create_dictionary_anchor.assert_not_called()


@pytest.mark.parametrize("other", ["fact", "entity"])
def test_candidate_handle_cannot_be_borrowed_from_another_same_type_target(other):
    raw, batch, db, llm = setup_stage()
    if other == "entity":
        raw.append(deepcopy(raw[0]))
        raw[1]["entity_id"] = "veh_98"
        response = {"entity_type_decisions": [{"entity_handle": "entity_0", "decision": "REUSE",
                                                "candidate_handle": "candidate_2_0"}]}
    else:
        raw[0]["facts"].append(deepcopy(raw[0]["facts"][0]))
        raw[0]["facts"][1]["attribute_name"] = "interior_color"
        response = {"attribute_decisions": [{"entity_handle": "entity_0", "fact_handle": "fact_0_0",
                                              "decision": "REUSE", "candidate_handle": "candidate_2_0"}]}
    llm.embed_batch.side_effect = lambda texts: [[1., 0.] for _ in texts]
    llm.parse_structured.return_value = response
    result = align_terms(raw, batch, llm, db)
    assert result.metrics["raw_fallback"] and result.entities == raw


def test_each_fact_receives_only_its_own_validated_source_text():
    raw, batch, db, llm = setup_stage()
    raw[0]["facts"].append({**deepcopy(raw[0]["facts"][0]),
                           "attribute_name": "nickname", "source_chat_id": "source-2"})
    llm.embed_batch.side_effect = lambda texts: [[1., 0.] for _ in texts]
    result = align_terms(raw, batch, llm, db)
    view = json.loads(llm.parse_structured.call_args.kwargs["user_text"])["entities"][0]["facts"]
    assert view[0]["source_text"] == [batch[0]["content"]]
    assert view[1]["source_text"] == [batch[1]["content"]]
    assert result.entities == raw and result.metrics["embedding_texts"] == 3


def test_feedback_is_filtered_during_preparation_before_any_write_scope():
    from tomte.engine.core.assertions import MutationPlanResponse
    from tomte.engine.workflows.consolidation import ConsolidationWorkflow
    raw, batch = example()
    entity = raw[0]
    operation = {**deepcopy(entity), "tenant_id": "synthetic", "user_id": "synthetic",
                 "db_id": str(uuid4()), "payload": deepcopy(entity),
                 "extracted_facts": deepcopy(entity["facts"]), "mode": "MUTATE",
                 "_dictionary_evidence": [
                     {"keyword": entity["entity_type"], "keyword_type": "ENTITY", "vector": [1., 0.], "dictionary_id": None},
                     {"keyword": entity["facts"][0]["attribute_name"], "keyword_type": "ATTRIBUTE", "vector": [0., 1.], "dictionary_id": None}]}
    workflow = ConsolidationWorkflow.__new__(ConsolidationWorkflow)
    # No DB exists on this preparation object, so opening a write scope fails.
    workflow.llm = SimpleNamespace(parse_structured=Mock(return_value=MutationPlanResponse(plans=[])))
    prepared = workflow._plan_fact_mutations([operation],
        {"CURRENT_BATCH": batch, "RECENT_CONTEXT_REFERENCE_ONLY": []}, "synthetic-batch")
    assert prepared[0]["_dictionary_evidence"] == [operation["_dictionary_evidence"][0]]
    assert len(operation["_dictionary_evidence"]) == 2  # Preparation preserves its input.


def test_real_typed_sql_filters_before_limit_and_rollback_restores_dictionary(local_db):
    db = local_db.db
    vector = [0., 1.] + [0.] * 1534
    class Rollback(Exception): pass
    with pytest.raises(Rollback):
        with db.transaction():
            # Numerous wrong-type keywords cannot crowd out the requested type.
            for i in range(8):
                anchor = db.create_dictionary_anchor(vector)
                db.upsert_dictionary_keyword(anchor, f"type_{i}", "ENTITY")
            anchor = db.create_dictionary_anchor(vector)
            db.upsert_dictionary_keyword(anchor, "attribute_target", "ATTRIBUTE")
            rows = db.get_semantic_keywords(vector, .75, 1, keyword_type="ATTRIBUTE", keyword_limit=5)
            assert any(row["dictionary_id"] == anchor for row in rows)
            assert all(row["keyword_type"] == "ATTRIBUTE" for row in rows)
            assert len(rows) <= 5
            raise Rollback()
    assert db.get_dictionary_anchor_details(anchor) is None


def test_real_feedback_keeps_nearby_distinct_anchor_and_reuses_only_selected_anchor(local_db):
    db = local_db.db
    vector = [1.] + [0.] * 1535
    class Rollback(Exception): pass
    service = DictionaryService.__new__(DictionaryService)
    service.db = db
    with pytest.raises(Rollback):
        with db.transaction():
            chosen, nearby = db.create_dictionary_anchor(vector), db.create_dictionary_anchor(vector)
            db.upsert_dictionary_keyword(chosen, "paint_color", "ATTRIBUTE")
            db.upsert_dictionary_keyword(nearby, "interior_color", "ATTRIBUTE")
            service.apply_feedback([{"extracted_facts": [{"attribute_name": "paint_color"},
                                                       {"attribute_name": "seat_colour"}],
                "_dictionary_evidence": [
                    {"keyword": "paint_color", "keyword_type": "ATTRIBUTE", "vector": vector,
                     "dictionary_id": str(chosen)},
                    {"keyword": "seat_colour", "keyword_type": "ATTRIBUTE", "vector": vector,
                     "dictionary_id": None}]}])
            assert db.get_anchor_keywords(str(chosen))[0]["hits"] == 2
            assert [(r["keyword"], r["hits"]) for r in db.get_anchor_keywords(str(nearby))] == [("interior_color", 1)]
            rows = db.get_semantic_keywords(vector, .75, 20, keyword_type="ATTRIBUTE")
            separate = [r for r in rows if r["keyword"] == "seat_colour"]
            assert len(separate) == 1 and separate[0]["dictionary_id"] not in (chosen, nearby)
            raise Rollback()
    assert db.get_dictionary_anchor_details(chosen) is None


@pytest.mark.parametrize("fault", ["provider", "invalid_response"])
def test_alignment_failure_does_not_lose_committed_memory_or_replay_guarantee(mechanics, monkeypatch, fault):
    from test_consolidation_mechanics import queued_message, memory_rows, processed_count
    from tomte.engine.workflows.consolidation import ConsolidationWorkflow
    case = mechanics
    original = case.llm.parse_structured
    def model(system_prompt, user_text, schema, **kwargs):
        if schema is TerminologyAlignmentResponse:
            case.llm.record("align")
            if fault == "provider":
                raise RuntimeError("synthetic provider failure")
            return {"entity_type_decisions": [{"entity_handle": "entity_0", "decision": "REUSE",
                                               "candidate_handle": "invented"}]}
        return original(system_prompt, user_text, schema, **kwargs)
    monkeypatch.setattr(case.llm, "parse_structured", model)
    queue = queued_message(case)
    workflow = ConsolidationWorkflow()
    assert workflow.run_consolidation_cycle(case.tenant, case.user, queue)
    assert workflow.last_terminology_metrics["raw_fallback"]
    rows = memory_rows(case)
    assert len(rows) == 1 and rows[0]["payload"]["facts"][0]["value"] == "Python"
    assert "_dictionary_evidence" not in rows[0]["payload"]
    assert case.anchors == [] and processed_count(case) == 1
    calls = case.llm.calls[:]
    assert workflow.run_consolidation_cycle(case.tenant, case.user, queue)
    assert case.llm.calls == calls and memory_rows(case) == rows
