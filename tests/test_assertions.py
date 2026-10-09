"""Deterministic assertion lifecycle; no external model calls."""
from copy import deepcopy
import json
import subprocess
import sys
import pytest
from tomte.engine.core.assertions import (
    FactMutation, MutationPlanResponse, adapt_assertions, apply_mutations, project_facts,
)
from tomte.engine.core.extraction import ExtractionMemorySchema


SOURCES=[{"item_id":"source-1","timestamp":1791504000,"role":"user","content":"Synthetic statement"},
         {"item_id":"source-2","timestamp":1791504060,"role":"user","content":"Synthetic confirmation"}]


def op(kind="ADD", target=None, value="owned", attribute="ownership_status", source="source-1", **kw):
    data={"operation":kind,"target_fact_id":target,"attribute_name":attribute,
          "new_value":value,"source_chat_ids":[source],"interpretation_confidence":0.96,
          "evidence_type":"USER_EXPLICIT"}
    data.update(kw)
    return FactMutation.model_validate(data)


def apply(facts, operations, batch="batch-1", scope="entity-one"):
    return apply_mutations(facts,operations,SOURCES,scope,batch)


def seed():
    return apply([], [op()])


def test_add_stable_identity_evidence_and_provenance():
    facts=seed();fact,=facts
    assert fact["assertion_state"]=="ACTIVE" and fact["fact_id"]
    assert fact["source_chat_id"]=="source-1" and fact["evidence"][0]["source_chat_id"]=="source-1"
    assert fact["created_at"]==fact["evidence"][0]["recorded_at"]==fact["last_mentioned_at"]
    assert fact["effective_at"] is None and fact["ended_at"] is None
    assert seed()==facts
    assert apply([], [op()],scope="entity-two")[0]["fact_id"]!=fact["fact_id"]


def test_reinforce_keeps_id_adds_evidence_once_and_counts():
    before=seed();old=deepcopy(before)
    reinforced=apply(before,[op("REINFORCE",before[0]["fact_id"],None,source="source-2")],"batch-2")
    assert before==old
    assert len(reinforced)==1 and reinforced[0]["fact_id"]==before[0]["fact_id"]
    assert reinforced[0]["mention_count"]==2 and len(reinforced[0]["evidence"])==2
    assert apply(reinforced,[op("REINFORCE",before[0]["fact_id"],None,source="source-2")],"batch-2")==reinforced


def test_supersede_preserves_immutable_truth_provenance_and_links():
    before=seed();old=deepcopy(before[0])
    result=apply(before,[op("SUPERSEDE",old["fact_id"],"sold",source="source-2")],"batch-2")
    prior,current=result
    assert prior["fact_id"]==old["fact_id"] and prior["value"]=="owned"
    assert prior["source_chat_id"]==old["source_chat_id"] and prior["created_at"]==old["created_at"]
    assert prior["assertion_state"]=="HISTORICAL" and prior["ended_at"] is None
    assert current["value"]=="sold" and current["assertion_state"]=="ACTIVE"
    assert current["fact_id"]!=prior["fact_id"] and current["supersedes_fact_id"]==prior["fact_id"]
    assert project_facts(result)==[current] and project_facts(result,"history")==[prior]


def test_retract_has_no_replacement_and_preserves_history():
    before=apply([], [op(value="Japan",attribute="trip_plan")])
    result=apply(before,[op("RETRACT",before[0]["fact_id"],None,attribute="trip_plan",source="source-2")])
    assert len(result)==1 and result[0]["value"]=="Japan"
    assert result[0]["assertion_state"]=="RETRACTED" and project_facts(result)==[]
    assert project_facts(result,"history")==result
    assert result[0]["source_chat_id"]==before[0]["source_chat_id"]


@pytest.mark.parametrize("replacement",[None,"owned"])
def test_correct_invalidates_without_fabricating_history_or_restoration(replacement):
    before=apply([], [op(value="sold")])
    result=apply(before,[op("CORRECT",before[0]["fact_id"],replacement,source="source-2")])
    assert result[0]["assertion_state"]=="CORRECTED_INVALID" and result[0]["value"]=="sold"
    assert project_facts(result,"history")==[] and len(project_facts(result,"audit"))==len(result)
    assert len(result)==(1 if replacement is None else 2)
    if replacement is not None:
        assert result[1]["corrects_fact_id"]==before[0]["fact_id"]
        assert result[1]["value"]==replacement and project_facts(result)==[result[1]]
    else:
        assert project_facts(result)==[]


def test_correct_does_not_restore_a_superseded_predecessor():
    before=seed()
    changed=apply(before,[op("SUPERSEDE",before[0]["fact_id"],"sold",source="source-2")])
    invalid=apply(changed,[op("CORRECT",changed[1]["fact_id"],None)])
    assert project_facts(invalid)==[]
    assert [f["value"] for f in project_facts(invalid,"history")]==["owned"]


def test_multi_valued_add_keeps_both_active():
    facts=apply([], [op(value="Python",attribute="programming_language"),
                     op(value="Java",attribute="programming_language")])
    assert len(project_facts(facts))==2 and len({f["fact_id"] for f in facts})==2


def test_whole_plan_failure_leaves_original_unchanged():
    before=seed();copy=deepcopy(before)
    with pytest.raises(ValueError,match="offered entity"):
        apply(before,[op(value="red",attribute="paint_color"),op("RETRACT","foreign-fact",None)])
    assert before==copy


@pytest.mark.parametrize("source",["fabricated-id","unknown"])
def test_source_validation_rejects_fabrication(source):
    with pytest.raises(ValueError,match="frozen batch"):
        apply([], [op(source=source)])


def test_reference_only_source_not_allowed():
    with pytest.raises(ValueError):apply([], [op(source="previous-session-message")])


@pytest.mark.parametrize("extra",[{"created_at":"2026-01-01T00:00:00Z"},{"fact_id":"invented"},
                                   {"old_value":"sold"},{"effective_at":"2026-01-01T00:00:00Z"}])
def test_planner_rejects_authoritative_system_metadata(extra):
    with pytest.raises(ValueError):op(**extra)


@pytest.mark.parametrize("kind,target,value",[("ADD","invented","red"),("ADD",None,None),
    ("REINFORCE",None,None),("REINFORCE","id","new"),("RETRACT","id","new"),
    ("SUPERSEDE","id",None),("CORRECT",None,None)])
def test_invalid_operation_shapes_fail(kind,target,value):
    with pytest.raises(ValueError):op(kind,target,value)


def test_same_value_supersede_and_inactive_reinforce_fail():
    before=seed()
    with pytest.raises(ValueError,match="different"):apply(before,[op("SUPERSEDE",before[0]["fact_id"]," OWNED ")])
    historical=apply(before,[op("SUPERSEDE",before[0]["fact_id"],"sold")])
    with pytest.raises(ValueError,match="ACTIVE"):apply(historical,[op("REINFORCE",before[0]["fact_id"],None)])


def test_target_ids_are_entity_specific_and_offered_only():
    a=seed();b=apply([], [op()],scope="entity-two")
    with pytest.raises(ValueError,match="offered entity"):
        apply(b,[op("SUPERSEDE",a[0]["fact_id"],"sold")],scope="entity-two")
    with pytest.raises(ValueError):
        apply_mutations(a,[op("RETRACT",a[0]["fact_id"],None)],SOURCES,"entity-one","batch",offered_fact_ids=set())


def test_legacy_adaptation_stable_across_reads_and_process_restart():
    legacy=[{"attribute_name":"ownership","value":"owned","confidence":0.8,
             "source_chat_id":"old-message","last_mentioned_at":"2026-10-08T00:00:00Z","timeline":[]}]
    first=adapt_assertions(legacy,"entity-one")
    assert first==adapt_assertions(legacy,"entity-one") and "fact_id" not in legacy[0]
    code="from tomte.engine.core.assertions import adapt_assertions; import json; print(json.dumps(adapt_assertions("+repr(legacy)+",'entity-one')))"
    other=json.loads(subprocess.check_output([sys.executable,"-B","-c",code],text=True))
    assert first==other and first[0]["evidence"][0]["evidence_type"]=="LEGACY"


def test_legacy_past_and_future_interpretation():
    past={"attribute_name":"job","value":"old job","status":"past"}
    future={"attribute_name":"trip","value":"Japan","status":"future"}
    adapted=adapt_assertions([past,future])
    assert adapted[0]["assertion_state"]=="HISTORICAL"
    assert project_facts(adapted)==[adapted[1]]
    assert adapted[0]["evidence"]==[]  # Unknown legacy provenance is not invented.


def test_extraction_does_not_ask_model_for_system_assertion_fields():
    schema=ExtractionMemorySchema.model_json_schema()
    fields=schema["$defs"]["ExtractedFact"]["properties"]
    assert not set(fields)&{"fact_id","created_at","assertion_state","evidence","supersedes_fact_id","effective_at"}
