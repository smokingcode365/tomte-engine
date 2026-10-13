from tomte.engine.core.assertions import project_facts
from resolution_helpers import mutation_aware
"""Target-set contracts and local persistence; prescribed models do not prove LLM accuracy."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from tomte.engine import Tomte
from tomte.engine.core.resolution import ResolutionResponse
from tomte.engine.core.resolution import EntityResolution, AliasEvidence

from tomte.engine.core.extraction import ExtractionMemorySchema
from tomte.engine.core.enums import ChatSenderRole
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from tomte.engine.workflows.chat_hook import ChatHookWorkflow
from tomte.engine.utils.keys import generate_common_key, generate_consolidation_key, generate_consolidation_batch_key
from test_consolidation_mechanics import mechanics, memory_rows, processed_count
from test_postgres_transactions import local_db, memory_update


def fact(value="Sold"):
    return {"attribute_name":"ownership_status","value":value,"status":"active","importance":4,
            "confidence":1,"source_chat_id":"current-source","last_mentioned_at":"2026-10-09T12:00:00Z","timeline":[]}


def record(name, aliases=()):
    return {"id":uuid4(),"entity_id":"canonical-"+uuid4().hex,"entity_name":name,
            "entity_type":"vehicle","entity_domain":"assets_and_possessions",
            "payload":{"aliases":list(aliases),"facts":[fact("Owned")]},"similarity":0.99}


def entity():
    return {"entity_id":"veh_01","entity_name":"Weak vehicle reference","entity_type":"vehicle",
            "entity_domain":"assets_and_possessions","aliases":[],"facts":[fact()]}


def decision(kind, targets=(), evidence="EXPLICIT", confidence=1, equal=False, new_quote=None, accepted_aliases=()):
    return {"extracted_entity_id":"veh_01","kind":kind,"reasoning":"Prescribed synthetic evidence",
            "targets":[{"db_id":str(row["id"]),"resolution_confidence":confidence,"evidence":evidence} for row in targets],
            "applies_equally_to_all":equal,"explicitly_new":new_quote is not None,"new_entity_evidence":new_quote,
            "accepted_aliases":list(accepted_aliases)}


def resolve(monkeypatch, records, message, result, recent=(), incoming=None):
    calls=[]
    def model(system_prompt,user_text,schema):
        data=json.loads(user_text);calls.append(data)
        assert schema is ResolutionResponse
        assert "Similarity/type alone" in system_prompt
        return schema.model_validate({"resolutions":[result]})
    monkeypatch.setattr(Tomte,"_db",Mock())
    monkeypatch.setattr(Tomte,"_llm",SimpleNamespace(parse_structured=mutation_aware(model)))
    monkeypatch.setattr("tomte.engine.workflows.consolidation.get_dictionary_service",lambda:Mock())
    workflow=ConsolidationWorkflow()
    source = (incoming or entity()).get("facts", [{}])[0].get("source_chat_id") or "current-source"
    result = deepcopy(result)
    if result.get("accepted_aliases") and "alias_evidence" not in result:
        result["alias_evidence"] = [{"alias": alias, "source_chat_id": source, "evidence_quote": message}
                                    for alias in result["accepted_aliases"]]
    op=workflow._reconcile_with_golden_records("tenant","user",[incoming or entity()],records,
        [{"role":"user","content":text} for text in recent],
        [{"role":"user","content":message,"item_id":source}])
    # Identity and truth-state are separate stages. Supply deterministic local
    # semantic plans while exercising production lifecycle application.
    from datetime import datetime, timezone
    facts=(incoming or entity()).get("facts", [])
    source_items=[]
    for f in facts:
        source=f.get("source_chat_id") or "current-source"
        if not any(m["item_id"]==source for m in source_items):
            stamp=datetime.fromisoformat(f.get("last_mentioned_at", "2026-10-09T12:00:00Z").replace("Z", "+00:00")).timestamp()
            source_items.append({"item_id":source,"timestamp":stamp,"role":"user","content":message})
    if not source_items:
        source_items=[{"item_id":"current-source","timestamp":1791547200,"role":"user","content":message}]
    op=workflow._plan_fact_mutations(op,{"CURRENT_BATCH":source_items,
        "RECENT_CONTEXT_REFERENCE_ONLY":[{"role":"user","content":x} for x in recent]},"unit-batch")
    return op,calls,workflow


@pytest.mark.parametrize("scenario", ["exact", "alias", "pronoun", "truck"])
def test_one_target_contract_does_not_merge_similar_vehicles(monkeypatch,scenario):
    tesla,truck=record("Tesla",["Little Dog"]),record("Ford truck",["truck"])
    message,evidence,recent,target={
        "exact":("I sold my Tesla.","EXPLICIT",[],tesla),
        "alias":("I sold Little Dog.","ALIAS",[],tesla),
        "pronoun":("I painted it red.","RESOLVED_REFERENCE",["I bought a Tesla."],tesla),
        "truck":("I sold the truck.","ALIAS",[],truck),
    }[scenario]
    ops,calls,_=resolve(monkeypatch,[tesla,truck],message,decision("MATCH_ONE",[target],evidence),recent)
    assert len(ops)==1 and ops[0]["db_id"]==str(target["id"])
    assert ops[0]["entity_id"]==target["entity_id"]
    assert calls[0]["CURRENT_BATCH"][0]["content"]==message
    assert len(calls[0]["CANDIDATES_BY_OBJECT"]["veh_01"])==2


def test_explicit_plural_expands_only_strong_active_targets(monkeypatch):
    tesla,truck=record("Tesla Model Y"),record("Ford F-150")
    truck.update(entity_type="truck", entity_domain="unstructured_observations")
    incoming=entity()
    incoming.update(entity_name="both cars", entity_type="unspecified_entity",
                    entity_domain="unstructured_observations", aliases=["shared group label"])
    ops,_,_=resolve(monkeypatch,[tesla,truck],"I sold them both.",
        decision("MATCH_MANY",[tesla,truck],"GROUP_EXPLICIT",equal=True),
        ["Tesla Model Y and Ford F-150 are my cars."],incoming)
    assert {op["db_id"] for op in ops}=={str(tesla["id"]),str(truck["id"])}
    assert all(project_facts(op["payload"]["facts"])[0]["value"]=="Sold" for op in ops)
    for op in ops:
        target=next(row for row in [tesla,truck] if str(row["id"])==op["db_id"])
        for key in ("entity_id", "entity_name", "entity_type", "entity_domain"):
            assert op[key]==op["payload"][key]==target[key]
        assert op["payload"]["aliases"]==target["payload"]["aliases"]
    assert len({id(op["payload"]) for op in ops})==2


def test_my_cars_preserves_uncertainty_for_unmentioned_honda(monkeypatch):
    tesla,truck,honda=record("Tesla"),record("truck"),record("Honda")
    result=decision("MATCH_MANY",[tesla,truck],"GROUP_EXPLICIT",equal=True)
    result["targets"].append({"db_id":str(honda["id"]),"resolution_confidence":0.6,"evidence":"GROUP_INFERRED"})
    incoming=entity();incoming["facts"][0].update(attribute_name="paint_color",value="red")
    ops,_,_=resolve(monkeypatch,[tesla,truck,honda],"I painted my cars red. I like red.",result,
                    ["We discussed Tesla and truck."],incoming)
    assert {op["db_id"] for op in ops}=={str(tesla["id"]),str(truck["id"])}
    assert all(any(f["attribute_name"]=="paint_color" and f["value"]=="red" for f in op["payload"]["facts"]) for op in ops)


@pytest.mark.parametrize("message",["One of my cars has a broken window.","I sold both vehicles.","I painted Little Dog red."])
def test_unresolved_never_defaults_to_insert(monkeypatch,message):
    ops,_,workflow=resolve(monkeypatch,[record("Tesla"),record("truck"),record("motorcycle")],message,decision("UNRESOLVED"))
    assert ops==[] and workflow.last_resolution["resolutions"][0]["kind"]=="UNRESOLVED"


def test_model_abstains_on_unidentified_member_without_group_fanout(monkeypatch):
    records=[record("Tesla"),record("truck")]
    ops,_,_=resolve(monkeypatch,records,"One of my cars has a broken window.",
        decision("UNRESOLVED"),["Tesla and truck are my cars."])
    assert ops==[]


@pytest.mark.parametrize("safe,active",[(False,True),(False,False)])
def test_plural_requires_model_shared_update_approval(monkeypatch,safe,active):
    records=[record("Tesla"),record("truck")]
    ops,_,_=resolve(monkeypatch,records,"I sold them both.",decision("MATCH_MANY",records,"GROUP_EXPLICIT",equal=safe),
                    ["Tesla and truck"] if active else [])
    assert ops==[]


@pytest.mark.parametrize("evidence,confidence",[("SEMANTIC_CANDIDATE",1),("GROUP_INFERRED",1),("EXPLICIT",0.5)])
def test_similarity_temporary_id_and_weak_confidence_do_not_license_writes(monkeypatch,evidence,confidence):
    row=record("Tesla");row["entity_id"]="veh_01"
    ops,_,_=resolve(monkeypatch,[row],"I painted Little Dog red.",decision("MATCH_ONE",[row],evidence,confidence))
    assert ops==[]  # Fact.confidence=1 and matching bookkeeping strings cannot bypass identity policy.


def test_new_requires_explicit_current_introduction(monkeypatch):
    message="I bought a new BMW."
    incoming=entity()
    incoming.update(entity_name="BMW", entity_type="car", aliases=["my new BMW"])
    ops,_,_=resolve(monkeypatch,[],message,decision("NEW",new_quote=message),incoming=incoming)
    assert len(ops)==1 and ops[0]["mode"]=="INSERT" and "db_id" not in ops[0]
    for key in ("entity_id", "entity_name", "entity_type", "entity_domain"):
        assert ops[0][key]==ops[0]["payload"][key]==incoming[key]
    assert ops[0]["payload"]["aliases"]==[]
    ops,_,_=resolve(monkeypatch,[],message,decision("NEW",new_quote="I bought a Tesla."))
    assert ops==[]
    ops,_,_=resolve(monkeypatch,[],message,decision("NEW"))
    assert ops==[]


def test_missing_result_does_not_create_a_false_memory(monkeypatch):
    monkeypatch.setattr(Tomte,"_db",Mock());monkeypatch.setattr(Tomte,"_llm",SimpleNamespace(parse_structured=lambda **kw:ResolutionResponse(resolutions=[])))
    monkeypatch.setattr("tomte.engine.workflows.consolidation.get_dictionary_service",lambda:Mock())
    assert ConsolidationWorkflow()._reconcile_with_golden_records("t","u",[entity()],[])==[]


def test_foreign_or_unoffered_target_fails_before_persistence(monkeypatch):
    with pytest.raises(ValueError,match="offered scoped candidates"):
        resolve(monkeypatch,[record("Tesla")],"I sold Tesla.",decision("MATCH_ONE",[record("Foreign tenant")]))


def test_truncated_explicit_set_cannot_receive_partial_group_write(monkeypatch):
    records=[record(name) for name in ["Tesla","truck","Honda","BMW","Volvo"]]
    for row in records:row["identity_matches_total"]=6
    ops,calls,_=resolve(monkeypatch,records,"I sold all of them.",decision("MATCH_MANY",records,"GROUP_EXPLICIT",equal=True),
                    ["Tesla truck Honda BMW Volvo and Audi"])
    assert ops==[] and calls[0]["CANDIDATE_SET_TRUNCATED"]["veh_01"]


def test_response_cardinality_is_strict_and_fact_confidence_separate():
    with pytest.raises(ValueError):ResolutionResponse.model_validate({"resolutions":[decision("MATCH_ONE")]})
    row=record("Tesla")
    with pytest.raises(ValueError):ResolutionResponse.model_validate({"resolutions":[decision("MATCH_MANY",[row,row])]})
    assert "resolution_confidence" not in fact()


def test_candidate_search_is_bounded_normalized_and_tenant_user_scoped(local_db):
    case=local_db
    for i in range(8):
        op=memory_update(case.tenant,case.users[0]);op["entity_id"]=op["payload"]["entity_id"]=f"vehicle-{i}"
        op["entity_domain"]=op["payload"]["entity_domain"]="assets_and_possessions"
        op["entity_name"]=op["payload"]["entity_name"]= "  Tesla   Model Y  " if i==7 else f"car-{i}"
        op["payload"]["aliases"]=["Little Dog"] if i==7 else []
        assert case.db.commit_graph_updates([op])
    foreign=memory_update(case.tenant,case.users[1]);foreign["entity_name"]="Little Dog"
    assert case.db.commit_graph_updates([foreign])
    args=(case.tenant,case.users[0],["little dog"],"assets_and_possessions",op["embedding"],"I sold Little Dog.")
    rows=case.db.get_resolution_candidates(*args)
    assert len(rows)==5 and rows[0]["entity_id"]=="vehicle-7" and rows[0]["exact_match"]
    assert {row["entity_id"] for row in rows} <= {f"vehicle-{i}" for i in range(8)}
    rows=case.db.get_resolution_candidates(case.tenant,case.users[0],["tesla model y"],"assets_and_possessions",op["embedding"],"")
    assert rows[0]["entity_id"]=="vehicle-7"
    assert case.db.get_resolution_candidates(case.tenant+"-other",case.users[0],["little dog"],"assets_and_possessions",op["embedding"],"")==[]
    with pytest.raises(ValueError):case.db.get_resolution_candidates(*args,limit=6)


@pytest.mark.parametrize("fail_first",[False,True])
def test_real_group_update_replay_and_uncertainty_leave_honda_untouched(mechanics,monkeypatch,fail_first):
    case=mechanics
    for name in ["Tesla","truck","Honda"]:
        op=memory_update(case.tenant,case.user)
        op.update(entity_id=name,entity_name=name,entity_type="vehicle",entity_domain="assets_and_possessions",embedding=case.llm.vector())
        op["payload"].update(entity_id=name,entity_name=name,entity_type="vehicle",entity_domain="assets_and_possessions",aliases=[name],facts=[fact("Owned")])
        assert case.db.commit_graph_updates([op])
    case.cache.append_chat_history(case.tenant,case.session,"Tesla and truck are my cars.","user",case.user)
    message="I sold them both."
    calls=[]
    def model(system_prompt,user_text,schema,taxonomy=None,lexicon=None):
        data=json.loads(user_text);calls.append(data)
        if schema is ExtractionMemorySchema:
            return schema.model_validate({"assets_and_possessions":[{"entity_id":"veh_01","entity_name":"both cars",
                "entity_type":"vehicle","entity_domain":"assets_and_possessions","facts":[{
                "attribute_name":"ownership_status","value":"Sold","confidence":1}]}]})
        candidates=data["CANDIDATES_BY_OBJECT"]["veh_01"]
        selected=[row for row in candidates if row["name"] in {"Tesla","truck"}]
        assert len(selected)==2 and all(row["name"]!="Honda" for row in selected)
        result=decision("MATCH_MANY",[{"id":row["id"]} for row in selected],"GROUP_EXPLICIT",equal=True)
        return schema.model_validate({"resolutions":[result]})
    monkeypatch.setattr(case.llm,"parse_structured",mutation_aware(model))
    monkeypatch.setattr(case.llm,"is_junk",lambda text:False)
    monkeypatch.setattr(case.llm,"embed_text",lambda text:case.llm.vector())
    monkeypatch.setattr(case.llm,"embed_batch",lambda texts:[case.llm.vector() for _ in texts])
    # The whole graph API must not be used by the new runtime path.
    monkeypatch.setattr(case.db,"get_user_graph_skeleton",lambda *args:pytest.fail("Unbounded graph read"))
    ChatHookWorkflow().on_chat(case.tenant,case.session,message,ChatSenderRole.USER,case.user)
    frozen=case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)
    workflow=ConsolidationWorkflow()
    if fail_first:
        before=memory_rows(case)
        original=case.db.commit_graph_updates
        def fail_second(updates):
            assert len(updates)==2
            assert original(updates[:1])  # First child writes, but cannot commit the outer scope.
            raise RuntimeError("Synthetic second-target failure")
        monkeypatch.setattr(case.db,"commit_graph_updates",fail_second)
        with pytest.raises(RuntimeError,match="second-target failure"):
            workflow.run_consolidation_cycle(case.tenant,case.user,frozen)
        assert memory_rows(case)==before and processed_count(case)==0
        assert case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)==frozen
        first_input=deepcopy(calls[-1])
        monkeypatch.setattr(case.db,"commit_graph_updates",original)
    assert workflow.run_consolidation_cycle(case.tenant,case.user,frozen)
    if fail_first:
        assert calls[-1]==first_input  # Failed preparation retries use the same frozen sources/context/candidates.
    rows={row["entity_id"]:row for row in memory_rows(case)}
    assert project_facts(rows["Tesla"]["payload"]["facts"])[0]["value"]==project_facts(rows["truck"]["payload"]["facts"])[0]["value"]=="Sold"
    assert rows["Honda"]["payload"]["facts"][0]["value"]=="Owned"
    # Group plans must use each target's own assertion ID, never copy the Tesla ID.
    prior_ids=[]
    for name in ("Tesla", "truck"):
        prior,current=rows[name]["payload"]["facts"]
        assert prior["value"]=="Owned" and prior["assertion_state"]=="HISTORICAL"
        assert current["supersedes_fact_id"]==prior["fact_id"] and current["value"]=="Sold"
        prior_ids.append(prior["fact_id"])
    assert len(set(prior_ids))==2
    before=memory_rows(case);call_count=len(calls)
    assert workflow.run_consolidation_cycle(case.tenant,case.user,frozen)
    assert memory_rows(case)==before and len(calls)==call_count and processed_count(case)==1
    # A later ambiguous batch is processed observably with zero durable memory operations.
    assert case.cache.clear_session_4_consolidation(case.tenant,case.user,case.session,frozen)
    monkeypatch.setattr(case.llm,"parse_structured",lambda system_prompt,user_text,schema,**kwargs:
        model(system_prompt,user_text,schema,**kwargs) if schema is ExtractionMemorySchema else
        schema.model_validate({"resolutions":[decision("UNRESOLVED")]}))
    ChatHookWorkflow().on_chat(case.tenant,case.session,"One of my cars has a broken window.",ChatSenderRole.USER,case.user)
    frozen=case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)
    assert workflow.run_consolidation_cycle(case.tenant,case.user,frozen)
    assert memory_rows(case)==before and processed_count(case)==2


def test_retry_context_is_stable_isolated_and_removed_on_ack(mechanics):
    case=mechanics
    other_session=case.session+"-other";other_user=case.user+"-other";other_tenant=case.tenant+"-other"
    scoped=[(case.tenant,case.user,other_session),(case.tenant,other_user,case.session),(other_tenant,case.user,case.session)]
    keys=[]
    try:
        for tenant,user,session in scoped:
            key=generate_common_key(tenant,session,user);keys.append(key)
            case.cache.append_chat_history(tenant,session,"Foreign context must never enter this batch.","user",user)
        case.cache.append_chat_history(case.tenant,case.session,"Tesla is my vehicle.","user",case.user)
        history=case.cache.get_chat_history(case.tenant,case.session,case.user)
        assert [x["content"] for x in history]==["Tesla is my vehicle."]
        case.cache.enqueue_for_consolidation(case.tenant,case.user,case.session,"I painted it red.","user",case.llm.vector(),history)
        frozen=case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)
        case.cache.enqueue_for_consolidation(case.tenant,case.user,case.session,"Future message","user",case.llm.vector(),[{"role":"user","content":"future context"}])
        assert case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)==frozen
        assert frozen[0]["recent_context"]==[{"role":"user","content":"Tesla is my vehicle."}]
        assert case.cache.clear_session_4_consolidation(case.tenant,case.user,case.session,frozen)
        assert not case.cache.client.exists(generate_consolidation_batch_key(case.tenant,case.user,case.session))
        later=case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)
        assert later[0]["recent_context"]==[{"role":"user","content":"future context"}]
        assert case.cache.clear_session_4_consolidation(case.tenant,case.user,case.session,later)
        assert not case.cache.client.exists(generate_consolidation_key(case.tenant,case.user,case.session))
    finally:
        case.cache.client.delete(*keys)


@pytest.mark.parametrize("mode", ["MUTATE", "REINFORCE"])
def test_match_one_preserves_metadata_and_merges_nickname_facts(monkeypatch,mode):
    target=record("Tesla Model Y", ["my car", "old nickname"])
    old_fact=fact("blue")
    old_fact.update(attribute_name="paint_color", source_chat_id="old-source")
    target["payload"]["facts"].append(old_fact)
    incoming=entity()
    incoming.update(entity_name="Little Dog", entity_type="unspecified_entity",
                    entity_domain="unstructured_observations", aliases=["my car", "my car"])
    current_fact=fact("red" if mode=="MUTATE" else "blue")
    current_fact.update(attribute_name="paint_color", source_chat_id="new-source")
    incoming["facts"]=[current_fact]
    before_target,before_incoming=deepcopy(target),deepcopy(incoming)
    ops,_,_=resolve(monkeypatch,[target],"I named it Little Dog.",
                    decision("MATCH_ONE",[target],"ALIAS",accepted_aliases=["Little Dog"]),incoming=incoming)
    op,=ops
    assert op["mode"]==mode and op["db_id"]==str(target["id"])
    for key in ("entity_id", "entity_name", "entity_type", "entity_domain"):
        assert op[key]==op["payload"][key]==target[key]
    assert set(op["payload"]["aliases"])=={"my car", "old nickname", "Little Dog"}
    assert len(op["payload"]["aliases"])==3
    assert any(f["attribute_name"]=="ownership_status" and f["value"]=="Owned" for f in project_facts(op["payload"]["facts"]))
    paint=next(f for f in project_facts(op["payload"]["facts"]) if f["attribute_name"]=="paint_color")
    assert paint["value"]==current_fact["value"] and paint["assertion_state"]=="ACTIVE"
    assert any(e["source_chat_id"]=="new-source" for e in paint["evidence"])
    if mode=="MUTATE":
        historical=next(f for f in op["payload"]["facts"] if f["value"]=="blue")
        assert historical["assertion_state"]=="HISTORICAL" and paint["supersedes_fact_id"]==historical["fact_id"]
    else:
        assert paint["source_chat_id"]=="old-source" and paint["mention_count"]==2
    assert op["extracted_facts"]==[{"attribute_name":"paint_color"}]
    assert target==before_target and incoming==before_incoming


@pytest.mark.parametrize("mode", ["MUTATE", "REINFORCE"])
def test_matched_metadata_stays_consistent_in_postgres(local_db,monkeypatch,mode):
    case=local_db
    seed=memory_update(case.tenant,case.users[0])
    for key,value in {"entity_id":"canonical-tesla", "entity_name":"Tesla Model Y",
                      "entity_type":"vehicle", "entity_domain":"assets_and_possessions"}.items():
        seed[key]=seed["payload"][key]=value
    seed["payload"].update(aliases=["my car"], facts=[fact("Owned")])
    assert case.db.commit_graph_updates([seed])
    target,=case.db.get_user_graph_skeleton(case.tenant,case.users[0])
    incoming=entity()
    incoming.update(entity_name="Little Dog", entity_type="unidentified_entity",
                    entity_domain="unstructured_observations", aliases=["Little Dog", "my car"])
    incoming["facts"]=[fact("Owned") if mode=="REINFORCE" else {
        **fact("red"), "attribute_name":"paint_color"}]
    ops,_,_=resolve(monkeypatch,[target],"I named it Little Dog.",
                    decision("MATCH_ONE",[target],"ALIAS",accepted_aliases=["Little Dog"]),incoming=incoming)
    op,=ops
    assert op["mode"]==mode
    op.update(tenant_id=case.tenant,user_id=case.users[0],embedding=seed["embedding"])
    assert case.db.commit_graph_updates(ops)
    persisted,=case.db.get_user_graph_skeleton(case.tenant,case.users[0])
    assert persisted["id"]==target["id"]
    for key in ("entity_id", "entity_name", "entity_type", "entity_domain"):
        assert persisted[key]==persisted["payload"][key]==target[key]
    assert set(persisted["payload"]["aliases"])=={"my car", "Little Dog"}
    assert all(any(old["attribute_name"]==f["attribute_name"] and old["value"]==f["value"]
                   for old in project_facts(persisted["payload"]["facts"])) for f in incoming["facts"])
    visible=case.observer.execute(
        "SELECT id, entity_id, entity_name, entity_domain, entity_type, payload "
        "FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s",
        (case.tenant,case.users[0]),
    ).fetchone()
    assert visible==tuple(persisted[key] for key in (
        "id", "entity_id", "entity_name", "entity_domain", "entity_type", "payload"))
    retrieved,=case.db.get_recent_entities(case.tenant,case.users[0])
    assert retrieved["id"]==str(target["id"])
    for key in ("entity_id", "entity_name", "entity_type", "entity_domain"):
        assert retrieved[key]==target[key]


def test_alias_schema_backward_compatible_default():
    result = EntityResolution(extracted_entity_id="e", kind="NEW", reasoning="fixture")
    assert result.accepted_aliases == []


@pytest.mark.parametrize("kind,aliases", [("UNRESOLVED", ["Nickname"]),
    ("NEW", [""]), ("NEW", ["  "]), ("NEW", ["Nickname", "Nickname"])])
def test_invalid_alias_schema_rejected(kind, aliases):
    with pytest.raises(ValueError):
        EntityResolution(extracted_entity_id="e", kind=kind, reasoning="fixture", accepted_aliases=aliases)


def test_match_many_cannot_authorize_aliases():
    rows = [record("Tesla"), record("Honda")]
    with pytest.raises(ValueError):
        EntityResolution.model_validate(decision("MATCH_MANY", rows, equal=True, accepted_aliases=["Group"]))


@pytest.mark.parametrize("alias,text", [("Little Dog", "I named it Little Dog."),
    ("小蓝", "我把这辆车叫小蓝了"), ("Ann", "The nickname is Ann-Marie")])
def test_alias_provenance_does_not_require_standalone_word(alias, text):
    incoming = entity(); incoming["aliases"] = [alias]
    citation = AliasEvidence(alias=alias, source_chat_id="u1", evidence_quote=text)
    assert ConsolidationWorkflow._approved_aliases([alias], incoming,
        [{"role":"user", "content":text, "item_id":"u1"}], [citation]) == [alias]


@pytest.mark.parametrize("defect", ["missing", "duplicate", "foreign", "assistant", "wrong_quote", "blank", "unoffered", "different_alias"])
def test_alias_source_contract_rejects_individual_invalid_citation(defect):
    incoming = entity(); incoming["aliases"] = [] if defect == "unoffered" else ["Little Dog"]
    citation = AliasEvidence(alias="Other" if defect == "different_alias" else "Little Dog",
        source_chat_id="foreign" if defect == "foreign" else "u1",
        evidence_quote=" " if defect == "blank" else "Invented quote" if defect == "wrong_quote" else "I named it Little Dog.")
    citations = [] if defect == "missing" else [citation, citation] if defect == "duplicate" else [citation]
    assert ConsolidationWorkflow._approved_aliases(["Little Dog"], incoming,
        [{"role":"assistant" if defect == "assistant" else "user", "content":"I named it Little Dog.", "item_id":"u1"}], citations) == []


def test_rejected_identity_target_cannot_authorize_alias(monkeypatch):
    target = record("Tesla")
    incoming = entity()
    incoming["aliases"] = ["Little Dog"]
    ops, _, _ = resolve(monkeypatch, [target], "I named my Tesla Little Dog.",
        decision("MATCH_ONE", [target], confidence=.2, accepted_aliases=["Little Dog"]), incoming=incoming)
    assert ops == []


def test_new_alias_does_not_bypass_user_introduction(monkeypatch):
    incoming = entity()
    incoming["aliases"] = ["Little Dog"]
    ops, _, _ = resolve(monkeypatch, [], "Little Dog is my nickname for it.",
        decision("NEW", new_quote="unsupported introduction", accepted_aliases=["Little Dog"]), incoming=incoming)
    assert ops == []


def test_preparation_preserves_stored_alias_order_and_only_appends_approved():
    target = record("Tesla", ["existing Z", "existing A", "existing Z"])
    incoming = entity()
    incoming.update(entity_name="different name", aliases=["unapproved"])
    op = ConsolidationWorkflow._prepare_operation("tenant", "user", incoming, target,
                                                  ["Tesla", "Little Dog", "Little Dog"])
    assert op["payload"]["aliases"] == ["existing Z", "existing A", "existing Z", "Little Dog"]
    assert target["payload"]["aliases"] == ["existing Z", "existing A", "existing Z"]



@pytest.mark.parametrize("message", ["I'm not training for the marathon anymore.", "我不再为马拉松训练了"])
def test_model_resolved_paraphrase_reaches_retraction_without_literal_anchor(monkeypatch, message):
    from tomte.engine.core.assertions import adapt_assertions
    row = record("Marathon training")
    row["payload"]["facts"] = adapt_assertions([dict(fact("training"), attribute_name="training_status")], str(row["id"]))
    fact_id = row["payload"]["facts"][0]["fact_id"]
    incoming = entity(); incoming["facts"] = [dict(fact("not training"), attribute_name="training_status")]
    current = [{"item_id":"current-source", "role":"user", "content":message, "timestamp":1791547200}]
    calls = []
    def model(system_prompt, user_text, schema):
        calls.append(schema.__name__)
        if schema is ResolutionResponse:
            return schema.model_validate({"resolutions":[decision("MATCH_ONE", [row], "RESOLVED_REFERENCE", .98)]})
        target, = json.loads(user_text)["TARGETS"]
        return schema.model_validate({"plans":[{"target_key":target["target_key"], "operations":[{
            "operation":"RETRACT", "target_fact_id":fact_id, "source_chat_ids":["current-source"],
            "interpretation_confidence":1, "evidence_type":"USER_EXPLICIT"}]}]})
    monkeypatch.setattr(Tomte,"_db",Mock()); monkeypatch.setattr(Tomte,"_llm",SimpleNamespace(parse_structured=model))
    monkeypatch.setattr("tomte.engine.workflows.consolidation.get_dictionary_service",lambda:Mock())
    workflow = ConsolidationWorkflow()
    prepared = workflow._reconcile_with_golden_records("t","u",[incoming],[row, record("Other activity")], [], current)
    assert prepared[0]["db_id"] == str(row["id"]) and prepared[0]["payload"]["aliases"] == []
    planned = workflow._plan_fact_mutations(prepared,{"CURRENT_BATCH":current,"RECENT_CONTEXT_REFERENCE_ONLY":[]},"paraphrase-batch")
    assert calls == ["ResolutionResponse", "MutationPlanResponse"]
    old, = planned[0]["payload"]["facts"]
    assert old["fact_id"] == fact_id and old["assertion_state"] == "RETRACTED"
    assert project_facts(planned[0]["payload"]["facts"]) == []


@pytest.mark.parametrize("message", ["I sold my Tesal.", "我把小蓝卖了", "I sold it."])
def test_prescribed_identity_acceptance_does_not_require_surface_name(monkeypatch, message):
    row = record("Tesla Model Y")
    ops, _, _ = resolve(monkeypatch,[row,record("Honda Civic")],message,
        decision("MATCH_ONE",[row],"RESOLVED_REFERENCE",.98))
    assert len(ops) == 1 and ops[0]["db_id"] == str(row["id"])
    assert ops[0]["payload"]["aliases"] == []


@pytest.mark.parametrize("message", ["两辆车都涂红了", "Paint both of them red."])
def test_model_group_scope_without_literal_names(monkeypatch, message):
    records = [record("Tesla Model Y"), record("Honda Civic")]
    ops, _, _ = resolve(monkeypatch,records,message,decision("MATCH_MANY",records,"GROUP_EXPLICIT",equal=True))
    assert {op["db_id"] for op in ops} == {str(row["id"]) for row in records}
    assert all(op["payload"]["aliases"] == [] for op in ops)


@pytest.mark.parametrize("kind", ["MATCH_MANY", "UNRESOLVED"])
def test_group_or_unresolved_cannot_cite_aliases(kind):
    targets = [record("A"),record("B")] if kind == "MATCH_MANY" else []
    data = decision(kind, targets, equal=True)
    data["alias_evidence"] = [{"alias":"Group", "source_chat_id":"u1", "evidence_quote":"Group"}]
    with pytest.raises(ValueError, match="one resolved entity"):
        EntityResolution.model_validate(data)


def test_alias_quote_proves_source_not_alias_semantics():
    incoming = entity(); incoming["aliases"] = ["Blue"]
    quote = "我把车叫小蓝了"
    citation = AliasEvidence(alias="Blue",source_chat_id="u1",evidence_quote=quote)
    assert ConsolidationWorkflow._approved_aliases(["Blue"],incoming,
        [{"item_id":"u1","role":"user","content":quote}],[citation]) == ["Blue"]
    # This verifies the owner boundary, not whether a model should translate this name.
    assert ConsolidationWorkflow._approved_aliases([],incoming,
        [{"item_id":"u1","role":"user","content":quote}],[citation]) == []
