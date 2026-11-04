"""Model-owned corrections and structural safety; no Python language heuristics.

Historical English-rule checks are replaced by the same labeled scenarios with
prescribed decisions. They do not measure LLM accuracy. Saved provider evidence
is unchanged and still demonstrates a structurally valid wrong correction.
"""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from memory_fixtures import simple_entity, simple_fact, field_entries, fact_text, value
from tomte.engine.core.assertions import FactMutation, MutationPlanResponse, apply_mutations
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from tomte.engine.core.relationships import prepare_proposals
from test_consolidation_mechanics import mechanics, processed_count
from test_persistence_authorization_contract import contract, seed, freeze, payloads, rows, proposal

ROOT=Path(__file__).resolve().parents[1]
CASES=json.loads((ROOT/'evals/correction_safety/cases.json').read_text(encoding='utf-8'))


@pytest.mark.parametrize('case',CASES,ids=lambda c:c['case_id'])
def test_multi_domain_prescribed_correction_or_abstention(case):
    fact=simple_fact(case['attribute'],case['old'])
    batch=[dict(item_id='current',role='user',content=case['text'],timestamp=1791504060)]
    before=deepcopy(fact)
    operation=FactMutation(operation='UPDATE',target_fact_id=fact['fact_id'],
        fields=field_entries(case['new']),source_chat_ids=['current'],change_type='correction')
    # Expected decisions come from evaluation labels, never text matching.
    result=apply_mutations([fact],[operation] if case['expected_correction'] else [],batch,'scope','batch')
    assert result[0]['fact_id']==fact['fact_id']
    assert fact_text(result[0])==(case['new'] if case['expected_correction'] else case['old'])
    assert result[0]['history_values']==[] and fact==before


def replay():
    """Restate historical decisions in the new test contract, retaining strings,
    sources and selected fact UUIDs. This is never a runtime memory adapter.
    """
    fixture=json.loads((ROOT/'evals/correction_safety/honda_tesla_failure.json').read_text(encoding='utf-8'))
    call=fixture['stages']['MutationPlanResponse'];operations=[]
    for target in call['input']['TARGETS']:
        identity=target['identity']
        facts=[simple_fact(f['attribute_name'],f['value'],fact_id=f['fact_id'])
               for f in target['assertions'] if f['assertion_state']=='ACTIVE']
        payload=simple_entity(identity['entity_name'],kind=identity['entity_type'],domain=identity['entity_domain'],facts=facts)
        record={'id':payload['entity_id'],**{k:payload[k] for k in ('entity_id','entity_name','entity_domain','entity_type')},'payload':payload}
        incoming={**identity,'aliases':[],'facts':[
            {'attribute_name':f['attribute_name'],'value':value(f['value']),'source_chat_id':f['source_chat_id']}
            for f in target['incoming_semantics']]}
        operations.append(ConsolidationWorkflow._prepare_operation('synthetic','user',incoming,record))
    output={'plans':[]}
    for plan in call['output']['plans']:
        converted=[]
        for old in plan['operations']:
            kind=old['operation']
            new={key:old[key] for key in ('target_fact_id','attribute_name','source_chat_ids')}
            new['operation']={'ADD':'ADD','REINFORCE':'MENTION','CORRECT':'UPDATE','SUPERSEDE':'UPDATE'}[kind]
            new['fields']=field_entries(old['new_value']) if old['new_value'] is not None else []
            if kind in {'CORRECT','SUPERSEDE'}:
                new.update(change_type='correction' if kind=='CORRECT' else 'real_change',update_mode='replace_value')
            converted.append(new)
        output['plans'].append({'target_key':plan['target_key'],'operations':converted})
    workflow=ConsolidationWorkflow.__new__(ConsolidationWorkflow)
    workflow.llm=SimpleNamespace(parse_structured=lambda **kw:MutationPlanResponse.model_validate(output))
    data=dict(CURRENT_BATCH=call['input']['CURRENT_BATCH'],RECENT_CONTEXT_REFERENCE_ONLY=call['input']['REFERENCE_ONLY_CONTEXT'])
    operations,_=prepare_proposals(operations)
    return workflow,operations,data,fixture,output


def test_structurally_valid_recorded_wrong_correction_exposes_semantic_risk():
    """A wrong model decision can pass integrity checks; NOT safety evidence."""
    workflow,operations,data,fixture,output=replay();before=deepcopy(operations)
    assert fixture['recorded_correction']['interpretation_confidence']==0.88
    honda_before=next(op for op in operations if op['entity_name']=='Honda Civic')
    old=next(f for f in honda_before['payload']['facts'] if f['attribute_name']=='purchase_year')
    assert fact_text(old)=='2021'
    planned=workflow._plan_fact_mutations(operations,data,'recorded-risk')
    assert operations==before
    honda=next(op for op in planned if op['entity_name']=='Honda Civic')
    year=next(f for f in honda['payload']['facts'] if f['fact_id']==old['fact_id'])
    assert fact_text(year)=='2024' and year['history_values']==[]


@pytest.mark.parametrize('defect',['assistant','foreign_source','foreign_fact'])
def test_correction_structural_authorization_cannot_be_replaced_by_confidence(defect):
    fact=simple_fact('purchase_year','2021');before=deepcopy(fact)
    batch=[dict(item_id='current',role='assistant' if defect=='assistant' else 'user',content='Synthetic correction',timestamp=1791504060)]
    operation=FactMutation(operation='UPDATE',target_fact_id='unoffered' if defect=='foreign_fact' else fact['fact_id'],
        fields=field_entries('2022'),change_type='correction',source_chat_ids=['foreign' if defect=='foreign_source' else 'current'])
    with pytest.raises(ValueError):apply_mutations([fact],[operation],batch,'scope','batch')
    assert fact==before


@pytest.mark.parametrize('legitimate',[False,True])
def test_model_supported_correction_or_abstention_with_real_persistence(contract,legitimate):
    case=contract;seed(case,'Honda Civic','honda','2021',('Honda',));seed(case)
    before=deepcopy(payloads(case))
    case.extracted=[proposal('Honda Civic',value='2022' if legitimate else '2024')]
    case.matches={'temporary':'Honda Civic'}
    case.plan=lambda target,data:[] if not legitimate else [{
        'operation':'UPDATE','target_fact_id':target['facts'][0]['fact_id'],'fields':field_entries('2022'),
        'change_type':'correction','source_chat_ids':[data['CURRENT_BATCH'][0]['item_id']]}]
    batch=freeze(case,('user','I was wrong: I bought the Honda Civic in 2022.' if legitimate else 'The Honda paperwork seems wrong.'))
    workflow=ConsolidationWorkflow();assert workflow.run_consolidation_cycle(case.tenant,case.user,batch)
    after=payloads(case)
    assert after['Tesla Model Y']==before['Tesla Model Y']
    if legitimate:
        year,=after['Honda Civic']['facts'];prior=before['Honda Civic']['facts'][0]
        assert fact_text(year)=='2022' and year['fact_id']==prior['fact_id'] and year['history_values']==[]
    else:assert after==before
    committed=rows(case)
    assert workflow.run_consolidation_cycle(case.tenant,case.user,batch)
    assert rows(case)==committed and processed_count(case)==1
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)==batch


def test_invalid_fact_reference_rolls_back_an_explicit_outer_transaction(mechanics):
    case=mechanics;workflow,operations,data,fixture,output=replay()
    corrections=[op for plan in output['plans'] for op in plan['operations'] if op.get('change_type')=='correction']
    assert len(corrections)==1
    corrections[0]['target_fact_id']='foreign-unoffered-fact'
    MutationPlanResponse.model_validate(output)
    honda=next(op for op in operations if op['entity_name']=='Honda Civic')
    before=deepcopy(honda['payload']['facts'])
    with pytest.raises(ValueError,match='Target fact'):
        with case.db.transaction():
            case.db.conn.execute('INSERT INTO tomte_user_registry (tenant_id,user_id) VALUES (%s,%s)',(case.tenant,case.user))
            # Proposal filtering is recoverable; strict application failures
            # still roll back an explicitly owned database transaction.
            apply_mutations(honda['payload']['facts'],corrections,data['CURRENT_BATCH'],
                            'synthetic-honda','invalid-outer-transaction')
    assert honda['payload']['facts']==before
    assert case.observer.execute('SELECT count(*) FROM tomte_user_registry WHERE tenant_id=%s AND user_id=%s',(case.tenant,case.user)).fetchone()[0]==0
