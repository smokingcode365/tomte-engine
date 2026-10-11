"""Correction evidence policy and recorded failure replay; no model APIs."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tomte.engine.core.assertions import FactMutation, MutationPlanResponse, apply_mutations, project_facts
from tomte.engine.core.correction_safety import UnsupportedCorrection, validate_corrections
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from test_consolidation_mechanics import mechanics, memory_rows, processed_count

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / 'evals/correction_safety/cases.json').read_text(encoding='utf-8'))


def setup_case(case):
    target = dict(entity_id='target', entity_name=case['target'], payload={'aliases': []})
    other = dict(entity_id='other', entity_name=case['other'], payload={'aliases': []})
    batch = [dict(item_id='current', role='user', content=case['text'], timestamp=1791504060)]
    reference = [dict(role='user', content=case['reference'])]
    fact = dict(fact_id='old', attribute_name=case['attribute'], value=case['old'],
                assertion_state='ACTIVE', source_chat_id='original', last_mentioned_at='2026-10-09T00:00:00Z')
    mutation = FactMutation(operation='CORRECT', target_fact_id='old', attribute_name=case['attribute'],
                            new_value=case['new'], source_chat_ids=['current'],
                            interpretation_confidence=1, evidence_type='RESOLVED_REFERENCE')
    return target, other, batch, reference, fact, mutation


@pytest.mark.parametrize('case', CASES, ids=lambda c: c['case_id'])
def test_multi_domain_correction_evidence(case):
    target, other, batch, reference, fact, mutation = setup_case(case)
    before = deepcopy(fact)
    if case['expected_correction']:
        validate_corrections([mutation], target, [target, other], batch, reference)
        result = apply_mutations([fact], [mutation], batch, 'scope', 'batch')
        assert result[0]['assertion_state'] == 'CORRECTED_INVALID'
        assert project_facts(result)[0]['value'] == case['new']
    else:
        with pytest.raises(UnsupportedCorrection):
            validate_corrections([mutation], target, [target, other], batch, reference)
    assert fact == before


def replay():
    fixture = json.loads((ROOT / 'evals/correction_safety/honda_tesla_failure.json').read_text(encoding='utf-8'))
    call = fixture['stages']['MutationPlanResponse']
    operations = []
    for target in call['input']['TARGETS']:
        identity = target['identity']
        record = next(r for r in fixture['final_records'] if r['entity_id'] == identity['entity_id'])
        facts = []
        for assertion in target['assertions']:
            fact = deepcopy(next(f for f in record['payload']['facts'] if f['fact_id'] == assertion['fact_id']))
            fact.update(assertion)
            facts.append(fact)
        before = {**record, 'payload': {**record['payload'], 'facts': facts}}
        incoming = {**identity, 'facts': target['incoming_semantics'], 'aliases': []}
        operations.append(ConsolidationWorkflow._prepare_operation('synthetic', 'user', incoming, before))
    workflow = ConsolidationWorkflow.__new__(ConsolidationWorkflow)
    workflow.llm = SimpleNamespace(parse_structured=lambda **kw: MutationPlanResponse.model_validate(call['output']))
    data = dict(CURRENT_BATCH=call['input']['CURRENT_BATCH'], RECENT_CONTEXT_REFERENCE_ONLY=call['input']['REFERENCE_ONLY_CONTEXT'])
    return workflow, operations, data, fixture


def test_recorded_honda_tesla_destructive_plan_fails_before_mutation():
    workflow, operations, data, fixture = replay()
    before = deepcopy(operations)
    assert fixture['recorded_correction']['interpretation_confidence'] == 0.88
    with pytest.raises(UnsupportedCorrection):
        workflow._plan_fact_mutations(operations, data, 'recorded-failure')
    assert operations == before
    honda = next(op for op in operations if op['entity_name'] == 'Honda Civic')
    assert any(f['value'] == '2021' and f['assertion_state'] == 'ACTIVE' for f in honda['payload']['facts'])


@pytest.mark.parametrize('operation', ['ADD', 'REINFORCE', 'RETRACT', 'SUPERSEDE'])
def test_other_operations_keep_existing_semantics(operation):
    case = CASES[0]
    target, other, batch, reference, fact, mutation = setup_case(case)
    mutation = mutation.model_copy(update=dict(operation=operation,
        target_fact_id=None if operation == 'ADD' else 'old',
        new_value=None if operation in {'REINFORCE', 'RETRACT'} else case['new']))
    batch[0]['content'] = 'Non-corrective synthetic statement.'
    validate_corrections([mutation], target, [target, other], batch, reference)
    result = apply_mutations([fact], [mutation], batch, 'scope', 'batch')
    assert result[0]['assertion_state'] == {'ADD':'ACTIVE', 'REINFORCE':'ACTIVE', 'RETRACT':'RETRACTED', 'SUPERSEDE':'HISTORICAL'}[operation]


def test_long_gap_and_cross_session_explicit_target_is_supported():
    target, other, batch, reference, fact, mutation = setup_case(CASES[0])
    reference = [{'role':'user', 'content':'Unrelated discussion.'}] * 40
    batch[0]['session_id'] = 'new-session'
    validate_corrections([mutation], target, [target, other], batch, reference)


def test_assistant_assertion_cannot_authorize_correction():
    target, other, batch, reference, fact, mutation = setup_case(CASES[0])
    batch[0]['role'] = 'assistant'
    with pytest.raises(UnsupportedCorrection):
        validate_corrections([mutation], target, [target, other], batch, reference)


def test_non_updated_competing_candidate_vetoes_wrong_binding():
    target, other, batch, reference, fact, mutation = setup_case(CASES[0])
    batch[0]['content'] = 'I was wrong about Honda Civic; Tesla Model Y purchase_year was 2022.'
    with pytest.raises(UnsupportedCorrection):
        validate_corrections([mutation], target, [target, other], batch, reference)


@pytest.mark.parametrize('reference,accepted', [([],False), ([{'role':'assistant','content':'Your Honda Civic was bought in 2021.'}],True)])
def test_invalidation_without_replacement_needs_old_assertion_evidence(reference, accepted):
    target, other, batch, _, fact, mutation = setup_case(CASES[0])
    target['payload']['facts'] = [fact]
    batch[0]['content'] = "No, that's wrong."
    mutation = mutation.model_copy(update={'new_value':None})
    if accepted:
        validate_corrections([mutation], target, [target,other], batch, reference)
    else:
        with pytest.raises(UnsupportedCorrection):
            validate_corrections([mutation], target, [target,other], batch, reference)


def test_wrong_paperwork_is_not_evidence_that_the_stored_year_was_wrong():
    target, other, batch, reference, fact, mutation = setup_case(CASES[0])
    target['payload']['facts'] = [fact]
    batch[0]['content'] = 'The Honda Civic paperwork has the wrong purchase year.'
    mutation = mutation.model_copy(update={'new_value':None})
    with pytest.raises(UnsupportedCorrection):
        validate_corrections([mutation], target, [target,other], batch)


@pytest.mark.parametrize('legitimate', [False, True])
def test_real_persistence_and_frozen_batch_on_correction_decision(mechanics, monkeypatch, legitimate):
    from tomte.engine.core.schema import Entity, Fact
    from tomte.engine.core.extraction import ExtractionMemorySchema
    case = mechanics
    for eid, name, year in [('honda', 'Honda Civic', '2021'), ('tesla', 'Tesla Model Y', '2024')]:
        payload = Entity(entity_id=eid, entity_name=name, entity_domain='assets_and_possessions',
            entity_type='vehicle', facts=[Fact(attribute_name='purchase_year', value=year,
            confidence=1, source_chat_id='original', last_mentioned_at='2026-10-09T00:00:00Z')]).model_dump(mode='json')
        assert case.db.commit_graph_updates([dict(mode='INSERT', tenant_id=case.tenant, user_id=case.user,
            **{k:payload[k] for k in ('entity_id','entity_name','entity_domain','entity_type')},
            payload=payload, embedding=case.llm.vector())])
    _, _, evidence, fixture = replay()
    messages = evidence['CURRENT_BATCH'] if not legitimate else [dict(role='user', content='I was wrong: I bought the Honda Civic in 2022.')]
    for item in messages:
        case.cache.enqueue_for_consolidation(case.tenant, case.user, case.session, item['content'], item['role'], vector=case.llm.vector())
    frozen = case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session)
    new_value = '2022' if legitimate else '2024'
    def model(system_prompt, user_text, schema, **kwargs):
        data = json.loads(user_text)
        if schema.__name__ == 'TerminologyAlignmentResponse':
            return schema()
        if schema is ExtractionMemorySchema:
            source = frozen[0]['item_id'] if legitimate else next(i['item_id'] for i in frozen if 'car I bought in 2024' in i['content'])
            return schema.model_validate({'assets_and_possessions':[dict(entity_id='temporary', entity_name='Honda Civic',
                entity_domain='assets_and_possessions', entity_type='vehicle', facts=[dict(attribute_name='purchase_year', value=new_value, confidence=.9, source_chat_id=source)])]})
        if schema.__name__ == 'ResolutionResponse':
            honda = next(c for c in data['CANDIDATES_BY_OBJECT']['temporary'] if c['name'] == 'Honda Civic')
            return schema.model_validate({'resolutions':[dict(extracted_entity_id='temporary', kind='MATCH_ONE', reasoning='Recorded wrong extraction still resolves the named Honda.',
                targets=[dict(db_id=honda['id'], resolution_confidence=.99, evidence='EXPLICIT')])]})
        target = data['TARGETS'][0]
        sources = [frozen[0]['item_id']] if legitimate else [i['item_id'] for i in frozen if any(text in i['content'] for text in ('paperwork has the wrong','I mean the Honda','car I bought in 2024'))]
        return schema.model_validate({'plans':[dict(target_key=target['target_key'], operations=[dict(operation='CORRECT',
            target_fact_id=target['assertions'][0]['fact_id'], attribute_name='purchase_year', new_value=new_value,
            source_chat_ids=sources, interpretation_confidence=.88, evidence_type='RESOLVED_REFERENCE')])]})
    monkeypatch.setattr(case.llm, 'parse_structured', model)
    monkeypatch.setattr(case.llm, 'is_junk', lambda text: False)
    monkeypatch.setattr(case.llm, 'embed_batch', lambda texts: [case.llm.vector() for _ in texts])
    before = deepcopy(memory_rows(case))
    workflow = ConsolidationWorkflow()
    if legitimate:
        assert workflow.run_consolidation_cycle(case.tenant, case.user, frozen)
        honda = next(r for r in memory_rows(case) if r['payload']['entity_name'] == 'Honda Civic')
        assert project_facts(honda['payload']['facts'])[0]['value'] == '2022'
        assert honda['payload']['facts'][0]['assertion_state'] == 'CORRECTED_INVALID'
        assert processed_count(case) == 1
    else:
        with pytest.raises(UnsupportedCorrection):
            workflow.run_consolidation_cycle(case.tenant, case.user, frozen)
        assert memory_rows(case) == before
        assert processed_count(case) == 0
        visible = case.observer.execute('SELECT entity_id,payload FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s', (case.tenant, case.user)).fetchall()
        assert {eid:payload['facts'][0]['value'] for eid,payload in visible} == {'honda':'2021','tesla':'2024'}
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == frozen


def test_uncaught_rejection_rolls_back_an_explicit_outer_transaction(mechanics):
    case = mechanics
    workflow, operations, data, fixture = replay()
    with pytest.raises(UnsupportedCorrection):
        with case.db.transaction():
            case.db.conn.execute('INSERT INTO tomte_user_registry (tenant_id,user_id) VALUES (%s,%s)', (case.tenant,case.user))
            workflow._plan_fact_mutations(operations, data, 'rejected-outer-transaction')
    assert case.observer.execute('SELECT count(*) FROM tomte_user_registry WHERE tenant_id=%s AND user_id=%s', (case.tenant,case.user)).fetchone()[0] == 0
