"""Prescribed semantic responses exercise persistence mechanics, not LLM accuracy."""
from contextlib import contextmanager
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from memory_fixtures import field_entries, simple_entity, simple_fact, prepare_write
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


class MemoryDB:
    def __init__(self):
        self.rows = {}
        self.batches = set()
        self.fail_commit = False

    @contextmanager
    def transaction(self):
        snapshot = deepcopy((self.rows, self.batches))
        try:
            yield
        except Exception:
            self.rows, self.batches = snapshot
            raise

    def is_consolidation_processed(self, tenant, user, batch):
        return (tenant, user, batch) in self.batches

    def claim_consolidation_batch(self, tenant, user, batch):
        if self.is_consolidation_processed(tenant, user, batch):
            return False
        self.batches.add((tenant, user, batch))
        return True

    def owned(self, tenant, user):
        return [deepcopy(r) for r in self.rows.values() if (r['tenant_id'], r['user_id']) == (tenant, user)]

    def get_resolution_candidates(self, tenant, user, names, domain, vector, context, limit=5):
        return [r for r in self.owned(tenant, user) if r['entity_domain'] != 'unresolved_memories'][:limit]

    def get_unresolved_memories(self, tenant, user, vector, limit=5, recent_limit=0):
        return [r for r in self.owned(tenant, user) if r['entity_domain'] == 'unresolved_memories'
                and r['payload']['_unresolved']['status'] == 'pending'][:limit + recent_limit]

    def commit_graph_updates(self, operations):
        for op in operations:
            key = op['entity_id']
            if op.get('_expected_payload') is not None:
                assert self.rows[key]['payload'] == op['_expected_payload']
            self.rows[key] = {'id': key, 'tenant_id': op['tenant_id'], 'user_id': op['user_id'],
                **{k: op[k] for k in ('entity_id', 'entity_name', 'entity_type', 'entity_domain')},
                'payload': deepcopy(op['payload'])}
        if self.fail_commit:
            raise RuntimeError('Synthetic commit failure')
        return True


def message(text, source='u1', timestamp=1793664000, user='user', role='user'):
    return dict(item_id=source, timestamp=timestamp, role=role, content=text,
                tenant_id='tenant', user_id=user, session_id='session', vector=[1.0, 0.0])


def extraction(text='The user reported that their car was painted red.', source='u1'):
    return {'assets_and_possessions': [dict(entity_id='tmp_car', entity_name='My car',
        entity_type='car', entity_domain='assets_and_possessions', facts=[dict(
            attribute_name='repainting', value=field_entries(text), source_chat_id=source)])]}


def pending_payload():
    from tomte.engine.core.memory import Entity
    payload = simple_entity('My car', kind='car', domain='assets_and_possessions',
                            facts=[simple_fact('repainting', 'My car was painted red.')])
    payload.update(entity_domain='unresolved_memories', unresolved_detail='Which car is not established.',
                   _unresolved={'original_domain': 'assets_and_possessions', 'reported_at': '2026-11-03T00:00:00Z'})
    return Entity.model_validate(payload).model_dump(mode='json')


@pytest.fixture
def workflow(monkeypatch):
    import tomte.engine.workflows.consolidation as module
    monkeypatch.setattr(module, 'align_terms', lambda entities, *args:
                        SimpleNamespace(entities=entities, evidence=[[] for _ in entities], metrics={}))
    db = MemoryDB()
    calls = []
    def model(system_prompt, user_text, schema):
        data = json.loads(user_text)
        calls.append((schema.__name__, data))
        if schema.__name__ == 'ExtractionMemorySchema':
            return schema.model_validate(extraction())
        if schema.__name__ == 'ResolutionResponse':
            return schema.model_validate({'resolutions': [dict(extracted_entity_id='tmp_car',
                kind='UNRESOLVED', reasoning='Which car was painted is not established.')]})
        if schema.__name__ == 'ContextualActionResponse':
            return schema.model_validate({'actions': []})
        raise AssertionError(schema.__name__)
    llm = SimpleNamespace(parse_structured=model, get_vector_dimension=lambda: 2,
                          embed_batch=lambda texts: [[1.0, 0.0] for _ in texts])
    result = ConsolidationWorkflow(db=db, llm=llm)
    result.dict_service = SimpleNamespace(apply_feedback=lambda operations: None)
    result.test_calls = calls
    return result


def test_unresolved_statement_is_retained_without_inventing_a_car(workflow):
    chats = [message('My car was painted red.')]
    assert workflow.run_consolidation_cycle('tenant', 'user', chats)
    rows = workflow.db.owned('tenant', 'user')
    assert len(rows) == 1
    payload = rows[0]['payload']
    assert payload['entity_domain'] == 'unresolved_memories'
    assert payload['_unresolved']['status'] == 'pending'
    assert payload['_unresolved']['original_domain'] == 'assets_and_possessions'
    assert payload['_unresolved']['reported_at'] == '2026-11-03T00:00:00Z'
    assert payload['facts'][0]['value']['detail']['text'] == 'The user reported that their car was painted red.'
    assert payload['unresolved_detail'] == 'Which car was painted is not established.'
    assert 'tmp_car' not in json.dumps(payload)
    assert 'source_chat_id' not in json.dumps(payload)
    before = deepcopy(workflow.db.rows)
    count = len(workflow.test_calls)
    assert workflow.run_consolidation_cycle('tenant', 'user', chats)
    assert workflow.db.rows == before and len(workflow.test_calls) == count


def seed_pending(workflow, *, newer_colour=False, new_target=False):
    assert workflow.run_consolidation_cycle('tenant', 'user', [message('My car was painted red.')])
    pending_id = next(iter(workflow.db.rows))
    tesla = simple_entity('Tesla', kind='car', domain='assets_and_possessions', facts=[
        simple_fact('colour' if newer_colour else 'model', 'Repainted blue on 4 November 2026.' if newer_colour else 'Tesla Model Y')])
    if not new_target:
        workflow.db.commit_graph_updates([prepare_write('tenant', 'user', tesla)])
    return pending_id, tesla


def clarify(workflow, pending_id, tesla, *, binding=True, mutate=True, newer_colour=False,
            new_target=False, bad_source=False, bad_target=False, missing_key=False):
    from uuid import uuid4
    quote = 'The car I painted is my Tesla.'
    def model(system_prompt, user_text, schema):
        data = json.loads(user_text)
        workflow.test_calls.append((schema.__name__, data))
        if schema.__name__ == 'ExtractionMemorySchema':
            if not new_target:
                return schema.model_validate({})
            entity = extraction(source='u2')['assets_and_possessions'][0]
            entity.update(entity_name='Tesla')
            return schema.model_validate({'assets_and_possessions': [entity]})
        if schema.__name__ == 'ResolutionResponse':
            return schema.model_validate({'resolutions': [dict(extracted_entity_id='tmp_car', kind='NEW',
                reasoning='Introduces the previously unrecorded Tesla.', explicitly_new=True, new_entity_evidence=quote)]})
        if schema.__name__ == 'UnresolvedResolutionResponse':
            pending = data['PENDING'][0]
            assert pending['earlier_memory']['reported_at'] == '2026-11-03T00:00:00Z'
            target_id = pending['TARGETS'][0]['entity_id']
            return schema.model_validate({'bindings': [dict(unresolved_id=pending_id,
                target=dict(db_id=str(uuid4()) if bad_target else target_id, resolution_confidence=0.99, evidence='EXPLICIT'),
                source_chat_id='assistant' if bad_source else 'u2', evidence_quote=quote)] if binding else []})
        if schema.__name__ == 'ContextualActionResponse':
            return schema.model_validate({'actions': []})
        if schema.__name__ == 'MutationPlanResponse':
            assert all(item['item_id'] != 'u1' for item in data['CURRENT_BATCH'])
            target = data['TARGETS'][0]
            proposal = next(f for f in target['incoming_semantics'] if f.get('stored_unresolved_id'))
            assert proposal['reported_at'] == '2026-11-03T00:00:00Z'
            assert proposal['source_chat_id'] == 'u2'
            operation = dict(operation='APPEND_HISTORY' if newer_colour else 'ADD',
                attribute_name='colour' if newer_colour else 'repainting',
                fields=field_entries('Painted red, reported on 3 November 2026.'), source_chat_ids=['u2'],
                incoming_fact_key=None if missing_key else proposal['incoming_fact_key'])
            if newer_colour:
                operation['target_fact_id'] = tesla['facts'][0]['fact_id']
            return schema.model_validate({'plans': [dict(target_key=target['target_key'], operations=[operation])] if mutate else []})
        raise AssertionError(schema.__name__)
    workflow.llm.parse_structured = model
    return [message(quote, source='u2', timestamp=1793836800)]


def test_later_clarification_resolves_old_claim_even_when_extraction_is_empty(workflow):
    identity, tesla = seed_pending(workflow)
    chats = clarify(workflow, identity, tesla)
    assert workflow.run_consolidation_cycle('tenant', 'user', chats)
    resolved = workflow.db.rows[identity]['payload']['_unresolved']
    assert resolved['status'] == 'resolved'
    assert resolved['resolved_entity_ids'] == [tesla['entity_id']]
    assert len(workflow.db.rows[tesla['entity_id']]['payload']['facts']) == 2
    assert workflow.db.get_unresolved_memories('tenant', 'user', [1., 0.]) == []
    before = deepcopy(workflow.db.rows)
    count = len(workflow.test_calls)
    assert workflow.run_consolidation_cycle('tenant', 'user', chats)
    assert before == workflow.db.rows and count == len(workflow.test_calls)


@pytest.mark.parametrize('options', [dict(binding=False), dict(mutate=False), dict(bad_source=True),
                                   dict(bad_target=True)])
def test_unaccepted_clarification_keeps_pending_claim_and_target_unchanged(workflow, options):
    identity, tesla = seed_pending(workflow)
    chats = clarify(workflow, identity, tesla, **options)
    before = deepcopy(workflow.db.rows)
    assert workflow.run_consolidation_cycle('tenant', 'user', chats)
    assert workflow.db.rows == before


def test_missing_proposal_key_keeps_claim_pending_without_blocking_other_valid_writes(workflow):
    identity, tesla = seed_pending(workflow)
    chats = clarify(workflow, identity, tesla, missing_key=True)
    before = deepcopy(workflow.db.rows[identity])
    assert workflow.run_consolidation_cycle('tenant', 'user', chats)
    assert workflow.db.rows[identity] == before
    assert len(workflow.db.rows[tesla['entity_id']]['payload']['facts']) == 2


def test_target_and_completion_marker_roll_back_together(workflow):
    identity, tesla = seed_pending(workflow)
    chats = clarify(workflow, identity, tesla)
    before = deepcopy((workflow.db.rows, workflow.db.batches))
    workflow.db.fail_commit = True
    with pytest.raises(RuntimeError, match='Synthetic commit failure'):
        workflow.run_consolidation_cycle('tenant', 'user', chats)
    assert before == (workflow.db.rows, workflow.db.batches)
    workflow.db.fail_commit = False
    assert workflow.run_consolidation_cycle('tenant', 'user', chats)
    assert workflow.db.rows[identity]['payload']['_unresolved']['status'] == 'resolved'


def test_older_repaint_can_be_added_to_history_without_overwriting_newer_colour(workflow):
    identity, tesla = seed_pending(workflow, newer_colour=True)
    chats = clarify(workflow, identity, tesla, newer_colour=True)
    assert workflow.run_consolidation_cycle('tenant', 'user', chats)
    fact = workflow.db.rows[tesla['entity_id']]['payload']['facts'][0]
    assert fact['value'] == tesla['facts'][0]['value']
    assert fact['history_values'][0]['value']['detail']['text'] == 'Painted red, reported on 3 November 2026.'
    assert workflow.db.rows[identity]['payload']['_unresolved']['status'] == 'resolved'


def test_same_batch_new_entity_can_receive_earlier_claim(workflow):
    identity, tesla = seed_pending(workflow, new_target=True)
    chats = clarify(workflow, identity, tesla, new_target=True)
    assert workflow.run_consolidation_cycle('tenant', 'user', chats)
    pending = workflow.db.rows[identity]['payload']
    target = workflow.db.rows[pending['_unresolved']['resolved_entity_ids'][0]]['payload']
    assert target['entity_name'] == 'Tesla' and target['facts']


@pytest.mark.parametrize('fault', [None, 'bad_source', 'missing_key', 'empty_fields'])
def test_combined_clarification_write_resolves_only_after_accepted_keyed_operation(workflow, fault):
    """Prescribed model plan checks mechanics, not whether a live model deduplicates."""
    identity, tesla = seed_pending(workflow, new_target=True)
    chats = clarify(workflow, identity, tesla, new_target=True)
    chats[0]['content'] += ' I own it.'
    original_model = workflow.llm.parse_structured
    original_pending = deepcopy(workflow.db.rows[identity]['payload'])

    def model(system_prompt, user_text, schema):
        if schema.__name__ == 'ExtractionMemorySchema':
            entity = extraction(source='u2')['assets_and_possessions'][0]
            entity['entity_name'] = 'Tesla'
            entity['facts'] = [
                dict(attribute_name='ownership', value=field_entries('The user owns a Tesla.'), source_chat_id='u2'),
                dict(attribute_name='paint_colour', value=field_entries('The Tesla is the car previously painted red.'), source_chat_id='u2')]
            return schema.model_validate({'assets_and_possessions': [entity]})
        if schema.__name__ == 'MutationPlanResponse':
            target = json.loads(user_text)['TARGETS'][0]
            assert not target['facts']  # No fact UUID exists yet for a second MENTION.
            proposals = target['incoming_semantics']
            retained = next(f for f in proposals if f.get('stored_unresolved_id'))
            ownership = next(f for f in proposals if f['attribute_name'] == 'ownership')
            assert any(f['attribute_name'] == 'paint_colour' for f in proposals)
            combined = dict(operation='ADD', attribute_name='repainting',
                fields=[] if fault == 'empty_fields' else field_entries('The Tesla was painted red, reported on 3 November 2026.'),
                source_chat_ids=['assistant'] if fault == 'bad_source' else ['u2'],
                incoming_fact_key=None if fault == 'missing_key' else retained['incoming_fact_key'])
            independent = dict(operation='ADD', attribute_name='ownership', fields=field_entries('The user owns a Tesla.'),
                source_chat_ids=['u2'], incoming_fact_key=ownership['incoming_fact_key'])
            return schema.model_validate({'plans': [dict(target_key=target['target_key'], operations=[combined, independent])]})
        return original_model(system_prompt, user_text, schema)

    workflow.llm.parse_structured = model
    assert workflow.run_consolidation_cycle('tenant', 'user', chats)
    pending = workflow.db.rows[identity]['payload']
    target = next(r['payload'] for key, r in workflow.db.rows.items() if key != identity)
    assert sum(f['attribute_name'] == 'ownership' for f in target['facts']) == 1
    assert not any(f['attribute_name'] == 'paint_colour' for f in target['facts'])
    written = fault in (None, 'missing_key')
    assert len(target['facts']) == (2 if written else 1)
    if fault is None:
        assert pending['_unresolved']['status'] == 'resolved'
        assert pending['_unresolved']['resolved_entity_ids'] == [target['entity_id']]
        assert pending['facts'] == original_pending['facts']
    else:
        assert pending == original_pending
    before = deepcopy(workflow.db.rows)
    assert workflow.run_consolidation_cycle('tenant', 'user', chats)
    assert workflow.db.rows == before


def test_assistant_only_batch_cannot_resolve_pending_claim(workflow):
    seed_pending(workflow)
    before = deepcopy(workflow.db.rows)
    count = len(workflow.test_calls)
    assert workflow.run_consolidation_cycle('tenant', 'user', [message('The car is your Tesla.', source='a2', role='assistant')])
    assert before == workflow.db.rows and count == len(workflow.test_calls)


def test_unusable_identity_response_is_not_invented_as_pending_memory(workflow):
    original = workflow.llm.parse_structured
    def model(system_prompt, user_text, schema):
        if schema.__name__ == 'ResolutionResponse':
            return schema.model_validate({'resolutions': []})
        return original(system_prompt, user_text, schema)
    workflow.llm.parse_structured = model
    assert workflow.run_consolidation_cycle('tenant', 'user', [message('My car was painted red.')])
    assert not workflow.db.rows


def test_empty_resolver_explanation_does_not_drop_supported_unresolved_claim(workflow):
    original = workflow.llm.parse_structured
    def model(system_prompt, user_text, schema):
        result = original(system_prompt, user_text, schema)
        if schema.__name__ == 'ResolutionResponse':
            result.resolutions[0].reasoning = ''
        return result
    workflow.llm.parse_structured = model
    assert workflow.run_consolidation_cycle('tenant', 'user', [message('My car was painted red.')])
    payload = next(iter(workflow.db.rows.values()))['payload']
    assert payload['unresolved_detail'] == 'unresolved_subject_identity'
    assert payload['facts'][0]['value']['detail']['text'] == 'The user reported that their car was painted red.'


def test_multiple_retained_facts_resolve_independently(workflow):
    from tomte.engine.core.unresolved import retained_operations
    from tomte.engine.core.extraction import ExtractionMemorySchema, resolve_extraction
    chats = [message('My car was painted red and I also replaced its tyres.')]
    extracted = extraction()
    extracted['assets_and_possessions'][0]['facts'].append(dict(attribute_name='tyres',
        value=field_entries('Tyres replaced.'), source_chat_id='u1'))
    resolved = resolve_extraction(ExtractionMemorySchema.model_validate(extracted), chats)
    entity = resolved.model_dump(mode='json')['assets_and_possessions'][0]
    writes = retained_operations([dict(entity=entity, reason='Which car is unclear.')], chats, 'tenant', 'user', 'seed')
    assert len(writes) == 2 and writes[0]['entity_id'] != writes[1]['entity_id']
    workflow.db.commit_graph_updates(writes)
    tesla = simple_entity('Tesla', kind='car', domain='assets_and_possessions', facts=[simple_fact('model', 'Tesla')])
    workflow.db.commit_graph_updates([prepare_write('tenant', 'user', tesla)])
    clarification = clarify(workflow, writes[0]['entity_id'], tesla)
    assert workflow.run_consolidation_cycle('tenant', 'user', clarification)
    assert workflow.db.rows[writes[0]['entity_id']]['payload']['_unresolved']['status'] == 'resolved'
    assert workflow.db.rows[writes[1]['entity_id']]['payload']['_unresolved']['status'] == 'pending'


def test_resolution_capacity_defers_old_evidence_without_blocking_current_batch(workflow, monkeypatch):
    from tomte.engine.core.config import settings
    identity, tesla = seed_pending(workflow)
    chats = clarify(workflow, identity, tesla)
    # The current batch fits; the resolver's full old-evidence envelope does not.
    monkeypatch.setattr(settings, 'max_model_input_bytes', 1000)
    before = deepcopy(workflow.db.rows)
    assert workflow.run_consolidation_cycle('tenant', 'user', chats)
    assert workflow.db.rows == before


def test_mutation_capacity_defers_pending_claim_and_preserves_normal_proposal(workflow, monkeypatch):
    from tomte.engine.core import capacity
    identity, tesla = seed_pending(workflow, new_target=True)
    chats = clarify(workflow, identity, tesla, new_target=True)
    original_check = capacity.check_provider_capacity
    def check(operation, arguments):
        if arguments.get('schema').__name__ == 'MutationPlanResponse' and 'stored_unresolved_id' in arguments['user_text']:
            raise capacity.CapacityExceeded('synthetic optional history overflow')
        original_check(operation, arguments)
    monkeypatch.setattr(capacity, 'check_provider_capacity', check)
    original_model = workflow.llm.parse_structured
    def model(system_prompt, user_text, schema):
        if schema.__name__ != 'MutationPlanResponse':
            return original_model(system_prompt, user_text, schema)
        target = json.loads(user_text)['TARGETS'][0]
        assert all('stored_unresolved_id' not in f for f in target['incoming_semantics'])
        return schema.model_validate({'plans': [dict(target_key=target['target_key'], operations=[dict(
            operation='ADD', attribute_name='repainting', fields=field_entries('Painted red.'),
            source_chat_ids=['u2'], incoming_fact_key=target['incoming_semantics'][0]['incoming_fact_key'])])]})
    workflow.llm.parse_structured = model
    assert workflow.run_consolidation_cycle('tenant', 'user', chats)
    assert len(workflow.db.rows) == 2
    assert workflow.db.rows[identity]['payload']['_unresolved']['status'] == 'pending'
