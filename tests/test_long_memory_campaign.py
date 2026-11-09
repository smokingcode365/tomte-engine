"""Offline harness checks; synthetic decisions do not measure memory quality."""
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
pilot = importlib.import_module('evaluate_memory_quality')
long_cases = importlib.import_module('long_memory_cases')


def test_frozen_long_dataset_has_complete_split_sessions_and_source_gold():
    data, gold = pilot.load_cases(long=True)
    original = deepcopy(data)
    plan = long_cases.build_long_plan(data)
    assert len(data['cases']) == 6 and len(plan) == 18
    assert sum(len(c['batches']) for c, _ in plan) == 288
    assert sum(len(c['queries']) for c, _ in plan) == 144
    assert sum(s['user_messages'] for s in long_cases.dataset_statistics(data)) == 240
    assert all(s['english_words_mean'] >= 40 for s in long_cases.dataset_statistics(data))
    assert [c['split'] for c, _ in plan] == ['development'] * 9 + ['validation'] * 9
    smoke = long_cases.build_long_plan(data, 'smoke')
    rest = long_cases.build_long_plan(data, 'development')
    held = long_cases.build_long_plan(data, 'validation')
    assert len(smoke) == 2 and all(r == 1 for _, r in smoke)
    assert smoke + rest + held == plan and len({(c['case_id'], r) for c, r in plan}) == 18
    assert data == original
    assert 'required' not in json.dumps(data) and gold['rules']


@pytest.mark.parametrize('damage', ['future_gold', 'assistant_gold', 'repeat_id', 'query_time', 'session_collision', 'non_utc'])
def test_invalid_evidence_or_schedule_is_rejected(damage):
    data, gold = pilot.load_cases(long=True)
    case = data['cases'][0]
    expected = gold['cases'][case['case_id']]
    if damage == 'future_gold':
        expected['checkpoints'][0]['required'][0]['source_turns'] = ['u40']
    elif damage == 'assistant_gold':
        expected['checkpoints'][0]['required'][0]['source_turns'] = ['a01']
    elif damage == 'repeat_id':
        case['batches'][1][0]['message_id'] = 'u01'
    elif damage == 'query_time':
        case['queries'][0]['after_batch'] = 99
    elif damage == 'session_collision':
        case['batch_metadata'][2]['session_id'] = 'session_01'
    else:
        case['batch_metadata'][0]['at'] = '2026-09-01T12:00:00'
    with pytest.raises(AssertionError):
        long_cases.validate_case(case, expected)


def run_fake(monkeypatch, case, *, query_error=False, query_write=False, batch_size=None):
    """Exercise real runner control flow with observable stores and stage doubles."""
    db = SimpleNamespace(memory=[], processed=set())
    cache = SimpleNamespace(queues={})
    log, snapshots = [], []
    counter = [0]

    class Hook:
        def __init__(self, **kwargs):
            pass

        def on_chat(self, tenant, session, text, role, user):
            counter[0] += 1
            entry = {'item_id': str(counter[0]), 'tenant_id': tenant, 'user_id': user,
                     'session_id': session, 'content': text, 'role': role.value,
                     'timestamp': pilot.redis_provider.time.time()}
            cache.queues.setdefault(session, []).append(entry)
            log.append(entry)
            if session.startswith('eval-query-'):
                if query_write:
                    db.memory.append({'unsupported': text})
                if query_error:
                    return {'error': 'synthetic failure'}
            return {'query_relevant_memories': deepcopy(db.memory), 'background_memories': [], 'chat_history': []}

    class Workflow:
        def __init__(self, **kwargs):
            pass

        def run_consolidation_cycle(self, tenant, user, items):
            batch_key = tuple(i['item_id'] for i in items)
            assert not any(i['session_id'].startswith('eval-query-') for i in items)
            if batch_key not in db.processed:
                db.memory.extend(deepcopy([i for i in items if i['role'] == 'user']))
                db.processed.add(batch_key)
            return True

    cache.pop_session_chat_log_4_consolidation = lambda tenant, user, session: deepcopy(cache.queues.get(session, [])[:batch_size])

    def clear(tenant, user, session, items):
        assert cache.queues[session][:len(items)] == items
        cache.queues[session] = cache.queues[session][len(items):]
        return True

    cache.clear_session_4_consolidation = clear
    monkeypatch.setattr(pilot, 'ChatHookWorkflow', Hook)
    monkeypatch.setattr(pilot, 'ConsolidationWorkflow', Workflow)
    monkeypatch.setattr(pilot, 'stored', lambda db, tenant, user: deepcopy(db.memory))
    old_clock = pilot.redis_provider.time
    result = pilot.execute_case(case, 1, db, cache, 'test-scope', None,
                                SimpleNamespace(records=[]), 1000,
                                lambda value: snapshots.append(deepcopy(value)))
    assert pilot.redis_provider.time is old_clock
    return result, log, db, snapshots


def test_intermediate_queries_observe_only_past_batches_and_never_join_ingestion(monkeypatch):
    case = pilot.load_cases(long=True)[0]['cases'][0]
    result, log, db, snapshots = run_fake(monkeypatch, case)
    assert result['completed'] and len(db.memory) == 40 and len(db.processed) == 16
    assert len({b['session_id'] for b in result['batches']}) == 8
    assert all(b['replay_verified'] for b in result['batches'])
    assert [q['after_batch'] for q in result['retrieval']] == [3, 7, 9, 11, 15, 15, 15, 15]
    assert [len(q['payload']['query_relevant_memories']) for q in result['retrieval']] == [10,20,25,30,40,40,40,40]
    queries = [e for e in log if e['session_id'].startswith('eval-query-')]
    assert len(queries) == len({e['session_id'] for e in queries}) == 8
    assert not set(e['item_id'] for e in queries) & set(e['item_id'] for e in db.memory)
    ingestion = [e for e in log if not e['session_id'].startswith('eval-query-')]
    assert ingestion[7]['timestamp'] - ingestion[0]['timestamp'] == 4 * 86400
    assert snapshots[-1]['completed']


def test_checkpoint_waits_for_every_processing_batch(monkeypatch):
    case = pilot.load_cases(long=True)[0]['cases'][0]
    result, _, db, _ = run_fake(monkeypatch, case, batch_size=1)
    assert result['completed'] and len(db.memory) == 40
    assert [len(q['payload']['query_relevant_memories']) for q in result['retrieval']] == [10,20,25,30,40,40,40,40]
    for expected, actual in zip(case['batches'], result['batches']):
        assert len(actual['queued_messages']) == len(expected)
        assert len(actual['processing_batches']) == len(expected)
        assert all(part['success'] and part['replay_verified'] for part in actual['processing_batches'])


@pytest.mark.parametrize('mode', ['query_error', 'query_write'])
def test_retrieval_error_or_memory_write_cannot_report_completed_run(monkeypatch, mode):
    case = pilot.load_cases(long=True)[0]['cases'][0]
    result, _, _, snapshots = run_fake(monkeypatch, case, **{mode: True})
    assert not result['completed'] and result['error_type'] in ('RuntimeError', 'AssertionError')
    assert len(result['batches']) == 4 and not result['retrieval']
    assert snapshots[-1]['error_type'] == result['error_type']


def test_existing_short_pilot_still_gets_final_queries_in_independent_sessions(monkeypatch):
    case = pilot.load_cases(realistic=True)[0]['cases'][0]
    result, log, db, _ = run_fake(monkeypatch, case)
    assert result['completed'] and len(db.memory) == 8
    assert [q['after_batch'] for q in result['retrieval']] == [4,4]
    assert len({e['session_id'] for e in log if e['session_id'].startswith('eval-query-')}) == 2


def test_live_long_campaign_stops_before_reading_credentials_or_connecting(monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['evaluate_memory_quality.py', '--long-pilot', '--live'])
    def forbidden(*args, **kwargs):
        pytest.fail('No environment, database or paid-call setup before authorization')
    monkeypatch.setattr(pilot, 'load_dotenv', forbidden)
    with pytest.raises(SystemExit) as raised:
        pilot.main()
    assert raised.value.code == 2
