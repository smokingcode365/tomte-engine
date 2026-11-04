"""Check evaluation safeguards and frozen data, not Tomte's semantic accuracy."""
from decimal import Decimal
import asyncio
import importlib
import json
from pathlib import Path
import sys

import httpx
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
budget_module = importlib.import_module('eval_api_budget')
pilot = importlib.import_module('evaluate_memory_quality')
ApiBudget, BudgetStop = budget_module.ApiBudget, budget_module.BudgetStop


def request():
    return {'model': 'gpt-6.1-sol', 'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'synthetic text'}]}]}


def response(**kwargs):
    return httpx.Response(200, json={'usage': {'input_tokens': 100, 'output_tokens': 50}, 'service_tier': 'default', **kwargs})


def test_frozen_cases_are_separate_from_gold_and_have_twelve_batches():
    data, gold = pilot.load_cases()
    assert len(data['cases']) == 5
    assert sum(case['repetitions'] for case in data['cases']) == 7
    assert sum(case['repetitions'] * len(case['batches']) for case in data['cases']) == 12
    assert gold['wrong_memory_weight'] > gold['missed_requirement_weight']
    assert all('required' not in case and 'forbidden_examples' not in case for case in data['cases'])


def test_recovery_plan_starts_with_failures_and_repeats_without_editing_frozen_data():
    from copy import deepcopy
    data, _ = pilot.load_cases()
    original = deepcopy(data)
    plan = pilot.build_plan(data, recovery=True)
    assert [(c['case_id'], r) for c, r in plan[:2]] == [('concurrent_jobs', 1), ('two_car_correction', 1)]
    assert len(plan) == 9 and sum(len(case['batches']) for case, _ in plan) == 20
    assert [(c['case_id'], r) for c, r in plan[-4:]] == [('concurrent_jobs', 2), ('two_car_correction', 2), ('concurrent_jobs', 3), ('two_car_correction', 3)]
    assert len(pilot.build_plan(data)) == 7 and data == original


def test_budget_reserves_before_call_and_reconciles_usage_without_discounts():
    budget = ApiBudget()
    body, ticket = budget.reserve('https://api.openai.com/v1/responses', request())
    assert body['max_output_tokens'] == 8192
    assert body['service_tier'] == 'default' and body['store'] is False
    assert budget.charged == Decimal(ticket['reservation_usd']) > 0
    budget.reconcile(ticket, response())
    assert budget.usage_cost == Decimal('.0007')
    assert budget.charged == Decimal('.0007') * Decimal('1.375')
    with pytest.raises(ValueError):
        budget.reconcile(ticket, response())


def test_subject_pilot_uses_separate_frozen_data_and_nine_runs():
    original_data, original_gold = pilot.load_cases()
    data, gold = pilot.load_cases(subject=True)
    plan = pilot.build_plan(data)
    assert [c['case_id'] for c in data['cases']] == ['user_jobs', 'brother_jobs', 'stable_car_name']
    assert len(plan) == 9 and sum(len(case['batches']) for case, _ in plan) == 18
    assert all(sum(c['case_id'] == name for c, _ in plan) == 3 for name in gold['cases'])
    assert pilot.case_paths() != pilot.case_paths(subject=True)
    assert pilot.load_cases() == (original_data, original_gold)
    assert gold['cases']['brother_jobs']['coverage_checks']
    assert all('required' not in case and 'structure' not in case for case in data['cases'])


def test_realistic_pilot_keeps_gold_separate_and_covers_forty_user_turns():
    original = pilot.load_cases()
    subject = pilot.load_cases(subject=True)
    data, gold = pilot.load_cases(realistic=True)
    plan = pilot.build_plan(data)
    assert len(plan) == 5 and sum(len(c['batches']) for c, _ in plan) == 25
    assert all(sum(t['role'] == 'user' for b in c['batches'] for t in b) == 8 for c, _ in plan)
    assert len({pilot.case_paths(), pilot.case_paths(subject=True), pilot.case_paths(realistic=True)}) == 3
    assert pilot.load_cases() == original and pilot.load_cases(subject=True) == subject
    assert all(set(c) == {'case_id', 'repetitions', 'batches', 'queries'} for c in data['cases'])
    assert gold['satisfactory_gate'] and gold['cases']['ambiguous_brothers']['checkpoints']
    assert gold['cases']['correction_then_change']['checkpoints']
    assert gold['cases']['multilingual_uncertainty']['checkpoints']


@pytest.mark.parametrize('url,updates', [
    ('https://unexpected.example/v1/responses', {}),
    ('https://api.openai.com/v1/chat/completions', {}),
    ('https://api.openai.com/v1/responses', {'model': 'unknown'}),
    ('https://api.openai.com/v1/responses', {'stream': True}),
    ('https://api.openai.com/v1/responses', {'previous_response_id': 'unbounded'}),
    ('https://api.openai.com/v1/responses', {'tools': [{'type': 'web_search'}]}),
    ('https://api.openai.com/v1/responses', {'input': [{'content': [{'type': 'input_image'}]}]}),
    ('https://api.openai.com/v1/responses', {'input': 'x' * 96001}),
])
def test_unpriced_or_unbounded_requests_never_get_reserved(url, updates):
    budget = ApiBudget()
    with pytest.raises(BudgetStop):
        budget.reserve(url, {**request(), **updates})
    assert budget.charged == 0 and not budget.records


def test_limit_stops_before_sending_and_cannot_be_resumed_implicitly():
    budget = ApiBudget(ceiling='.01', estimate='.01')
    with pytest.raises(BudgetStop, match='spend_limit_before_dispatch'):
        budget.reserve('https://api.openai.com/v1/responses', request())
    with pytest.raises(BudgetStop):
        budget.reserve('https://api.openai.com/v1/embeddings', {'model': 'text-embedding-3-small', 'input': 'tiny'})
    assert budget.charged == 0


@pytest.mark.parametrize('answer', [
    httpx.Response(429, json={}), response(usage={}), response(service_tier='fast'),
    response(usage={'input_tokens': 1, 'output_tokens': 8193}),
])
def test_provider_failures_retain_reservation_and_stop_future_dispatch(answer):
    budget = ApiBudget()
    _, ticket = budget.reserve('https://api.openai.com/v1/responses', request())
    with pytest.raises(BudgetStop):
        budget.reconcile(ticket, answer)
    assert budget.charged == Decimal(ticket['reservation_usd'])
    with pytest.raises(BudgetStop):
        budget.reserve('https://api.openai.com/v1/responses', request())


def test_request_limit_counts_each_attempt_even_when_previous_call_was_cheap():
    budget = ApiBudget(request_limit=1)
    _, ticket = budget.reserve('https://api.openai.com/v1/embeddings', {'model': 'text-embedding-3-small', 'input': 'hello'})
    budget.reconcile(ticket, httpx.Response(200, json={'usage': {'prompt_tokens': 1}}))
    with pytest.raises(BudgetStop, match='request_limit'):
        budget.reserve('https://api.openai.com/v1/responses', request())


def test_transport_hooks_restore_after_local_failure():
    original_async, original_sync = httpx.AsyncClient.send, httpx.Client.send
    with pytest.raises(RuntimeError):
        with budget_module.bounded_http(ApiBudget(), lambda value: None):
            assert httpx.AsyncClient.send is not original_async
            raise RuntimeError('test')
    assert httpx.AsyncClient.send is original_async and httpx.Client.send is original_sync


def test_transport_enforces_limits_on_actual_rewritten_request(monkeypatch):
    sent, evidence = [], []
    async def fake_send(client, request, *args, **kwargs):
        sent.append(request)
        return response()
    monkeypatch.setattr(httpx.AsyncClient, 'send', fake_send)
    budget = ApiBudget(request_limit=1)
    async def run():
        async with httpx.AsyncClient() as client:
            with budget_module.bounded_http(budget, evidence.append):
                await client.post('https://api.openai.com/v1/responses', json=request(), headers={'authorization': 'synthetic-secret'})
                with pytest.raises(BudgetStop):
                    await client.post('https://api.openai.com/v1/responses', json=request())
    asyncio.run(run())
    assert len(sent) == 1 and len(evidence) == 1
    body = json.loads(sent[0].content)
    assert body['max_output_tokens'] == 8192 and body['service_tier'] == 'default'
    assert int(sent[0].headers['content-length']) == len(sent[0].content)
    assert 'synthetic-secret' not in json.dumps(evidence)


def test_transport_error_prevents_a_second_network_attempt(monkeypatch):
    sent = []
    async def fake_send(client, request, *args, **kwargs):
        sent.append(request)
        raise httpx.ReadTimeout('synthetic timeout')
    monkeypatch.setattr(httpx.AsyncClient, 'send', fake_send)
    budget = ApiBudget()
    async def run():
        async with httpx.AsyncClient() as client:
            with budget_module.bounded_http(budget, lambda value: None):
                with pytest.raises(httpx.ReadTimeout):
                    await client.post('https://api.openai.com/v1/responses', json=request())
                with pytest.raises(BudgetStop):
                    await client.post('https://api.openai.com/v1/responses', json=request())
    asyncio.run(run())
    assert len(sent) == 1 and budget.charged > 0
