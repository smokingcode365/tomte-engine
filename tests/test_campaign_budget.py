from copy import deepcopy
from decimal import Decimal
import importlib
import json
from pathlib import Path
import sys

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
module = importlib.import_module('campaign_budget')
runner = importlib.import_module('run_memory_campaign')
CampaignBudget, BudgetStop = module.CampaignBudget, module.BudgetStop


def approval():
    return {'approved': True, 'estimate_usd': '32', 'ceiling_usd': '48', 'request_limit': 2,
            'active_seconds_limit': 21600, 'pins': {'synthetic': 'fixed'},
            'reviews': {'smoke': {'accepted': False, 'evidence': ''}, 'development': {'accepted': False, 'evidence': ''}}}


def reserve(budget):
    return budget.reserve('https://api.openai.com/v1/embeddings',
                          {'model': 'text-embedding-3-small', 'input': 'synthetic'})


def reconcile(budget, ticket):
    budget.reconcile(ticket, httpx.Response(200, json={'usage': {'prompt_tokens': 100}}))


def test_budget_needs_explicit_larger_approval_and_matching_fingerprints(tmp_path):
    with pytest.raises(ValueError, match='approved_ceiling'):
        module.ApiBudget(estimate='32', ceiling='48')
    a = approval()
    for changed in ({**a, 'approved': False}, {**a, 'pins': {'wrong': 'pin'}}):
        with pytest.raises(ValueError):
            CampaignBudget(tmp_path, changed, a['pins'])
    assert list(tmp_path.iterdir()) == []


def test_reservation_is_persisted_before_dispatch_and_unknown_usage_prevents_resume(tmp_path):
    a = approval()
    with CampaignBudget(tmp_path, a, a['pins']) as budget:
        _, ticket = reserve(budget)
        state = json.loads((tmp_path / 'budget.json').read_text())
        assert state['records'][0]['state'] == 'reserved'
        assert Decimal(state['conservative_charged_usd']) == Decimal(ticket['reservation_usd'])
    with pytest.raises(BudgetStop, match='unresolved billing'):
        CampaignBudget(tmp_path, a, a['pins'])
    assert not (tmp_path / 'campaign.lock').exists()


def test_resume_keeps_prior_cost_requests_and_run_identity(tmp_path):
    a = approval()
    with CampaignBudget(tmp_path, a, a['pins']) as budget:
        budget.begin_run('case-1')
        _, first = reserve(budget)
        reconcile(budget, first)
        charged = budget.charged
        budget.end_run('case-1', True)
        budget.phases.append('smoke')
    with CampaignBudget(tmp_path, a, a['pins']) as resumed:
        assert resumed.charged == charged and len(resumed.records) == 1
        assert resumed.phases == ['smoke']
        with pytest.raises(ValueError, match='already attempted'):
            resumed.begin_run('case-1')
        _, second = reserve(resumed)
        reconcile(resumed, second)
        assert resumed.charged == charged * 2
        with pytest.raises(BudgetStop, match='request_limit'):
            reserve(resumed)


def test_ledger_does_not_allow_parallel_writer_or_changed_limits(tmp_path):
    a = approval()
    with CampaignBudget(tmp_path, a, a['pins']):
        with pytest.raises(FileExistsError):
            CampaignBudget(tmp_path, a, a['pins'])
    with pytest.raises(ValueError, match='different bounds'):
        CampaignBudget(tmp_path, {**a, 'ceiling_usd': '49'}, a['pins'])


def test_ledger_failure_stops_before_network_dispatch(tmp_path, monkeypatch):
    a = approval()
    budget = CampaignBudget(tmp_path, a, a['pins'])
    original = budget.checkpoint
    def broken():
        raise OSError('synthetic disk failure')
    monkeypatch.setattr(budget, 'checkpoint', broken)
    with pytest.raises(OSError):
        reserve(budget)
    assert budget.stopped == 'reservation_or_ledger_failure'
    monkeypatch.setattr(budget, 'checkpoint', original)
    budget.close()
    with pytest.raises(BudgetStop):
        CampaignBudget(tmp_path, a, a['pins'])


def test_failed_case_and_corrupt_totals_cannot_reset_budget(tmp_path):
    a = approval()
    with CampaignBudget(tmp_path, a, a['pins']) as budget:
        budget.begin_run('case-1')
        budget.end_run('case-1', False)
    with pytest.raises(BudgetStop):
        CampaignBudget(tmp_path, a, a['pins'])
    state_path = tmp_path / 'budget.json'
    state = json.loads(state_path.read_text())
    state.update(stopped=None, conservative_charged_usd='-1')
    state_path.write_text(json.dumps(state))
    with pytest.raises(ValueError, match='inconsistent'):
        CampaignBudget(tmp_path, a, a['pins'])


def test_validation_requires_completed_development_and_saved_semantic_review(tmp_path):
    a = approval()
    with CampaignBudget(tmp_path, a, a['pins']) as budget:
        with pytest.raises(ValueError, match='semantic acceptance'):
            runner.check_phase(a, budget, 'validation', tmp_path)
        budget.phases.extend(['smoke', 'development'])
        review = tmp_path / 'development-review.json'
        review.write_text('{"accepted": true}')
        a['reviews']['development'] = {'accepted': True, 'evidence': review.name}
        runner.check_phase(a, budget, 'validation', tmp_path)
        with pytest.raises(ValueError, match='already completed'):
            runner.check_phase(a, budget, 'smoke', tmp_path)


def test_unapproved_live_phase_never_loads_environment(tmp_path, monkeypatch):
    a = approval()
    a['approved'] = False
    (tmp_path / 'approval.json').write_text(json.dumps(a))
    monkeypatch.setattr(runner, 'pins', lambda: a['pins'])
    def forbidden(*args, **kwargs):
        pytest.fail('Unapproved execution reached environment/service setup')
    monkeypatch.setattr(runner.pilot, 'load_dotenv', forbidden)
    with pytest.raises(ValueError, match='not approved'):
        runner.run_phase(tmp_path, 'smoke', {}, {})


def test_pre_dispatch_phase_error_does_not_poison_an_unused_budget(tmp_path):
    a = approval()
    with pytest.raises(ValueError, match='semantic acceptance'):
        with CampaignBudget(tmp_path, a, a['pins']) as budget:
            runner.check_phase(a, budget, 'validation', tmp_path)
    with CampaignBudget(tmp_path, a, a['pins']) as resumed:
        assert not resumed.stopped and not resumed.records and resumed.charged == 0


@pytest.mark.parametrize('complete,expected_requests', [(True,2),(False,1)])
def test_phase_executor_persists_cost_and_stops_on_incomplete_case(tmp_path, monkeypatch, complete, expected_requests):
    from contextlib import contextmanager, nullcontext
    # This frozen campaign intentionally reproduces its original small-model run.
    # Do not inherit the engine's current embedding default into a budget fixture.
    monkeypatch.setattr(runner.pilot.settings.openai, 'embed_model', 'openai:text-embedding-3-small')
    a = approval()
    (tmp_path / 'approval.json').write_text(json.dumps(a))
    data, gold = runner.pilot.load_cases(long=True)
    monkeypatch.setattr(runner, 'pins', lambda: a['pins'])
    monkeypatch.setattr(runner.pilot, 'load_dotenv', lambda *args, **kwargs: None)
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic')
    monkeypatch.setenv('POSTGRES_URL', 'synthetic')
    monkeypatch.setenv('REDIS_URL', 'synthetic')
    monkeypatch.setattr(runner.pilot, 'OpenAILLMProvider', lambda key: None)
    monkeypatch.setattr(runner, 'bounded_http', lambda *args: nullcontext())

    @contextmanager
    def isolated(postgres, redis, cleanup):
        cleanup['verified'] = True
        yield None, None, 'synthetic'

    def execute(case, repetition, db, cache, tenant, llm, budget, stamp, progress):
        _, ticket = reserve(budget)
        reconcile(budget, ticket)
        result = {'case_id':case['case_id'], 'completed':complete}
        progress(result)
        return result

    monkeypatch.setattr(runner.pilot, 'isolated_store', isolated)
    monkeypatch.setattr(runner.pilot, 'execute_case', execute)
    assert runner.run_phase(tmp_path, 'smoke', data, gold) is complete
    state = json.loads((tmp_path / 'budget.json').read_text())
    result = json.loads((tmp_path / 'smoke-results.json').read_text())
    assert len(state['records']) == expected_requests
    assert result['completed_runs'] == (2 if complete else 0)
    assert bool(state['stopped']) is not complete
    assert state['phases'] == (['smoke'] if complete else [])
    assert json.loads((tmp_path / 'approval.json').read_text())['reviews']['smoke']['accepted'] is False
