"""Prepare or run one approved phase, sharing the entire campaign's budget ledger."""
import argparse
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import subprocess

import evaluate_memory_quality as pilot
from campaign_budget import CampaignBudget
from eval_api_budget import bounded_http
from long_memory_cases import build_long_plan, dataset_statistics


def pins():
    inputs, gold = pilot.case_paths(long=True)
    return {'dataset_sha256': pilot.sha(inputs), 'expectations_sha256': pilot.sha(gold),
            'source_hashes': pilot.source_hashes(), 'models': pilot.settings.openai.model_dump()}


def proposal():
    return {'approved': False, 'estimate_usd': '32', 'ceiling_usd': '48',
            'request_limit': 5000, 'active_seconds_limit': 21600, 'pins': pins(),
            'reviews': {'smoke': {'accepted': False, 'evidence': ''},
                        'development': {'accepted': False, 'evidence': ''}}}


def check_phase(approval, budget, phase, directory):
    if phase in budget.phases:
        raise ValueError('Phase already completed; no implicit repeat')
    previous = {'development': 'smoke', 'validation': 'development'}.get(phase)
    if previous:
        review = approval['reviews'][previous]
        evidence = (directory / review['evidence']).resolve()
        if (previous not in budget.phases or review.get('accepted') is not True
                or not evidence.is_relative_to(directory) or not evidence.is_file()):
            raise ValueError('Previous phase needs completed runs and saved semantic acceptance evidence')


def run_phase(directory, phase, data, gold):
    approval = json.loads((directory / 'approval.json').read_text(encoding='utf-8'))
    current = pins()
    # No environment/service/model setup until approved bounds and pins validate.
    with CampaignBudget(directory, approval, current) as budget:
        check_phase(approval, budget, phase, directory)
        pilot.load_dotenv(pilot.ROOT / '.env', override=False)
        for name in ('tomte.engine', 'httpx', 'httpcore', 'openai'):
            logging.getLogger(name).setLevel(logging.CRITICAL)
        assert pilot.settings.openai.strong_model == 'gpt-6.1-sol'
        assert pilot.settings.openai.cheap_model == 'gpt-6-luna'
        assert pilot.settings.openai.embed_model == 'openai:text-embedding-3-small'
        assert pilot.settings.openai.vector_dimension == 1536
        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError('API key missing')
        postgres_url, redis_url = os.environ['POSTGRES_URL'], os.environ['REDIS_URL']
        plan = build_long_plan(data, phase)
        calls, results = [], []
        live = pilot.OpenAILLMProvider(os.environ['OPENAI_API_KEY'])
        stamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')).timestamp()

        def record_http(entry):
            with (directory / 'http_evidence.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(entry, ensure_ascii=False) + '\n')

        try:
            with bounded_http(budget, record_http):
                for case, repetition in plan:
                    if pins() != current:
                        raise ValueError('Frozen campaign changed')
                    label = case['case_id'] + '-' + str(repetition)
                    budget.begin_run(label)
                    cleanup, case_calls = {}, []
                    calls.append({'run': label, 'calls': case_calls})
                    try:
                        with pilot.isolated_store(postgres_url, redis_url, cleanup) as (db, cache, tenant):
                            llm = pilot.RecordedLLM(live, case_calls)
                            result = pilot.execute_case(case, repetition, db, cache, tenant, llm, budget, stamp,
                                lambda row: pilot.save(directory / (label + '.json'), row))
                        result['cleanup_verified'] = cleanup['verified']
                        pilot.save(directory / (label + '.json'), result)
                        results.append(result)
                        budget.end_run(label, result['completed'] and result['cleanup_verified'])
                    finally:
                        pilot.save(directory / (label + '-cleanup.json'), cleanup)
                        pilot.save(directory / (phase + '-model_calls.json'), calls)
                    if budget.stopped:
                        break
                if len(results) == len(plan) and all(r['completed'] and r['cleanup_verified'] for r in results):
                    budget.phases.append(phase)  # Mechanical completion only; never semantic acceptance.
                    budget.checkpoint()
        finally:
            pilot.save(directory / (phase + '-model_calls.json'), calls)
            pilot.save(directory / (phase + '-results.json'), {'planned_runs': len(plan),
                'completed_runs': sum(r['completed'] for r in results), 'runs': results,
                'finished_utc': datetime.now(timezone.utc).isoformat(), 'source_unchanged': pins() == current})
        print(json.dumps({'phase': phase, 'completed_runs': sum(r['completed'] for r in results),
                          'budget': {k:v for k,v in budget.summary().items() if k != 'records'}}))
        return phase in budget.phases and not budget.stopped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--campaign-dir', type=Path, required=True)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--prepare', action='store_true', help='Offline only; writes an unapproved proposal.')
    modes.add_argument('--phase', choices=('smoke', 'development', 'validation'))
    args = parser.parse_args()
    directory = args.campaign_dir.resolve()
    allowed = (pilot.ROOT / 'log/memory_quality_campaign').resolve()
    if directory == allowed or not directory.is_relative_to(allowed):
        parser.error('Use a new named campaign directory under log/memory_quality_campaign')
    data, gold = pilot.load_cases(long=True)
    if args.prepare:
        directory.mkdir(parents=True, exist_ok=False)
        pilot.save(directory / 'approval.json', proposal())
        pilot.save(directory / 'conversations.json', data)
        pilot.save(directory / 'expectations.json', gold)
        pilot.save(directory / 'preparation.json', {'git_head': subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=pilot.ROOT, text=True).strip(),
            'statistics': dataset_statistics(data), 'prepared_utc': datetime.now(timezone.utc).isoformat(),
            'plan': [{'case_id': c['case_id'], 'repetition': r} for c,r in build_long_plan(data)],
            'package_versions': {p:pilot.version(p) for p in ('pydantic','pydantic-ai','openai','httpx','psycopg','redis')},
            'settings': pilot.settings.model_dump()})
        print(json.dumps({'prepared': str(directory), 'approved': False, 'estimate_usd': '32', 'ceiling_usd': '48'}))
    else:
        if not run_phase(directory, args.phase, data, gold):
            raise SystemExit(1)


if __name__ == '__main__':
    main()
