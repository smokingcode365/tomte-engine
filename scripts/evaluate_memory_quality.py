"""Small real-model memory-quality pilot with frozen cases and reviewable evidence.

Default mode validates inputs only. --live uses bounded API calls and a new
PostgreSQL schema per run, with scoped Redis keys and cleanup verification.
Semantic judgments are made separately from execution; gold never enters Tomte.
"""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import time
import traceback
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import load_dotenv
import psycopg
from psycopg import sql
from psycopg.conninfo import make_conninfo
import redis

from eval_api_budget import ApiBudget, BudgetStop, bounded_http
from tomte.engine.core.config import settings
from tomte.engine.core.enums import ChatSenderRole
from tomte.engine.core.memory import Entity
from tomte.engine.providers.openai_provider import OpenAILLMProvider
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.providers.redis_provider import RedisCacheProvider
from tomte.engine.providers import redis_provider
from tomte.engine.utils.keys import get_section_key_active_sessions
from tomte.engine.workflows.chat_hook import ChatHookWorkflow
from tomte.engine.workflows.consolidation import ConsolidationWorkflow

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'evals/memory_quality'
TABLES = ('tomte_memory_engine', 'tomte_memory_search', 'tomte_processed_consolidations',
          'tomte_user_registry', 'tomte_semantic_anchors', 'tomte_keyword_registry', 'tomte_keyword_audit_log')


def save(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_paths(subject=False, realistic=False):
    assert not (subject and realistic)
    suffix = 'realistic_v1' if realistic else 'subjects_v1' if subject else 'v1'
    return DATA / f'conversations_{suffix}.json', DATA / f'expectations_{suffix}.json'


def load_cases(subject=False, realistic=False):
    data_path, gold_path = case_paths(subject, realistic)
    data = json.loads(data_path.read_text(encoding='utf-8'))
    gold = json.loads(gold_path.read_text(encoding='utf-8'))
    assert data['dataset_version'] == gold['dataset_version']
    ids = [case['case_id'] for case in data['cases']]
    assert len(set(ids)) == len(ids) and set(ids) == set(gold['cases'])
    for case in data['cases']:
        assert set(case) == {'case_id', 'repetitions', 'batches', 'queries'}
        assert re.fullmatch('[a-z_]+', case['case_id'])
        assert type(case['repetitions']) is int and 1 <= case['repetitions'] <= 3
        assert case['batches'] and case['queries']
        for batch in case['batches']:
            assert batch and any(turn['role'] == 'user' for turn in batch)
            assert all(set(turn) == {'role', 'content'} and turn['role'] in ('user', 'assistant')
                       and isinstance(turn['content'], str) and 0 < len(turn['content']) <= settings.max_input_length for turn in batch)
        assert all(isinstance(query, str) and query.strip() for query in case['queries'])
    return data, gold


def build_plan(data, recovery=False):
    """Keep frozen conversations/gold unchanged; declare any repetition override."""
    if not recovery:
        return [(case, repeat) for case in data['cases'] for repeat in range(1, case['repetitions'] + 1)]
    cases = {case['case_id']: case for case in data['cases']}
    repeats = {'concurrent_jobs': 3, 'two_car_correction': 3,
               'uncertain_multilingual': 1, 'bob_employer': 1, 'brother_address': 1}
    return [(cases[name], repeat) for repeat in range(1, 4)
            for name, count in repeats.items() if repeat <= count]


def public_state(connection, cache):
    """Digest only: never export existing memories or credentials."""
    tables = {}
    for name in TABLES:
        row = connection.execute(sql.SQL(
            "SELECT count(*),md5(COALESCE(string_agg(h,'' ORDER BY h),'')) "
            "FROM (SELECT md5(row_to_json(t)::text) AS h FROM {} t) s"
        ).format(sql.Identifier('public', name))).fetchone()
        tables[name] = list(row)
    keys = {key.decode('utf-8'): hashlib.sha256(cache.dump(key) or b'').hexdigest()
            for key in sorted(cache.scan_iter())}
    return {'tables': tables, 'redis_key_digests': keys}


@contextmanager
def isolated_store(postgres_url, redis_url, cleanup):
    """Own one newly-created schema; all existing public tables are protected."""
    if any(urlsplit(url).hostname not in ('localhost', '127.0.0.1', '::1') for url in (postgres_url, redis_url)):
        raise ValueError('Local services required')
    schema = 'qm_' + uuid4().hex
    assert re.fullmatch(r'qm_[0-9a-f]{32}', schema)
    created = False
    provider = cache = None
    with psycopg.connect(postgres_url, autocommit=True) as owner, redis.Redis.from_url(redis_url) as raw:
        before = public_state(owner, raw)
        cleanup.update(schema=schema, before=before, verified=False)
        try:
            assert owner.execute('SELECT 1 FROM pg_namespace WHERE nspname=%s', (schema,)).fetchone() is None
            owner.execute(sql.SQL('CREATE SCHEMA {}').format(sql.Identifier(schema)))
            created = True
            scoped = make_conninfo(postgres_url, options='-c search_path=' + schema + ',public')
            provider = PostgresProvider(scoped)
            provider.ensure_infrastructure(settings.openai.vector_dimension)
            for name in TABLES:
                namespace = provider.conn.execute(
                    'SELECT n.nspname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.oid=%s::regclass', (name,)).fetchone()['nspname']
                assert namespace == schema, 'Evaluation table escaped isolated schema'
            cache = RedisCacheProvider(redis_url)
            yield provider, cache, schema
        finally:
            if cache is not None:
                # Tenant is the exact random schema name, never supplied by a user.
                keys = list(cache.client.scan_iter(match='*:' + schema + ':*'))
                if keys:
                    cache.client.delete(*keys)
                active = get_section_key_active_sessions()
                refs = [ref for ref in cache.client.zrange(active, 0, -1) if ref.startswith(schema + ':')]
                if refs:
                    cache.client.zrem(active, *refs)
                cache.client.close()
            if provider is not None:
                provider.close()
            if created:
                owner.execute(sql.SQL('DROP SCHEMA {} CASCADE').format(sql.Identifier(schema)))
            after = public_state(owner, raw)
            cleanup.update(after=after, verified=before == after)
            if before != after:
                raise RuntimeError('Protected-state digest changed')


class RecordedLLM:
    def __init__(self, live, calls):
        self.live, self.calls = live, calls
        self.vectors = {}

    def get_vector_dimension(self):
        return self.live.get_vector_dimension()

    def embed_text(self, text):
        key = ('query', text)
        if key not in self.vectors:
            self.vectors[key] = self.live.embed_text(text)
        return self.vectors[key]

    def embed_batch(self, texts):
        missing = list(dict.fromkeys(text for text in texts if ('document', text) not in self.vectors))
        if missing:
            values = self.live.embed_batch(missing)
            assert len(values) == len(missing)
            self.vectors.update({('document', text): value for text, value in zip(missing, values)})
        return [self.vectors[('document', text)] for text in texts]

    def is_junk(self, text):
        entry = {'stage': 'junk', 'input': json.loads(text)}
        self.calls.append(entry)
        try:
            entry['output'] = self.live.is_junk(text)
            return entry['output']
        except Exception as exc:
            entry['error_type'] = type(exc).__name__
            raise

    def parse_structured(self, system_prompt, user_text, schema, **kwargs):
        entry = {'stage': schema.__name__, 'input': json.loads(user_text),
                 'schema_sha256': hashlib.sha256(json.dumps(schema.model_json_schema(), sort_keys=True).encode()).hexdigest()}
        self.calls.append(entry)
        try:
            result = self.live.parse_structured(system_prompt, user_text, schema, **kwargs)
            entry['output'] = result.model_dump(mode='json')
            if hasattr(result, 'recovery_summary'):
                entry['recovery'] = result.recovery_summary()
            return result
        except Exception as exc:
            entry['error_type'] = type(exc).__name__
            raise


def stored(db, tenant, user):
    rows = db.conn.execute('SELECT payload FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s ORDER BY entity_id',
                           (tenant, user)).fetchall()
    for row in rows:
        Entity.model_validate(row['payload'])
    return [row['payload'] for row in rows]


def source_hashes():
    paths = list((ROOT / 'src').rglob('*.py')) + [Path(__file__), ROOT / 'scripts/eval_api_budget.py']
    return {str(path.relative_to(ROOT)).replace('\\', '/'): sha(path) for path in sorted(paths)}


def execute_case(case, repetition, db, cache, tenant, llm, budget, stamp, progress):
    user, session = 'synthetic-user', 'conversation'
    hook = ChatHookWorkflow(db=db, cache=cache, llm=llm)
    workflow = ConsolidationWorkflow(db=db, llm=llm)
    result = {'case_id': case['case_id'], 'repetition': repetition, 'batches': [], 'retrieval': [], 'completed': False}
    original_clock = redis_provider.time
    clock = [stamp]
    redis_provider.time = SimpleNamespace(time=lambda: clock[0])
    try:
        for batch_index, batch in enumerate(case['batches']):
            row = {'batch_index': batch_index, 'turns': deepcopy(batch), 'before': stored(db, tenant, user)}
            result['batches'].append(row)
            try:
                for turn in batch:
                    role = ChatSenderRole.USER if turn['role'] == 'user' else ChatSenderRole.ASSISTANT
                    payload = hook.on_chat(tenant, session, turn['content'], role, user)
                    if payload.get('error'):
                        raise RuntimeError('Chat hook error')
                    clock[0] += 60
                items = cache.pop_session_chat_log_4_consolidation(tenant, user, session)
                assert items and all(item['tenant_id'] == tenant and item['user_id'] == user for item in items)
                row['queued_messages'] = [{k: v for k, v in item.items() if k != 'vector'} for item in items]
                row['success'] = workflow.run_consolidation_cycle(tenant, user, items)
                row['after'] = stored(db, tenant, user)
                row['dictionary_feedback'] = getattr(workflow, 'last_dictionary_feedback', None)
                row['terminology_metrics'] = getattr(workflow, 'last_terminology_metrics', None)
                row['proposal_recovery'] = deepcopy(getattr(workflow, 'last_proposal_recovery', {}))
                row['mutation_recovery'] = deepcopy(getattr(workflow, 'last_mutation_recovery', {}))
                if row['success']:
                    before_requests = len(budget.records)
                    assert workflow.run_consolidation_cycle(tenant, user, items)
                    assert len(budget.records) == before_requests and stored(db, tenant, user) == row['after']
                    row['replay_verified'] = True
                    assert cache.clear_session_4_consolidation(tenant, user, session, items)
                else:
                    raise RuntimeError('Consolidation did not complete')
                progress(result)
                print(case['case_id'], repetition, 'batch', batch_index + 1, 'committed', len(row['after']), flush=True)
            except Exception as exc:
                row.update(error_type=type(exc).__name__, after=stored(db, tenant, user))
                row['error_frames'] = [{'file': Path(frame.filename).name, 'method': frame.name, 'line': frame.lineno}
                                       for frame in traceback.extract_tb(exc.__traceback__)]
                result['error_type'] = type(exc).__name__
                progress(result)
                return result
        result['final'] = stored(db, tenant, user)
        for query in case['queries']:
            clock[0] += 60
            payload = hook.on_chat(tenant, 'fresh-query-session', query, ChatSenderRole.USER, user)
            result['retrieval'].append({'query': query, 'payload': payload})
        result['completed'] = True
        return result
    except Exception as exc:
        result['error_type'] = type(exc).__name__
        return result
    finally:
        result['final'] = stored(db, tenant, user)
        redis_provider.time = original_clock
        progress(result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    variants = parser.add_mutually_exclusive_group()
    variants.add_argument('--recovery-pilot', action='store_true', help='Failed cases first; repeat jobs/cars three times, other cases once.')
    variants.add_argument('--subject-pilot', action='store_true', help='User jobs, brother jobs and stable car naming; three repetitions each.')
    variants.add_argument('--realistic-pilot', action='store_true', help='Five eight-message conversations; ambiguity, history, mixed topics and languages.')
    args = parser.parse_args()
    data_path, gold_path = case_paths(args.subject_pilot, args.realistic_pilot)
    data, gold = load_cases(args.subject_pilot, args.realistic_pilot)
    plan = build_plan(data, args.recovery_pilot)
    if not args.live and not args.preflight:
        print(json.dumps({'mode': 'validate', 'cases': len(data['cases']), 'runs': len(plan),
                          'batches': sum(len(case['batches']) for case, _ in plan),
                          'dataset_sha256': sha(data_path),
                          'expectations_sha256': sha(gold_path)}))
        return
    load_dotenv(ROOT / '.env', override=False)
    for name in ('tomte.engine', 'httpx', 'httpcore', 'openai'):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    assert settings.openai.strong_model == 'gpt-6.1-sol'
    assert settings.openai.cheap_model == 'gpt-6-luna'
    assert settings.openai.embed_model == 'openai:text-embedding-3-small'
    assert settings.openai.vector_dimension == 1536
    postgres_url, redis_url = os.environ['POSTGRES_URL'], os.environ['REDIS_URL']
    if args.preflight:
        cleanup = {}
        with isolated_store(postgres_url, redis_url, cleanup) as (db, cache, tenant):
            assert stored(db, tenant, 'preflight') == []
            cache.append_chat_history(tenant, 'preflight', 'synthetic isolation probe', role='user', user_id='preflight')
        print(json.dumps({'mode': 'preflight', 'isolated_schema_and_redis_cleanup_verified': cleanup['verified']}))
        return
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('API key missing')
    output = ROOT / 'log/memory_quality_v1' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    extended = args.recovery_pilot or args.subject_pilot
    budget = (ApiBudget(estimate='2.50', ceiling='3.70', request_limit=450, seconds=3600)
              if args.realistic_pilot else
              ApiBudget(estimate='1.50', ceiling='2.20', request_limit=320) if extended else ApiBudget())
    manifest = {'dataset_sha256': sha(data_path), 'expectations_sha256': sha(gold_path),
        'source_hashes': source_hashes(), 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_diff_sha256': hashlib.sha256(subprocess.check_output(['git', 'diff'], cwd=ROOT)).hexdigest(),
        'models': settings.openai.model_dump(), 'settings': settings.model_dump(),
        'package_versions': {package: version(package) for package in ('pydantic', 'pydantic-ai', 'openai', 'httpx', 'psycopg', 'redis')},
        'budget': budget.summary(), 'request_limit': budget.request_limit, 'time_limit_seconds': budget.seconds, 'output_token_limit': budget.output_tokens,
        'pricing_checked': '2026-11-04', 'pricing_source': 'https://developers.openai.com/api/docs/pricing',
        'estimate_basis': ('Approximately 100 strong calls at 7500 input/900 output tokens = $2.40; cheap calls and embeddings plus rounding about $0.10. Not the worst-case reservation.' if args.realistic_pilot else
                          'Approximately 70 strong calls at 6000 input/800 output tokens = $1.40; cheap calls and embeddings plus rounding about $0.10. Not the worst-case reservation.' if extended else
                          'Approximately 48 strong calls at 4000 input/1700 output tokens = $1.20; cheap calls and embeddings about $0.05. Not the worst-case reservation.'),
        'plan_variant': 'realistic-conversations-1' if args.realistic_pilot else 'subject-naming-1' if args.subject_pilot else 'recovery-1' if args.recovery_pilot else 'original-1',
        'controls': 'Standard tier, max 8192 output tokens, 96KB request limit, full input-byte token reservation plus 8192 framing tokens; unknown usage/error stops future calls.',
        'plan': [{'case_id': case['case_id'], 'repetition': repetition} for case, repetition in plan],
        'limitations': ['Small authored pilot, no population accuracy claim.', 'No prompt tuning; semantic adjudication by the coding assistant, not blinded independent humans.',
                       'Manual consolidation invocation; no worker scheduling/concurrency claims.', 'Small fresh corpus; recent-memory fallback can aid retrieval.',
                       'Fresh dictionary per run; not a mature shared vocabulary evaluation.', 'Explicit output cap may produce a distinct truncation failure.']}
    save(output / 'manifest.json', manifest)
    save(output / 'conversations.json', data)
    save(output / 'expectations.json', gold)
    print('OUTPUT', output, flush=True)
    print(f'BUDGET estimate={budget.estimate} ceiling={budget.ceiling} requests={budget.request_limit} time={budget.seconds}s', flush=True)
    results = []
    calls = []
    stamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00')).timestamp()
    live = OpenAILLMProvider(os.environ['OPENAI_API_KEY'])
    def record_http(entry):
        with (output / 'http_evidence.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + '\n')
    try:
        with bounded_http(budget, record_http):
            consecutive_failures = 0
            for case, repetition in plan:
                assert source_hashes() == manifest['source_hashes'], 'Source changed during frozen run'
                label = case['case_id'] + '-' + str(repetition)
                cleanup = {}
                case_calls = []
                calls.append({'run': label, 'calls': case_calls})
                try:
                    with isolated_store(postgres_url, redis_url, cleanup) as (db, cache, tenant):
                        llm = RecordedLLM(live, case_calls)
                        result = execute_case(case, repetition, db, cache, tenant, llm, budget, stamp,
                                              lambda row: save(output / (label + '.json'), row))
                    result['cleanup_verified'] = cleanup['verified']
                    results.append(result)
                    save(output / (label + '.json'), result)
                    consecutive_failures = 0 if result['completed'] else consecutive_failures + 1
                finally:
                    save(output / (label + '-cleanup.json'), cleanup)
                    save(output / 'model_calls.json', calls)
                    save(output / 'budget.json', budget.summary())
                if budget.stopped or consecutive_failures >= 2:
                    break
    except Exception as exc:
        manifest['execution_error_type'] = type(exc).__name__
    finally:
        save(output / 'model_calls.json', calls)
        save(output / 'budget.json', budget.summary())
        save(output / 'results.json', {'planned_runs': len(plan), 'attempted_runs': len(results),
                                     'completed_runs': sum(row['completed'] for row in results), 'runs': results})
        manifest['source_unchanged'] = source_hashes() == manifest['source_hashes']
        manifest['finished_utc'] = datetime.now(timezone.utc).isoformat()
        save(output / 'manifest.json', manifest)
    print(json.dumps({'output': str(output), 'completed_runs': sum(row['completed'] for row in results),
                      'attempted_runs': len(results), 'planned_runs': len(plan), 'budget': {k: v for k, v in budget.summary().items() if k != 'records'}}), flush=True)


if __name__ == '__main__':
    main()
