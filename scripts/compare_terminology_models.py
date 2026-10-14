"""Evaluation-only fixed-candidate comparison; no persistence or embeddings.

Run from tomte-engine. Uses the production prompt, schema, model runner and
validator, without altering global configuration. Synthetic outputs are saved
under ignored log/. Each request contains ten scored targets and ten unscored
empty-candidate companion targets. Native retries are unchanged (default: one).
Compares the configured strong baseline with the fast model; current production
terminology routing uses the fast model.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from urllib.parse import urlsplit

from dotenv import load_dotenv
import httpx
import redis
from pydantic_ai import Agent, capture_run_messages
from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart, TextPart

from evaluate_dictionary_representations import dictionary_state
from tomte.engine.core.config import settings
from tomte.engine.dictionary.alignment import TerminologyAlignmentResponse, apply_decisions
from tomte.engine.llm.prompts.terminology_alignment import TERMINOLOGY_ALIGNMENT_PROMPT
from tomte.engine.providers import openai_provider
from tomte.engine.providers.postgres_provider import PostgresProvider

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / 'log' / 'terminology_model_comparison.json'


def fixture(cases):
    raw, targets, view, scored = [], {}, [], []
    for i, case in enumerate(cases):
        eh, fh = f'entity_{i}', f'fact_{i}_0'
        attribute = case['raw_term'] if case['kind'] == 'ATTRIBUTE' else 'description'
        entity = {'entity_id': f'synthetic_{i}', 'entity_name': 'Synthetic subject',
                  'entity_domain': case['domain'], 'entity_type': case['entity_type'],
                  'aliases': [], 'facts': [{'attribute_name': attribute, 'value': case['value'],
                  'source_chat_id': f'synthetic_source_{i}', 'timeline': 1791547200}]}
        raw.append(entity)
        entry = {'entity_handle': eh, 'entity_name': entity['entity_name'],
                 'domain': case['domain'], 'raw_entity_type': entity['entity_type'],
                 'candidates': [], 'facts': []}
        fact = {'entity_handle': eh, 'fact_handle': fh, 'raw_attribute_name': attribute,
                'value_read_only': case['value'], 'source_text': [case['source']], 'candidates': []}
        entry['facts'].append(fact)
        view.append(entry)
        for kind, key, item, term, j in (
                ('ENTITY', (eh, None), entry, entity['entity_type'], None),
                ('ATTRIBUTE', (eh, fh), fact, attribute, 0)):
            candidates = []
            if case['kind'] == kind:
                for n, word in enumerate([case['candidate'], *case['distractors']]):
                    candidates.append({'candidate_handle': f'candidate_{2*i+(kind=="ATTRIBUTE")}_{n}',
                                       'keyword': word, 'keyword_type': kind,
                                       'dictionary_id': f'synthetic_anchor_{i}_{n}'})
                scored.append({'case': case, 'key': key})
            item['candidates'] = [{k: c[k] for k in ('candidate_handle', 'keyword')} for c in candidates]
            targets[key] = {'entity_index': i, 'fact_index': j, 'keyword_type': kind,
                            'raw_keyword': term, 'candidates': candidates, 'vector': []}
    # Do not send labels, database identifiers, scores, vectors or evaluator metadata.
    text = json.dumps({'entities': view}, ensure_ascii=False)
    return raw, targets, scored, text


def diagnostics(messages, targets):
    retries = unsupported = value_attempts = 0
    attempts = []
    for message in messages:
        for part in message.parts:
            if isinstance(part, RetryPromptPart):
                retries += 1
            if not isinstance(message, ModelResponse):
                continue
            data = None
            if isinstance(part, ToolCallPart):
                try:
                    data = json.loads(part.args) if isinstance(part.args, str) else part.args
                except (TypeError, ValueError):
                    pass
            elif isinstance(part, TextPart):
                try:
                    data = json.loads(part.content)
                except (TypeError, ValueError):
                    pass
            if not isinstance(data, dict):
                continue
            attempts.append(data)
            for group in ('entity_type_decisions', 'attribute_decisions'):
                for decision in data.get(group, []):
                    if not isinstance(decision, dict):
                        continue
                    key = (decision.get('entity_handle'), decision.get('fact_handle'))
                    if decision.get('decision') == 'REUSE' and not any(
                            c['candidate_handle'] == decision.get('candidate_handle')
                            for c in targets.get(key, {}).get('candidates', [])):
                        unsupported += 1
                    value_attempts += sum(k in decision for k in ('value', 'new_value', 'value_read_only'))
            value_attempts += sum(k in data for k in ('value', 'new_value', 'facts', 'entities'))
    return {'retries': retries, 'unsupported_attempts': unsupported,
            'value_change_attempts': value_attempts, 'wire_outputs': attempts}


def state(db, cache):
    counts = {}
    with db.conn.cursor() as cur:
        for table in ('tomte_user_registry', 'tomte_memory_engine', 'tomte_processed_consolidations'):
            cur.execute('SELECT COUNT(*) AS count FROM ' + table)
            counts[table] = cur.fetchone()['count']
    keys = sorted(k.hex() for k in cache.scan_iter(count=100))
    return {'counts': counts, 'dictionary': dictionary_state(db), 'redis_count': cache.dbsize(),
            'redis_key_digest': hashlib.sha256(json.dumps(keys).encode()).hexdigest()}


def percentile95(values):
    values = sorted(values)
    index = (len(values)-1)*.95
    lo = int(index)
    return values[lo] + (values[min(lo+1, len(values)-1)]-values[lo])*(index-lo)


def summarize(records):
    decisions = [d for r in records for d in r['decisions']]
    safe = [d for d in decisions if d['expected'] == 'REUSE']
    keep = [d for d in decisions if d['expected'] == 'KEEP_RAW']
    groups = defaultdict(list)
    for d in decisions:
        groups[d['id']].append((d['decision'], d['keyword']))
    repeated = {k: v for k, v in groups.items() if len(v) == 3}
    latencies = [r['seconds'] for r in records]
    usage = Counter()
    for r in records:
        usage.update(r['usage'])
    return {'decisions': len(decisions), 'safe_reuse_correct': sum(d['correct'] for d in safe),
            'safe_reuse_total': len(safe), 'keep_raw_correct': sum(d['correct'] for d in keep),
            'keep_raw_total': len(keep), 'overall_correct': sum(d['correct'] for d in decisions),
            'false_reuse': sum(d['false_reuse'] for d in decisions),
            'unsafe_terminal_proposals': sum(d['unsafe_proposal'] for d in decisions),
            'missing_decisions': sum(d['missing'] for d in decisions),
            'unstable_cases': {k: v for k, v in repeated.items() if len(set(v)) > 1},
            'repeated_cases': len(repeated), 'structured_retries': sum(r['retries'] for r in records),
            'raw_fallback_batches': sum(r['fallback'] for r in records),
            'unsupported_attempts': sum(r['unsupported_attempts'] for r in records),
            'value_change_attempts': sum(r['value_change_attempts'] for r in records),
            'latency_seconds': {'mean': mean(latencies), 'median': median(latencies),
                'p95': percentile95(latencies), 'min': min(latencies), 'max': max(latencies)},
            'usage': dict(usage), 'logical_requests': len(records),
            'average_input_tokens_per_request': usage['input_tokens']/max(usage['requests'],1),
            'average_output_tokens_per_request': usage['output_tokens']/max(usage['requests'],1)}


def main():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError('Run from tomte-engine')
    load_dotenv(ROOT / '.env', override=False)
    for name in ('OPENAI_API_KEY', 'POSTGRES_URL', 'REDIS_URL'):
        if not os.environ.get(name):
            raise RuntimeError('Missing required configuration: ' + name)
    for name in ('POSTGRES_URL', 'REDIS_URL'):
        if urlsplit(os.environ[name]).hostname not in ('localhost', '127.0.0.1', '::1'):
            raise RuntimeError('Only local infrastructure is allowed')
    for name in ('openai', 'httpx', 'httpcore'):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    cases = json.loads((ROOT/'scripts/terminology_model_comparison_cases.json').read_text())['cases']
    safe = [c for c in cases if c['expected']=='REUSE']
    keep = [c for c in cases if c['expected']=='KEEP_RAW']
    assert len(safe)==len(keep)==30 and sum(c['high_risk'] for c in cases)==20
    ordered = [c for pair in zip(safe, keep) for c in pair]
    batches = [ordered[i:i+10] for i in range(0,60,10)]
    batches += [batches[0], batches[1], batches[0], batches[1]]
    assert sum(len(b) for b in batches)==100
    models = {'strong_baseline': settings.openai.strong_model, 'fast': settings.openai.cheap_model}
    report = {'models': models, 'reasoning_effort': 'low for both', 'unique_cases':60,
              'scored_decisions_per_model':100, 'records': [], 'pricing':'No local pricing metadata',
              'prompt_sha256':hashlib.sha256(TERMINOLOGY_ALIGNMENT_PROMPT.encode()).hexdigest(),
              'schema_sha256':hashlib.sha256(json.dumps(TerminologyAlignmentResponse.model_json_schema(),sort_keys=True).encode()).hexdigest()}
    api = Counter()
    original_send = httpx.AsyncClient.send
    async def count_send(client, request, *args, **kwargs):
        if not request.url.path.endswith('/responses'):
            raise RuntimeError('Unexpected external route')
        api['responses'] += 1
        if api['responses'] > 60:
            raise RuntimeError('Evaluation API request budget exceeded')
        return await original_send(client, request, *args, **kwargs)
    db = PostgresProvider(os.environ['POSTGRES_URL'])
    cache = redis.Redis.from_url(os.environ['REDIS_URL'])
    # Initializing the existing provider makes no external call.
    provider = openai_provider.OpenAILLMProvider(os.environ['OPENAI_API_KEY'])
    report['before'] = state(db,cache)
    httpx.AsyncClient.send = count_send
    try:
        for batch_index, batch in enumerate(batches):
            raw, targets, scored, text = fixture(batch)
            # Sanity-check fixtures with production validation before paid calls.
            unchanged, _ = apply_decisions(raw, targets, TerminologyAlignmentResponse())
            assert unchanged == raw
            names = ('strong_baseline','fast') if batch_index%2==0 else ('fast','strong_baseline')
            for name in names:
                agent = Agent(output_type=TerminologyAlignmentResponse,
                              system_prompt=TERMINOLOGY_ALIGNMENT_PROMPT)
                result = None
                error = None
                with capture_run_messages() as messages:
                    started = perf_counter()
                    try:
                        result = openai_provider._run_structured_agent(agent, TerminologyAlignmentResponse,
                            'terminology_alignment', text, model='openai-responses:'+models[name],
                            model_settings={'openai_reasoning_effort':'low'})
                    except Exception as exc:
                        error = type(exc).__name__
                    seconds = perf_counter()-started
                info = diagnostics(messages, targets)
                responses = [m for m in messages if isinstance(m, ModelResponse)]
                usage = {'requests':len(responses), 'input_tokens':sum(m.usage.input_tokens for m in responses),
                         'output_tokens':sum(m.usage.output_tokens for m in responses)}
                output = result.output.model_dump(mode='json') if result is not None else None
                fallback = result is None
                try:
                    aligned, _ = apply_decisions(raw, targets, output) if output is not None else (deepcopy(raw), [])
                except Exception as exc:
                    error, fallback, aligned = type(exc).__name__, True, deepcopy(raw)
                terminal = {(d['entity_handle'],d.get('fact_handle')):d
                            for group in (output or {}).values() for d in group}
                scored_results=[]
                for item in scored:
                    case, key = item['case'], item['key']
                    target = targets[key]
                    proposal = terminal.get(key)
                    selected = next((c for c in target['candidates'] if proposal and
                                     c['candidate_handle']==proposal.get('candidate_handle')),None)
                    unsafe = bool(proposal and proposal['decision']=='REUSE' and
                                  (case['expected']=='KEEP_RAW' or selected is None or selected['keyword']!=case['candidate']))
                    i,j=target['entity_index'],target['fact_index']
                    word = aligned[i]['entity_type'] if j is None else aligned[i]['facts'][j]['attribute_name']
                    decision = 'KEEP_RAW' if fallback or proposal is None else proposal['decision']
                    correct = decision==case['expected'] and (decision=='KEEP_RAW' or word==case['candidate'])
                    scored_results.append({'id':case['id'],'expected':case['expected'],'decision':decision,
                        'keyword':word,'correct':correct,'false_reuse':decision=='REUSE' and not correct,
                        'unsafe_proposal':unsafe,'missing':proposal is None})
                # Application may change vocabulary only; everything else stays byte-equivalent.
                restored = deepcopy(aligned)
                for i,e in enumerate(raw):
                    restored[i]['entity_type']=e['entity_type']
                    restored[i]['facts'][0]['attribute_name']=e['facts'][0]['attribute_name']
                assert restored==raw
                record={'model':name,'batch':batch_index+1,'case_ids':[c['id'] for c in batch],
                        'input_sha256':hashlib.sha256(text.encode()).hexdigest(),'input':json.loads(text),
                        'seconds':seconds,'usage':usage,'fallback':fallback,'error_type':error,
                        'output':output,'decisions':scored_results,**info}
                report['records'].append(record)
                report['api_calls']=dict(api)
                OUTPUT.parent.mkdir(exist_ok=True)
                OUTPUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
                print(json.dumps({'model':name,'batch':batch_index+1,'seconds':round(seconds,3),
                    'correct':sum(d['correct'] for d in scored_results),'unsafe':sum(d['unsafe_proposal'] for d in scored_results),
                    'retries':info['retries'],'fallback':fallback,'error_type':error}),flush=True)
                if result is None:
                    raise RuntimeError('Model call failed; see sanitized evaluation result')
        report['summary']={name:summarize([r for r in report['records'] if r['model']==name]) for name in models}
    finally:
        httpx.AsyncClient.send=original_send
        report['after']=state(db,cache)
        report['state_unchanged']=report['before']==report['after']
        report['api_calls']=dict(api)
        OUTPUT.write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
        db.close()
        cache.close()
    assert report['state_unchanged']
    print(json.dumps(report['summary'],indent=2))


if __name__=='__main__':
    main()
