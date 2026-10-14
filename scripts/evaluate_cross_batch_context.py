"""Offline raw-context counterfactuals using unchanged real-provider prompts.

Run from the repository root. No daemon, retrieval, persistence, dictionary or
Redis writes. Candidate fixtures are controlled, not a retrieval benchmark.
Snapshots/results go only under ignored log/cross_batch_context. Equivalent
matrix inputs reuse one observation; repetition runs explicitly bypass reuse.
"""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import time
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid5, NAMESPACE_URL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'log/cross_batch_context'
STRATEGIES = ['production', 'messages_4', 'messages_8', 'messages_16',
              'messages_32', 'previous_batch', 'tokens_500', 'tokens_1000',
              'tokens_2000', 'tokens_4000']


def dump(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def tokens(messages):
    # Explicit approximation, not the model tokenizer. Includes JSON role framing.
    return (len(json.dumps(messages, ensure_ascii=False)) + 3) // 4


def select(case, strategy):
    prior = case['prior']
    if strategy == 'production':
        return prior[-min(4, case.get('production_history_cap', 20)):] if case['session_scope'] == 'same_session' else []
    if strategy.startswith('messages_'):
        return prior[-int(strategy.split('_')[1]):]
    if strategy == 'previous_batch':
        return prior[case.get('previous_batch_start', 0):]
    result = []
    budget = int(strategy.split('_')[1])
    for item in reversed(prior):
        if tokens([item, *result]) > budget:
            break  # Whole-message tail; never silently cut an antecedent mid-message.
        result.insert(0, item)
    return result


def row(key, name, domain, kind, facts, aliases=()):
    return dict(id=str(uuid5(NAMESPACE_URL, 'cross-batch-eval:' + key)), entity_id=key,
                entity_name=name, entity_domain=domain, entity_type=kind,
                tenant_id='offline-context-eval', user_id='offline-user', payload=dict(
                    entity_id=key, entity_name=name, entity_domain=domain, entity_type=kind,
                    aliases=list(aliases), facts=[dict(fact_id=key+'-'+attribute,
                        attribute_name=attribute, value=value, assertion_state='ACTIVE',
                        confidence=1, source_chat_id='prior-'+key,
                        last_mentioned_at='2026-10-09T00:00:00Z') for attribute, value in facts]))


def datasets():
    traces = json.loads((ROOT/'evals/real_world/traces.json').read_text(encoding='utf-8'))
    fixture = json.loads((ROOT/'evals/correction_safety/honda_tesla_failure.json').read_text(encoding='utf-8'))
    cars = [row('tesla', 'Tesla Model Y', 'assets_and_possessions', 'vehicle',
                [('purchase_year','2024'), ('paint_color','red')], ['Tesla','Little Dog']),
            row('honda', 'Honda Civic', 'assets_and_possessions', 'vehicle', [('purchase_year','2021')], ['Honda']),
            row('terry', 'Terry', 'relationships', 'person', [('hobby','basketball')]),
            row('bryan', 'Bryan', 'relationships', 'person', [('hobby','swimming')])]
    cases = []
    def add(key, prior, current, expected, kind, candidates=None, scope='same_session', **kw):
        p=[{'role':v['role'], 'content':v['content']} for v in prior]
        b=[dict(item_id=key+'-'+str(i), role=v['role'],content=v['content'],
                timestamp=1791504060+i*12) for i,v in enumerate(current)]
        cases.append(dict(case_id=key, prior=p, current=b, expected=expected,
            failure_class=kind, session_scope=scope, candidates=deepcopy(candidates if candidates is not None else cars), **kw))
    t=fixture['turns']; current=fixture['stages']['ExtractionMemorySchema']['input']['CURRENT_BATCH']
    add('honda_recorded_new_session', t[:36], current,
        [('honda','sold'),('tesla','own'),('terry','Thursday')], 'descriptive',scope='new_session',previous_batch_start=28)
    cases[-1]['current']=deepcopy(current)  # Preserve exact recorded IDs/timestamps.
    add('honda_same_session_counterfactual',t[:36],current,
        [('honda','sold'),('tesla','own'),('terry','Thursday')], 'descriptive',previous_batch_start=28)
    def trace_case(tid, start, stop, expected, kind, **kw):
        tr=next(v for v in traces if v['trace_id']==tid)
        add(tid+'-'+start,tr['turns'][:next(i for i,v in enumerate(tr['turns']) if v['turn_id']==start)],
            tr['turns'][next(i for i,v in enumerate(tr['turns']) if v['turn_id']==start):next(i for i,v in enumerate(tr['turns']) if v['turn_id']==stop)+1],expected,kind,**kw)
    trace_case('long-01','u15','a18',[('tesla','blue')],'topic_return',previous_batch_start=0)
    trace_case('long-01','u08','a08',[('terry','Thursday')],'pronoun')
    trace_case('long-05','u05','a05',[('honda','2021')],'correction')
    trace_case('long-01','u13','a13',[('terry','basketball')],'repeated_fact')
    trace_case('long-09','u19','a19',[],'ordinal_abstain',scope='new_session',previous_batch_start=28)
    trace_case('long-01','u15','a15',[],'old_context_only')
    prefix=next(v for v in traces if v['trace_id']=='long-01')['turns'][:28]
    def turns(*messages):
        return [dict(role='user' if i%2==0 else 'assistant',content=text) for i,text in enumerate(messages)]
    focused=[
        ('employers', 'relationships','employer','Northstar Labs','Cedar Systems',
         'I work at Northstar Labs now. Cedar Systems was my previous employer.',
         'The current employer lets me work remotely on Fridays.', 'northstar','Fridays','descriptive'),
        ('projects','activities_and_skills','software_project','Tomte','ReefWatch',
         'Tomte is my memory software project. ReefWatch is my aquarium monitoring project.',
         'The aquarium project now uses SQLite.', 'reefwatch','SQLite','descriptive'),
        ('schools','relationships','school','Oak Secondary','Maple Elementary',
         'Terry attends Oak Secondary. Bryan attends Maple Elementary.',
         'The younger son has a trip to his school on Friday.', 'maple','Friday','descriptive'),
        ('homes','assets_and_possessions','home','Vancouver Home','Toronto Apartment',
         'My Vancouver Home is current. My Toronto Apartment was the previous place.',
         'The old place has a leaking roof.', 'toronto','leak','descriptive'),
    ]
    for label,domain,kind,a,b,antecedent,text,key,value,reference in focused:
        keys={'employers':('northstar','cedar'),'projects':('tomte','reefwatch'),
              'schools':('oak','maple'),'homes':('vancouver','toronto')}[label]
        rows=[row(keys[0],a,domain,kind,[('description',antecedent)]),row(keys[1],b,domain,kind,[('description',antecedent)])]
        add(label, prefix+turns(antecedent,'Okay.', 'That is the pair I mean.','Got it.',
             'Can you explain indexes?', 'An index speeds lookups.', 'yeah','Okay.'),
             turns(text,'Noted.'),[(key,value)],reference,rows,previous_batch_start=28)
    add('pronoun_new_unpersisted',prefix+turns('I started a software project named HarborNotes.','Tell me more.'),
        turns('It uses Rust for the backend.','Okay.'), [('harbornotes','Rust')], 'new_not_persisted',[])
    add('ordinal_other_car',prefix+turns('The Tesla needs charging. The Honda is the other car.','Okay.'),
        turns('The other one needs tires.','Noted.'),[('honda','tires')],'ordinal')
    add('assistant_useful',prefix+turns('I bought the Tesla in 2024 and the Honda in 2021.',
        'So the Tesla is the newer car.'),turns('The newer car needs tires.','Okay.'),[('tesla','tires')],'assistant')
    add('assistant_guess',prefix+turns('Terry plays basketball.', 'Perhaps Terry practices every day; that is only my guess.'),
        turns('yeah','Okay.'),[],'assistant_old_context')
    add('distraction',prefix+turns('I bought a Tesla in 2024 and a Honda in 2021.','Okay.',
        'Luna needs a checkup. Terry and Bryan have school events. Tomte uses Redis.', 'Got it.',
        'My Toronto apartment is old. ReefWatch is another project. I am planning a vacation.','Okay.',
        'Can we discuss aquarium filters?','Sure.'),turns('The car I bought in 2024 needs tires.','Okay.'),[('tesla','tires')],'distraction')
    long_text='I bought the Tesla Model Y in 2024 and the Honda Civic in 2021. '+(
        'Work interruptions make scheduling difficult. I want to keep the discussion organized while I explain the background. '*45)
    add('long_message',prefix+turns(long_text,'Okay.'),turns('The car I bought in 2024 needs tires.','Okay.'),[('tesla','tires')],'long_message')
    prior=prefix+turns('The Tesla is the car I charge at home.','Got it.')
    indirect=turns('It needs new tires.','Okay.')
    add('boundary_before_reference',prior,indirect,[('tesla','tires')],'boundary')
    add('boundary_after_reference',prefix,[*prior[len(prefix):],*indirect],[('tesla','tires')],'boundary')
    add('correction_pronoun',prefix+turns('The Honda Civic was bought in 2021.','Okay.'),
        turns('Actually, I was wrong: it was 2022, not 2021.','Got it.'),[('honda','2022')],'correction')
    return cases, fixture


def worker(job):
    from dotenv import load_dotenv
    import httpx
    from tomte.engine.providers import openai_provider
    from tomte.engine.core.extraction import ExtractionMemorySchema, resolve_extraction
    from tomte.engine.core.correction_safety import UnsupportedCorrection
    from tomte.engine.workflows.consolidation import ConsolidationWorkflow
    load_dotenv(ROOT/'.env', override=False)
    for name in ('tomte.engine','httpx','httpcore','openai'):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    requests=[]; retries=[]; calls=[]; original=httpx.AsyncClient.send; debug=openai_provider._debug_metadata
    async def record(client,request,*args,**kwargs):
        response=await original(client,request,*args,**kwargs)
        body=response.json()
        requests.append(dict(model=json.loads(request.content).get('model'),status=response.status_code,usage=body.get('usage',{})))
        return response
    httpx.AsyncClient.send=record
    openai_provider._debug_metadata=lambda metadata:retries.append(metadata)
    live=openai_provider.OpenAILLMProvider(os.environ['OPENAI_API_KEY'])
    class Recorder:
        def parse_structured(self,system_prompt,user_text,schema,**kw):
            payload=json.loads(user_text)
            # Counterfactual serialized model input only. Production helper still
            # applies its own four-message acceptance policy, measured separately.
            if schema.__name__=='ResolutionResponse':
                payload['REFERENCE_ONLY_CONTEXT']=job['reference']
            call=dict(stage=schema.__name__,input=payload,system_prompt=system_prompt)
            calls.append(call)
            result=live.parse_structured(system_prompt,json.dumps(payload,ensure_ascii=False),schema,**kw)
            call['output']=result.model_dump(mode='json')
            return result
    case=job['case']; data=dict(CURRENT_BATCH=case['current'],RECENT_CONTEXT_REFERENCE_ONLY=job['reference'])
    workflow=ConsolidationWorkflow.__new__(ConsolidationWorkflow)
    workflow.llm=Recorder()
    workflow.db=SimpleNamespace(get_resolution_candidates=lambda *a,**kw:case['candidates'][:5])
    result=dict(job_id=job['job_id'],case_id=case['case_id'],reference=job['reference'],calls=calls,requests=requests,retries=retries)
    start=time.perf_counter()
    try:
        extracted=workflow.llm.parse_structured(None,json.dumps(data),ExtractionMemorySchema)
        result['raw']=extracted.model_dump(mode='json')
        normalized=resolve_extraction(extracted,case['current'])
        entities=workflow._flatten_schema(normalized)
        result['entities']=entities
        if entities:
            operations=workflow._reconcile_with_golden_records('offline-context-eval','offline-user',
                entities,case['candidates'],reference_context=job['reference'],current_batch=case['current'])
        else:
            operations=workflow._resolve_contextual_actions('offline-context-eval','offline-user',data,[0.0]*1536)
        result['accepted_before_plan']=operations
        if operations is not None:
            try:
                prepared=workflow._plan_fact_mutations(operations,data,'offline-batch',case['candidates'])
                result['prepared']=prepared
                result['safeguard']='allowed'
            except UnsupportedCorrection:
                result['safeguard']='rejected'
            result['plan']=getattr(workflow,'last_mutation_plan',None)
        else:
            result['safeguard']='unresolved'
    except Exception as exc:
        result['error_type']=type(exc).__name__  # Never persist provider exception text.
    finally:
        result['seconds']=time.perf_counter()-start
        httpx.AsyncClient.send=original
        openai_provider._debug_metadata=debug
    return result


def local_state():
    import psycopg,redis
    from evaluate_memory_retrieval import state
    for name in ('POSTGRES_URL','REDIS_URL'):
        if urlsplit(os.environ.get(name,'')).hostname not in ('localhost','127.0.0.1','::1'):
            raise RuntimeError('Local services required; values withheld')
    with psycopg.connect(os.environ['POSTGRES_URL'],autocommit=True,options='-c default_transaction_read_only=on') as conn:
        cache=redis.Redis.from_url(os.environ['REDIS_URL'])
        result=state(conn,cache)
        digest=hashlib.sha256()
        for key in sorted(cache.scan_iter()):
            digest.update(hashlib.sha256(key).digest())
            digest.update(hashlib.sha256(cache.dump(key) or b'').digest())
        result['redis']['value_digest']=digest.hexdigest()
        cache.close()
        return result


def main():
    from dotenv import load_dotenv
    assert Path.cwd().resolve()==ROOT
    load_dotenv(ROOT/'.env',override=False)
    if not os.environ.get('OPENAI_API_KEY'):
        raise RuntimeError('OPENAI_API_KEY missing; value not inspected')
    OUT.mkdir(parents=True,exist_ok=True)
    cases,fixture=datasets(); dump('cases.json',cases)
    from tomte.engine.llm.prompts.extractor import EXTRACTOR_SYSTEM_PROMPT
    dump('instructions.json',dict(extractor=EXTRACTOR_SYSTEM_PROMPT,token_estimator='JSON characters / 4, rounded up',
        notes=['No persistence or retrieval; controlled candidate fixtures.',
            'Reference-only data separated from current batch; prompts unchanged.',
            'Nonproduction windows on new-session cases are cross-session counterfactuals.',
            'Identity model gets expanded offline input; unchanged helper acceptance still limits reference to four.',
            'Current junk filter bypassed so old-only extraction is directly measured.',
            'Identical matrix inputs share one observation, not independent trials.']))
    dump('honda_input_comparison.json',dict(full_conversation=fixture['turns'],
        actual_extraction=fixture['stages']['ExtractionMemorySchema'],
        actual_resolution=fixture['stages']['ResolutionResponse'],
        actual_plan=fixture['stages']['MutationPlanResponse'],
        batches=[dict(turns='u01–a14',reason='harness explicit a14 freeze'),
            dict(turns='u15–a18',reason='harness session switch flush'),
            dict(turns='u19–a27',reason='new session b, final flush')]))
    baseline=local_state();dump('state_before.json',baseline)
    jobs={};matrix=[]
    for case in cases:
        for strategy in STRATEGIES:
            ref=select(case,strategy)
            key=hashlib.sha256(json.dumps([case,ref],sort_keys=True).encode()).hexdigest()[:20]
            jobs.setdefault(key,dict(job_id=key,case=case,reference=ref))
            matrix.append(dict(case_id=case['case_id'],strategy=strategy,job_id=key,
                messages=len(ref),approx_tokens=tokens(ref)))
        if case['case_id'] in ('assistant_useful','assistant_guess'):
            ref=[v for v in select(case,'messages_8') if v['role']=='user']
            key='user-only-'+case['case_id']
            jobs[key]=dict(job_id=key,case=case,reference=ref)
            matrix.append(dict(case_id=case['case_id'],strategy='user_only_8',job_id=key,messages=len(ref),approx_tokens=tokens(ref)))
    dump('matrix.json',matrix)
    # Fixed repeat strategies include a small window and the full available raw
    # tail; selection was specified before seeing results to avoid cherry-picking.
    honda=next(c for c in cases if c['case_id']=='honda_same_session_counterfactual')
    for strategy in ('production','messages_32','tokens_2000'):
        for rep in range(1,11):
            key=f'honda-repeat-{strategy}-{rep}'
            jobs[key]=dict(job_id=key,case=honda,reference=select(honda,strategy))
    results=[]
    print('Cases',len(cases),'matrix entries',len(matrix),'unique trials including repeats',len(jobs),flush=True)
    try:
        with ProcessPoolExecutor(max_workers=3) as executor:
            futures={executor.submit(worker,job):key for key,job in jobs.items()}
            for future in as_completed(futures):
                result=future.result();results.append(result)
                dump('results.json',results)
                print('Completed',len(results),'/',len(jobs),result['case_id'],result.get('error_type','ok'),
                    'requests',len(result['requests']),flush=True)
    finally:
        after=local_state()
        dump('cleanup.json',dict(before=baseline,after=after,unchanged=baseline==after,
            synthetic_rows_or_keys_written=0,synthetic_rows_or_keys_removed=0))
    if baseline!=after:
        raise RuntimeError('Read-only evaluation state digest changed; investigate concurrent service writes')


if __name__=='__main__':
    main()
