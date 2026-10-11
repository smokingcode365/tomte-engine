"""Evaluation-only retrieval ablation. No production changes or model/chat calls.

Run from tomte-engine. Reuses prior real embeddings; only missing synthetic
embeddings are requested. SQL fixtures roll back and unique Redis state is deleted.
Alternative stacks merge the SAME live stream outputs, in production merge order.
Latency excludes external embedding latency and transaction begin/end round trips
because SQL runs within a rollback-only outer evaluation transaction.
"""
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import statistics
import threading
import time
from urllib.parse import urlsplit
from uuid import uuid4

from dotenv import load_dotenv
import httpx
import psycopg
import redis

from evaluate_entity_activation import dataset
from evaluate_memory_retrieval import entity, state, RollbackEvaluation
from tomte.engine import Tomte
from tomte.engine.core.config import settings
from tomte.engine.core.enums import ChatSenderRole, MemoryDomain
from tomte.engine.llm.embedder import get_entity_summary
from tomte.engine.providers.openai_provider import OpenAILLMProvider
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.providers.redis_provider import RedisCacheProvider
from tomte.engine.utils.keys import generate_session_entity_key, get_section_key_active_sessions
from tomte.engine.workflows.chat_hook import ChatHookWorkflow

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['persona', 'frequent', 'recent', 'semantic', 'session']
FREQUENT = ['guitar','degree','cat','current_employer','tokyo','certificate']
RECENT = ['dog','python','terry','main_aquarium','honda','tomte']


def rotated(values, i):
    offset=i % len(values)
    return values[offset:]+values[:offset]
CONFIGS = {
    'A': ['persona', 'frequent', 'recent', 'semantic', 'session'],
    'B': ['persona', 'semantic', 'session'],
    'C': ['persona', 'recent', 'session'],
    'D': ['persona', 'frequent', 'session'],
    'E': ['persona', 'session'],
    'F': ['persona', 'raw', 'session'],
    'G': ['persona', 'recent', 'raw', 'session'],
    'H': ['persona', 'frequent', 'raw', 'session'],
}


def build_dataset():
    bank, _, cases, previous_conversations = dataset()
    additions = [
        ('paris', 'Paris Holiday', 'events_and_logistics', 'vacation', ['France Trip'], [('destination', 'Paris'), ('travel_year', '2023')]),
        ('rabbit', 'Clover', 'relationships', 'pet_rabbit', ['my rabbit'], [('species', 'rabbit'), ('age', 'three years')]),
        ('rust', 'Rust', 'activities_and_skills', 'programming_language', [], [('use', 'systems programming'), ('learning_year', '2022')]),
        ('orchard', 'Garden Orchard', 'activities_and_skills', 'hobby', ['Apple Garden'], [('crop', 'apples'), ('location', 'backyard')]),
    ]
    for spec in additions:
        bank[spec[0]] = entity(*spec)
    # Labels are specified before retrieval, never inferred from returned IDs.
    category = {'name': 'A_explicit', 'alias': 'A_explicit', 'type': 'B_type',
                'ambiguous': 'B_type', 'fact': 'C_fact', 'indirect': 'D_indirect',
                'confusable': 'C_fact', 'multi': 'F_multi', 'negated': 'C_fact', 'unrelated': 'L_none'}
    cold = [dict(text=c['text'], required=c['gold'], group=category[c['category']], label=c['id']) for c in cases]
    for spec in additions:
        cold.append(dict(text='Tell me about ' + spec[1] + '.', required=[spec[0]], group='A_explicit', label='added-'+spec[0]))
    # Explicitly labeled prior-session topic, independent of retrieval output.
    # The prior conversation is not supplied to the new session API; recency
    # timestamps are the sole global hint. No claim of uniquely resolvable text.
    for i in range(30):
        topic=rotated(RECENT,i)[0]
        cold.append(dict(text='Following up on our last conversation, what should I do next?',
            required=[topic],group='CROSS_SESSION_recent',label='prior-session-'+str(i),
            user='u'+str(i),prior_session_topic=topic))
    # Thirty independent users/conversations with eight turns each. First two
    # new-session probes are cold; remaining six share a fresh conversation.
    conversations = []
    by_group = {g: [c for c in cold if c['group'] == g] for g in ['A_explicit','B_type','C_fact','D_indirect']}
    for i in range(30):
        g = list(by_group)[i % 4]
        first = deepcopy(by_group[g][(i // 4) % len(by_group[g])])
        second = dict(text='The vehicle I bought in 2024 needs service.', required=['tesla'],
                      group=['I_old','J_frequent_noise','K_recent_noise'][i % 3], label='controlled')
        turns = [dict(first, new_session=True), dict(second, new_session=True),
                 dict(text='My Tesla needs service.', required=['tesla'], group='A_explicit', new_session=True),
                 dict(text='What about winter?', required=['tesla'], group='E_continuity', new_session=False),
                 dict(text='My dog has been scratching.', required=['dog'], group='G_switch', new_session=False),
                 dict(text='Terry has basketball tonight.', required=['terry'], group='F_multi', new_session=False),
                 dict(text='Back to the car, what should I do?', required=['tesla'], group='H_return', new_session=False),
                 dict(text='Explain quantum mechanics.', required=[], group='L_none', new_session=False)]
        conversations.append(dict(user='u'+str(i), turns=turns))
    kinds={'discovery':'A_explicit','continuity':'E_continuity','reactivation':'H_return',
           'switch':'G_switch','ambiguous':'F_multi','ordinary':'F_multi'}
    for i,conversation in enumerate(previous_conversations):
        turns=[]
        first=True
        for turn in conversation['turns']:
            if turn['role']=='assistant':
                turns.append(dict(role='assistant',text=turn['text'],new_session=False))
                continue
            turns.append(dict(text=turn['text'],required=turn['gold'],
                group=kinds.get(turn['kind'],'F_multi') if turn['gold'] else 'L_none',new_session=first))
            first=False
        conversations.append(dict(user='u'+str(i%30),turns=turns))
    return bank, cold, conversations


def summarize(rows):
    required = sum(len(r['required']) for r in rows)
    hits = sum(len(set(r['ids']) & set(r['required'])) for r in rows)
    returned = sum(len(r['ids']) for r in rows)
    acceptable = sum(len(set(r['ids']) & set(r['acceptable'])) for r in rows)
    positive = [r for r in rows if r['required']]
    return dict(turns=len(rows), required=required, hits=hits,
        recall=hits/required if required else None, precision=acceptable/returned if returned else None,
        irrelevant_per_turn=statistics.mean(r['irrelevant'] for r in rows),
        unique_per_turn=statistics.mean(len(r['ids']) for r in rows),
        chars_per_turn=statistics.mean(r['chars'] for r in rows),
        approximate_tokens_per_turn=statistics.mean(r['chars']/4 for r in rows),
        duplicates_per_turn=statistics.mean(r['duplicates'] for r in rows),
        data_statements_per_turn=statistics.mean(r['statements'] for r in rows),
        db_ms_per_turn=statistics.mean(r['db_ms'] for r in rows),
        top_availability={str(k): statistics.mean(bool(set(r['ids'][:k]) & set(r['required'])) for r in positive) if positive else None for k in (1,3,5)},
        top_required_recall={str(k):sum(len(set(r['ids'][:k])&set(r['required'])) for r in rows)/required if required else None for k in (1,3,5)})


def stream_metrics(rows, order):
    counters=defaultdict(Counter)
    for row in rows:
        seen=set()
        required=set(row['required']);acceptable=set(row['acceptable'])
        for stream in order:
            ids=set(row['streams'][stream])
            others=set(eid for other in order if other!=stream for eid in row['streams'][other])
            counters[stream].update(candidates=len(ids),unique_useful_earlier=len((ids-seen)&required),
                redundant_earlier=len(ids&seen),unique_noise_earlier=len((ids-seen)-acceptable),
                required_exclusive=len((ids-others)&required),noise_exclusive=len((ids-others)-acceptable),
                duplicate_anywhere=len(ids&others))
            seen|=ids
    return dict(counters)


def main():
    assert Path.cwd().resolve() == ROOT
    load_dotenv(ROOT/'.env', override=False)
    for name in ('POSTGRES_URL','REDIS_URL'):
        if not os.environ.get(name) or urlsplit(os.environ[name]).hostname not in ('localhost','127.0.0.1','::1'):
            raise RuntimeError('Local services required; values withheld')
    assert settings.openai.embed_model == 'openai:text-embedding-3-small'
    assert settings.openai.vector_dimension == 1536
    assert settings.session_entity_cosine_floor == .25
    for name in ('tomte.engine','httpx','httpcore','openai'):
        logging.getLogger(name).setLevel(logging.CRITICAL)
    bank, cold, conversations = build_dataset()
    path = ROOT/'log/memory_retrieval_embeddings.json'
    vectors = json.loads(path.read_text())
    def key(role,text):
        return hashlib.sha256(json.dumps([settings.openai.embed_model,1536,role,text]).encode()).hexdigest()
    texts = {'document': sorted({get_entity_summary(e) for e in bank.values()}),
             'query': sorted({c['text'] for c in cold} | {t['text'] for c in conversations for t in c['turns']})}
    api = dict(requests=0, embedded_texts=0)
    original_send = httpx.AsyncClient.send
    async def embeddings_only(client, request, *args, **kwargs):
        if request.url.path != '/v1/embeddings':
            raise AssertionError('Only embeddings requests allowed')
        api['requests'] += 1
        return await original_send(client,request,*args,**kwargs)
    httpx.AsyncClient.send = embeddings_only
    try:
        for role in ('document','query'):
            missing = [text for text in texts[role] if key(role,text) not in vectors]
            if missing:
                if not os.environ.get('OPENAI_API_KEY'):
                    raise RuntimeError('Embedding credential missing; value withheld')
                provider = OpenAILLMProvider(os.environ['OPENAI_API_KEY'])
                result = provider.embedder.embed_documents_sync(missing) if role == 'document' else provider.embedder.embed_query_sync(missing)
                for text,vector in zip(missing,result.embeddings):
                    assert len(vector) == 1536
                    vectors[key(role,text)] = list(vector)
                api['embedded_texts'] += len(missing)
                path.write_text(json.dumps(vectors),encoding='utf-8')
    finally:
        httpx.AsyncClient.send = original_send
    print('Embedding cache ready:',api,flush=True)
    db = PostgresProvider(os.environ['POSTGRES_URL'])
    cache = RedisCacheProvider(os.environ['REDIS_URL'])
    observer = psycopg.connect(os.environ['POSTGRES_URL'],autocommit=True,options='-c default_transaction_read_only=on')
    raw_cache = redis.Redis.from_url(os.environ['REDIS_URL'])
    before = state(observer,raw_cache)
    tenant = 'stack-evaluation-'+uuid4().hex
    samples = defaultdict(list)
    turn_calls = Counter()
    turn_ms = Counter()
    capturing = False
    sql_count = [0]
    original_cursor = db.conn.cursor_factory
    class CountingCursor(original_cursor):
        def execute(self,query,*args,**kwargs):
            if capturing and str(query).lstrip().upper().startswith(('SELECT','INSERT','UPDATE','DELETE')):
                sql_count[0] += 1
            return super().execute(query,*args,**kwargs)
    db.conn.cursor_factory = CountingCursor
    names = dict(upsert_user_activity='heartbeat',get_important_entities='persona',
        get_recent_entities='recent',get_frequent_entities='frequent',semantic_retrieval='semantic',get_raw_semantic_context='raw',
        find_session_entity_mentions='lexical',discover_session_entities='activation',get_entities_by_ids='fetch')
    for method,label in names.items():
        original = getattr(db,method)
        def timed(*args,_original=original,_label=label,**kwargs):
            start=time.perf_counter()
            try: return _original(*args,**kwargs)
            finally:
                elapsed=(time.perf_counter()-start)*1000
                if capturing:
                    samples[_label].append(elapsed); turn_calls[_label]+=1;turn_ms[_label]+=elapsed
        setattr(db,method,timed)
    class CachedEmbedding:
        def embed_text(self,text): return vectors[key('query',text)]
    original_start=threading.Thread.start
    def forbidden(*args,**kwargs): raise AssertionError('Worker forbidden')
    threading.Thread.start=forbidden
    rows=[]; direct=[]; contribution=defaultdict(Counter); overlap=defaultdict(Counter)
    assistant_checks=0
    production_calls=Counter()
    persona_ids={eid for eid,e in bank.items() if e['entity_domain']==MemoryDomain.IDENTITY.value}
    captured={}
    try:
        Tomte.init(db,cache,CachedEmbedding(),start_daemon_worker=False)
        workflow=ChatHookWorkflow()
        original_build=workflow._build_knowledge_context
        def record(persona,recent,semantic,session=None):
            captured.update(persona=persona,recent=recent,raw=semantic,session=session or [])
            return original_build(persona,recent,semantic,session)
        workflow._build_knowledge_context=record
        try:
            with db.transaction():
                updates=[dict(mode='INSERT',tenant_id=tenant,user_id=c['user'],
                    **{f:e[f] for f in ('entity_id','entity_name','entity_domain','entity_type')},
                    payload=deepcopy(e),embedding=vectors[key('document',get_entity_summary(e))])
                    for c in conversations[:30] for e in bank.values()]
                assert db.commit_graph_updates(updates)
                assert db.conn.execute('SELECT count(*) AS n FROM tomte_memory_engine WHERE tenant_id=%s',(tenant,)).fetchone()['n']==900
                # Deliberately adverse old/low-frequency Tesla versus rotating
                # recent/high-frequency distractors. No ties in priority ranks.
                for i,c in enumerate(conversations[:30]):
                    noise=rotated(FREQUENT,i)
                    recent=rotated(RECENT,i)
                    for rank,eid in enumerate(bank):
                        mentions=1+rank
                        age=86400*(rank+1)
                        if eid in noise:
                            n=noise.index(eid); mentions=1000-n*100;age=86400*(2+n)
                        if eid in recent:
                            n=recent.index(eid);age=60+n*60
                        if eid=='tesla': mentions=1;age=86400*365
                        db.conn.execute('UPDATE tomte_memory_engine SET mention_count=%s,last_mentioned_at=now()-(%s * interval \'1 second\'),importance=%s WHERE tenant_id=%s AND user_id=%s AND entity_id=%s',
                            (mentions,age,50-rank,tenant,c['user'],eid))
                def evaluate(user,session,turn,cold_session,valid_topics):
                    nonlocal capturing
                    turn_calls.clear();turn_ms.clear();sql_count[0]=0
                    capturing=True
                    try:
                        payload=workflow.on_chat(tenant,session,turn['text'],ChatSenderRole.USER,user)
                    finally: capturing=False
                    actual_statements=sql_count[0]
                    assert turn_calls['frequent']==0 and turn_calls['semantic']==0
                    assert turn_calls['raw']==1 and turn_calls['recent']==1
                    production_calls.update(turn_calls)
                    vector=vectors[key('query',turn['text'])]
                    # Legacy comparison queries run AFTER the actual workflow,
                    # outside its counters. The raw stream comes from production.
                    streams=dict(captured)
                    legacy_ms={}
                    for label,call in [('frequent',lambda:db.get_frequent_entities(tenant,user)),
                                       ('semantic',lambda:db.semantic_retrieval(tenant,user,vector))]:
                        start=time.perf_counter();streams[label]=call()
                        legacy_ms[label]=(time.perf_counter()-start)*1000
                        samples[label].append(legacy_ms[label])
                    raw_ms=turn_ms['raw']
                    required=set(turn['required'])
                    # Prior declared-valid topics and all returned persona are
                    # acceptable; accidental MRU activations do NOT become gold.
                    acceptable=required|valid_topics|persona_ids
                    for config,order in CONFIGS.items():
                        merged=workflow.cog.merge_entities(*(streams[s] for s in order))
                        ids=[e['entity_id'] for e in merged]
                        context=workflow.cog.format_entities(merged)
                        if config=='G': assert context==payload['context_text']
                        candidates=sum(len(streams[s]) for s in order)
                        statements=actual_statements-int('recent' not in order)-int('raw' not in order)+int('frequent' in order)+int('semantic' in order)
                        db_ms=sum(turn_ms.values())-sum(turn_ms[s] for s in ('recent','raw') if s not in order)+sum(legacy_ms[s] for s in ('frequent','semantic') if s in order)
                        rows.append(dict(config=config,user=user,session=session,text=turn['text'],group=turn['group'],
                            cold=cold_session,required=sorted(required),acceptable=sorted(acceptable),ids=ids,
                            streams={s:[e['entity_id'] for e in streams[s]] for s in order},
                            irrelevant=len(set(ids)-acceptable),chars=len(context),duplicates=candidates-len(ids),
                            statements=statements,db_ms=db_ms,actual_production_statements=actual_statements,persona_miss=bool(turn_calls['persona'])))
                    for mode in ('semantic','raw'):
                        ids=[e['entity_id'] for e in streams[mode]]
                        direct.append(dict(mode=mode,text=turn['text'],group=turn['group'],cold=cold_session,
                            required=sorted(required),acceptable=sorted(acceptable),ids=ids,irrelevant=len(set(ids)-acceptable),
                            chars=len(workflow.cog.format_entities(streams[mode])),duplicates=0,statements=1,db_ms=legacy_ms['semantic'] if mode=='semantic' else raw_ms))
                    seen=set()
                    for s in ORDER:
                        ids=set(e['entity_id'] for e in streams[s])
                        others=set(e['entity_id'] for t in ORDER if t!=s for e in streams[t])
                        count=contribution[s]
                        count.update(candidates=len(ids),useful_unique_earlier=len((ids-seen)&required),redundant_earlier=len(ids&seen),
                            noise_unique_earlier=len((ids-seen)-acceptable),required_exclusive=len((ids-others)&required),
                            noise_exclusive=len((ids-others)-acceptable),duplicate_anywhere=len(ids&others),required_hits=len(ids&required))
                        for t in ORDER:
                            if t!=s: overlap[s][t]+=len(ids&{e['entity_id'] for e in streams[t]})
                        seen|=ids
                for i,turn in enumerate(cold):
                    evaluate(turn.get('user','u'+str(i%30)),'cold-'+str(i),turn,True,set())
                for c in conversations:
                    valid=set();session=None
                    for i,turn in enumerate(c['turns']):
                        if turn.get('role')=='assistant':
                            old=cache.get_session_entity_ids(tenant,c['user'],session,10)
                            workflow.on_chat(tenant,session,turn['text'],ChatSenderRole.ASSISTANT,c['user'])
                            assert cache.get_session_entity_ids(tenant,c['user'],session,10)==old
                            assistant_checks+=1
                            continue
                        if turn['new_session']:
                            valid=set();session='conversation-'+str(conversations.index(c))+'-'+str(i)
                        evaluate(c['user'],session,turn,turn['new_session'],valid)
                        valid|=set(turn['required'])
                raise RollbackEvaluation()
        except RollbackEvaluation: pass
    finally:
        capturing=False
        keys=list(cache.client.scan_iter(match='*'+tenant+'*'))
        if keys:cache.client.delete(*keys)
        active=get_section_key_active_sessions()
        refs=[ref for ref,_ in cache.client.zscan_iter(active) if ref.startswith(tenant+':')]
        if refs:cache.client.zrem(active,*refs)
        after=state(observer,raw_cache)
        observer.close();raw_cache.close();cache.client.close();db.close()
        threading.Thread.start=original_start
    assert before==after,'Cleanup failed'
    baseline_rows=[r for r in rows if r['config']=='A']
    report=dict(dataset=dict(users=30,conversations=len(conversations),entity_concepts=len(bank),stored_synthetic_records=len(bank)*30,
        user_turns=len(rows)//8,cold_sessions=sum(r['cold'] for r in baseline_rows),warm_turns=sum(not r['cold'] for r in baseline_rows),
        assistant_nonpromotion_checks=assistant_checks),api=api,
        scenarios=dict(Counter(r['group'] for r in rows if r['config']=='A')),
        configs={config:dict(streams=order,overall=summarize([r for r in rows if r['config']==config]),
            cold=summarize([r for r in rows if r['config']==config and r['cold']]),
            warm=summarize([r for r in rows if r['config']==config and not r['cold']]),
            groups={g:summarize([r for r in rows if r['config']==config and r['group']==g]) for g in sorted({r['group'] for r in rows})},
            first_supply=dict(Counter(s for r in rows if r['config']==config for eid in r['ids'] for s in [next(s for s in order if eid in r['streams'][s])])),
            stream_metrics=stream_metrics([r for r in rows if r['config']==config],order)) for config,order in CONFIGS.items()},
        direct={mode:dict(overall=summarize([r for r in direct if r['mode']==mode]),
            cold=summarize([r for r in direct if r['mode']==mode and r['cold']]),
            groups={g:summarize([r for r in direct if r['mode']==mode and r['group']==g]) for g in sorted({r['group'] for r in direct})}) for mode in ('semantic','raw')},
        contribution=dict(contribution),overlap=dict(overlap),
        latency={s:dict(n=len(v),mean_ms=statistics.mean(v),median_ms=statistics.median(v),p95_ms=sorted(v)[int(.95*(len(v)-1))]) for s,v in samples.items()},
        production_config='G',production_calls=dict(production_calls),
        production_statement_distribution=dict(Counter(r['actual_production_statements'] for r in rows if r['config']=='G')),
        cleanup_verified=True,before=before,after=after,rows=rows,direct_rows=direct,
        limitations=['Single-machine rollback transaction latency; excludes begin/end and embeddings',
          'Production G context matches actual workflow exactly; legacy comparison queries are outside live counters',
          'Alternatives replay identical outputs; statement savings derived from actual production SQL counts',
          'Synthetic 30-user catalogs share 30 concepts; outcomes are correlated, not independent samples',
          'Prior gold topics acceptable; all identity persona acceptable even for no-memory queries',
          'Cross-session followups require a labeled prior topic; text alone cannot uniquely identify it',
          'Top-k availability means any required entity available, not answer accuracy; context is not relevance-ranked'])
    conceptual=[]
    for row in rows:
        if row['config']!='A':continue
        ids=list(dict.fromkeys(eid for stream in ['persona','session','frequent','recent','semantic'] for eid in row['streams'][stream]))
        assert set(ids)==set(row['ids'])
        conceptual.append(dict(row,ids=ids))
    report['conceptual_order_comparison']=summarize(conceptual)
    (ROOT/'log/retrieval_stack_evaluation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','direct_rows','before','after')},indent=2))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        import traceback
        print('Evaluation failed:',type(exc).__name__,'details withheld')
        print([(Path(f.filename).name,f.lineno,f.name) for f in traceback.extract_tb(exc.__traceback__)])
        raise SystemExit(1)
