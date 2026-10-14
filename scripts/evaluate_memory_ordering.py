"""Evaluation-only memory ordering. Cached embeddings only; no production edits.
SQL fixtures are rolled back; only unique synthetic Redis keys are removed.
Full-cosine diagnostic uses cached fixture vectors, not an available public contract.
"""
from evaluate_retrieval_stack import (ROOT, build_dataset, rotated, FREQUENT, RECENT)
from evaluate_memory_retrieval import cosine, state, RollbackEvaluation
from evaluation_context_contract import assert_public_context_contract
from collections import Counter, defaultdict
from copy import deepcopy
import hashlib, json, logging, os, statistics, threading, time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
from dotenv import load_dotenv
import psycopg, redis
from tomte.engine import Tomte
from tomte.engine.core.config import settings
from tomte.engine.core.enums import ChatSenderRole, MemoryDomain
from tomte.engine.llm.embedder import get_entity_summary
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.providers.redis_provider import RedisCacheProvider
from tomte.engine.utils.keys import get_section_key_active_sessions
from tomte.engine.workflows.chat_hook import ChatHookWorkflow


def age_for(user,eid,bank):
    i=int(user[1:]); noise=rotated(FREQUENT,i); recent=rotated(RECENT,i)
    age=86400*(list(bank).index(eid)+1)
    if eid in noise: age=86400*(2+noise.index(eid))
    if eid in recent: age=60+60*recent.index(eid)
    if eid=='tesla': age=86400*365
    return age


def rank_candidates(entities,meta):
    ids=[e['entity_id'] for e in entities]; orders={}; timing={}
    policies={'current':('recent','raw','session'),'session_recent_semantic':('session','recent','raw'),
        'session_semantic_recent':('session','raw','recent'),'semantic_session_recent':('raw','session','recent'),
        'recent_session_semantic':('recent','session','raw')}
    for name,streams in policies.items():
        start=time.perf_counter_ns()
        orders[name]=list(dict.fromkeys(eid for s in streams for eid in sorted(
            [eid for eid in ids if meta[eid]['positions'][s] is not None],key=lambda eid:meta[eid]['positions'][s])))
        timing[name]=(time.perf_counter_ns()-start)/1000
    def score(eid,policy):
        m=meta[eid]; p=m['positions']; sem=m['similarity'] or 0
        session=0 if p['session'] is None else (5-p['session'])/5
        recent=0 if p['recent'] is None else (5-p['recent'])/5
        if policy=='add_small': return sem+.05*session+.02*recent
        if policy=='add_medium': return sem+.10*session+.05*recent
        if policy=='max_overlap': return max(sem,.35*session,.25*recent)+.02*max(0,len(m['sources'])-1)
        if policy=='time_continuum':
            age=m['age_seconds']; bonus=.10 if age<=600 else .07 if age<=28800 else .04 if age<=172800 else .02 if age<=604800 else 0
            return sem+.10*session+bonus
        if policy=='full_cosine_only': return m['full_similarity']
        if policy=='full_cosine_diagnostic': return m['full_similarity']+.10*session+.05*recent
        raise AssertionError(policy)
    for policy in ('add_small','add_medium','max_overlap','time_continuum','full_cosine_only','full_cosine_diagnostic'):
        start=time.perf_counter_ns()
        orders[policy]=sorted(ids,key=lambda eid:(-score(eid,policy),eid))
        timing[policy]=(time.perf_counter_ns()-start)/1000
    assert orders['current']==ids
    return orders,timing


def summarize(rows,policy):
    required=sum(len(r['required']) for r in rows)
    result=dict(turns=len(rows),required=required)
    for k in (1,3,5):
        hit=sum(len(set(r['orders'][policy][:k])&set(r['required'])) for r in rows)
        count=sum(len(r['orders'][policy][:k]) for r in rows)
        good=sum(len(set(r['orders'][policy][:k])&set(r['acceptable'])) for r in rows)
        result['recall'+str(k)]=hit/required if required else None
        result['precision'+str(k)]=good/count if count else None
        result['irrelevant'+str(k)]=(count-good)/len(rows) if rows else None
        chars=[sum(r['chars'][eid] for eid in r['orders'][policy][:k])+2*max(0,len(r['orders'][policy][:k])-1) for r in rows]
        result['chars'+str(k)]=statistics.mean(chars) if chars else None
        result['tokens'+str(k)]=result['chars'+str(k)]/4 if chars else None
    ranks=[r['orders'][policy].index(eid)+1 for r in rows for eid in r['required'] if eid in r['orders'][policy]]
    result['mean_required_rank']=statistics.mean(ranks) if ranks else None
    result['availability']=len(ranks)/required if required else None
    for source in ('recent','session'):
        eligible=[r for r in rows if set(r['required'])&set(r['orders'][policy])]
        for k in (3,5):
            # Missed available gold with an irrelevant source candidate ranked ahead.
            bad=sum(any(eid not in r['orders'][policy][:k] and any(
                other not in r['acceptable'] and source in r['metadata'][other]['sources']
                for other in r['orders'][policy][:k]) for eid in r['required'] if eid in r['metadata']) for r in eligible)
            result[source+'_displacement'+str(k)]=bad/len(eligible) if eligible else None
    times=[r['cpu_us'][policy] for r in rows]
    result['cpu_mean_us']=statistics.mean(times) if times else None
    result['cpu_p95_us']=sorted(times)[int(.95*(len(times)-1))] if times else None
    return result

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
    missing=[text for role in texts for text in texts[role] if key(role,text) not in vectors]
    if missing: raise RuntimeError('Cached embeddings missing; no external requests permitted')
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
    rows=[]
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
                    streams=dict(captured)
                    baseline=workflow.cog.merge_entities(streams['recent'],streams['raw'],streams['session'])
                    memories=[e for e in baseline if e['entity_domain']!=MemoryDomain.IDENTITY.value]
                    assert_public_context_contract(payload, workflow.cog)
                    scope_keys=[(e['tenant_id'],e['user_id'],e['id']) for e in memories]
                    assert len(scope_keys)==len(set(scope_keys))
                    required=set(turn['required'])-persona_ids
                    acceptable=(required|valid_topics|set(turn.get('acceptable',[])))-persona_ids
                    metadata={}
                    for e in memories:
                        eid=e['entity_id']
                        positions={s:next((i for i,x in enumerate(streams[s]) if x['id']==e['id']),None) for s in ('session','recent','raw')}
                        raw=next((x.get('raw_cosine') for x in streams['raw'] if x['id']==e['id']),None)
                        full=cosine(vector,vectors[key('document',get_entity_summary(bank[eid]))])
                        if raw is not None: assert abs(full-raw)<1e-5
                        age=age_for(user,eid,bank)
                        if turn.get('age_override'): age=turn['age_override'].get(eid,86400*400)
                        metadata[eid]=dict(positions=positions,similarity=raw,full_similarity=full,age_seconds=age,
                            sources=[s for s,p in positions.items() if p is not None])
                    orders,timings=rank_candidates(memories,metadata)
                    expected=set(e['entity_id'] for e in memories)
                    assert all(set(ids)==expected and len(ids)==len(expected) for ids in orders.values())
                    rows.append(dict(user=user,session=session,text=turn['text'],group=turn['group'],cold=cold_session,
                        required=sorted(required),acceptable=sorted(acceptable),metadata=metadata,orders=orders,cpu_us=timings,
                        candidates=sum(sum(e['entity_domain']!=MemoryDomain.IDENTITY.value for e in streams[s]) for s in ('session','recent','raw')),
                        unique=len(memories),chars={e['entity_id']:len(workflow.cog.format_entities([e])) for e in memories},
                        statements=actual_statements,calls=dict(turn_calls)))
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
                # Extra time-continuum probes reuse cached text, not new embeddings.
                # Seeded timestamps are evidence, not model labels inferred from candidates.
                for age,tag in ((120,'minutes'),(21600,'today'),(86400,'yesterday'),(259200,'few_days')):
                    for repeat in range(3):
                        overrides={'terry':age,'basketball':age+1}
                        db.conn.execute("UPDATE tomte_memory_engine SET last_mentioned_at=now()-interval '400 days' WHERE tenant_id=%s AND user_id=%s",(tenant,'u0'))
                        for eid,seconds in overrides.items():
                            db.conn.execute("UPDATE tomte_memory_engine SET last_mentioned_at=now()-(%s * interval '1 second') WHERE tenant_id=%s AND user_id=%s AND entity_id=%s",(seconds,tenant,'u0',eid))
                        turn=dict(text='What is the weather tomorrow?',required=[],acceptable=['terry','basketball'],group='TIME_'+tag,age_override=overrides)
                        evaluate('u0','time-'+tag+'-'+str(repeat),turn,True,set())
                for age,tag in ((120,'minutes'),(21600,'today'),(86400,'yesterday'),(259200,'few_days')):
                    for repeat in range(3):
                        overrides={'tesla':age}
                        db.conn.execute("UPDATE tomte_memory_engine SET last_mentioned_at=now()-interval '400 days' WHERE tenant_id=%s AND user_id=%s",(tenant,'u0'))
                        db.conn.execute("UPDATE tomte_memory_engine SET last_mentioned_at=now()-(%s * interval '1 second') WHERE tenant_id=%s AND user_id=%s AND entity_id='tesla'",(age,tenant,'u0'))
                        evaluate('u0','followup-'+tag+'-'+str(repeat),dict(text='What about winter?',required=['tesla'],acceptable=[],group='FOLLOWUP_'+tag,age_override=overrides),True,set())
                for kind in ('recent','session'):
                    for repeat in range(3):
                        overrides={'dog':120,'tesla':86400*365}
                        db.conn.execute("UPDATE tomte_memory_engine SET last_mentioned_at=now()-interval '400 days' WHERE tenant_id=%s AND user_id=%s",(tenant,'u0'))
                        for eid,seconds in overrides.items():
                            db.conn.execute("UPDATE tomte_memory_engine SET last_mentioned_at=now()-(%s * interval '1 second') WHERE tenant_id=%s AND user_id=%s AND entity_id=%s",(seconds,tenant,'u0',eid))
                        session='adverse-'+kind+'-'+str(repeat)
                        if kind=='session': cache.promote_session_entities(tenant,'u0',session,['dog'],10,86400)
                        evaluate('u0',session,dict(text='The vehicle I bought in 2024 needs service.',required=['tesla'],acceptable=[],group='ADVERSE_'+kind,age_override=overrides),kind!='session',set())
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
    policies=list(rows[0]['orders'])
    overlap=defaultdict(Counter)
    for row in rows:
        for eid,m in row['metadata'].items():
            tag='+'.join(m['sources'])
            overlap[tag].update(candidates=1,required=int(eid in row['required']),acceptable=int(eid in row['acceptable']))
    stats={p:summarize(rows,p) for p in policies}
    for p in policies:
        stats[p]['pareto']=not any(q!=p and all(stats[q][x]>=stats[p][x] for x in ('recall3','recall5','precision5'))
            and stats[q]['mean_required_rank']<=stats[p]['mean_required_rank']
            and (any(stats[q][x]>stats[p][x] for x in ('recall3','recall5','precision5')) or stats[q]['mean_required_rank']<stats[p]['mean_required_rank']) for q in policies)
    report=dict(dataset=dict(users=30,conversations=len(conversations),user_turns=len(rows),base_turns=606,extra_turns=len(rows)-606,
            entity_concepts=len(bank),stored_records=900,assistant_checks=assistant_checks),
        scenarios=dict(Counter(r['group'] for r in rows)),strategies=stats,
        strata={group:{p:summarize([r for r in rows if condition(r)],p) for p in policies}
            for group,condition in [('cold',lambda r:r['cold']),('warm',lambda r:not r['cold']),
                ('cross_session',lambda r:r['group']=='CROSS_SESSION_recent'),('old',lambda r:r['group'] in ('I_old','J_frequent_noise','K_recent_noise','ADVERSE_recent','ADVERSE_session'))]},
        groups={g:{p:summarize([r for r in rows if r['group']==g],p) for p in policies} for g in sorted({r['group'] for r in rows})},
        duplication=dict(before=sum(r['candidates'] for r in rows),after=sum(r['unique'] for r in rows),
            before_per_turn=statistics.mean(r['candidates'] for r in rows),after_per_turn=statistics.mean(r['unique'] for r in rows),
            duplicate_formatted_chars=sum(sum(r['chars'][eid]*max(0,len(m['sources'])-1) for eid,m in r['metadata'].items()) for r in rows)),
        overlap=dict(overlap),calls=dict(production_calls),statements=dict(Counter(r['statements'] for r in rows)),
        source_effect=dict(turns_where_add_medium_differs_from_semantic_precedence=sum(r['orders']['add_medium'][:5]!=r['orders']['semantic_session_recent'][:5] for r in rows),
            source_blind_to_informed={str(k):dict(improved=sum(len(set(r['orders']['full_cosine_diagnostic'][:k])&set(r['required']))>len(set(r['orders']['full_cosine_only'][:k])&set(r['required'])) for r in rows),harmed=sum(len(set(r['orders']['full_cosine_diagnostic'][:k])&set(r['required']))<len(set(r['orders']['full_cosine_only'][:k])&set(r['required'])) for r in rows)) for k in (3,5)},
            improved_top5_turns=sum(len(set(r['orders']['add_medium'][:5])&set(r['required']))>len(set(r['orders']['semantic_session_recent'][:5])&set(r['required'])) for r in rows),
            harmed_top5_turns=sum(len(set(r['orders']['add_medium'][:5])&set(r['required']))<len(set(r['orders']['semantic_session_recent'][:5])&set(r['required'])) for r in rows),
            explanation='Different internal evidence changes ordering, not evidence of caller-facing source-tag need; no downstream caller experiment.'),
        score_metadata=dict(missing_cosine=sum(m['similarity'] is None for r in rows for m in r['metadata'].values()),
            explanation='Observed-score policies use zero for unavailable cosine. Full-cosine diagnostic calculates cached fixture dot products without new retrieval; production lacks this all-candidate score contract.'),
        api=api,cleanup_verified=before==after,before=before,after=after,rows=rows,
        limitations=['Synthetic correlated 30-user catalogs, no held-out model evaluation',
            'No chat or embedding calls; fixed cached real embeddings',
            'Persona excluded by domain from all rankings/metrics',
            'Time probe substitutes basketball for soccer using existing cached corpus; weather text differs',
            'Timeline comes from controlled fixture timestamps; current recent provider does not expose timestamps',
            'All ranking strategies share candidates and freshest merged payloads; no suppression',
            'No required gold for weather; recent son/club acceptable, not automatically required',
            'Mean required rank conditional on available gold; availability reported separately',
            'Original corpus prior valid topics only apply within conversation; new time probes explicitly label cross-session continuity',
            'No caller behavior study; source identity necessity cannot be inferred from recall alone'])
    out=ROOT/'log/memory_ordering_evaluation.json'
    out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','before','after','groups')},indent=2))

if __name__=='__main__':
    try: main()
    except Exception as exc:
        import traceback
        print('Evaluation failed:',type(exc).__name__,'details withheld')
        print([(Path(f.filename).name,f.lineno,f.name) for f in traceback.extract_tb(exc.__traceback__)])
        raise SystemExit(1)
