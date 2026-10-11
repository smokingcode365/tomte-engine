"""Reusable real-provider evaluation. Run from tomte-engine, with local services.
No daemon, production edits or gold-label injection. Only synthetic data is removed.
Requires an initially empty dictionary so shared dictionary writes are all ours.
"""
import argparse, csv, hashlib, html, json, logging, os, statistics, threading, time
from collections import Counter
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit
from uuid import uuid4
import httpx, psycopg, redis
from dotenv import load_dotenv
from evaluate_memory_retrieval import state
from evaluate_memory_ordering import rank_candidates
from tomte.engine import Tomte
from tomte.engine.core.config import settings
from tomte.engine.core.enums import ChatSenderRole
from tomte.engine.core.assertions import project_facts
from tomte.engine.dictionary import service
from tomte.engine.exceptions import TomteSessionStateUnavailable
from tomte.engine.providers import openai_provider, redis_provider
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.providers.redis_provider import RedisCacheProvider
from tomte.engine.workflows.chat_hook import ChatHookWorkflow
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from tomte.engine.utils.keys import get_section_key_active_sessions, generate_consolidation_key
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'evals/real_world'
OUTPUT=ROOT/'log/real_world'
POLICIES=('current','session_recent_semantic','session_semantic_recent','semantic_session_recent','max_overlap','add_medium')


def load_traces():
    traces=json.loads((INPUT/'traces.json').read_text(encoding='utf-8'))
    truth={t['trace_id']:t for t in json.loads((INPUT/'truth.json').read_text(encoding='utf-8'))}
    assert len(traces)>=12 and len({t['user_id'] for t in traces})>=6
    assert sum(t['trace_class']=='manual' for t in traces)>=5
    assert sum(t['trace_class']=='adversarial' for t in traces)>=4
    for trace in traces:
        assert 50<=len(trace['turns'])<=150
        assert all(set(turn)=={'turn_id','role','content','session_id','relative_seconds'} for turn in trace['turns'])
        ids={t['turn_id'] for t in trace['turns']}
        assert len(ids)==len(trace['turns'])
        assert all(c['turn_id'] in ids for c in truth[trace['trace_id']]['retrieval_checkpoints'])
    return traces,truth


def entity_key(row,truth):
    names=' '.join([row['entity_name'],*row['payload'].get('aliases',[])]).lower()
    matched=[key for key,spec in truth['entities'].items() if any(name.lower() in names for name in spec['names'])]
    return matched[0] if len(matched)==1 else None


def fact_matches(fact,spec):
    key=fact['attribute_name'].lower(); value=str(fact['value']).lower()
    scalar_color=any(k in ('color','colour','exterior') for k in spec['attribute_contains'])
    value_match=any(value.strip()==v.lower() if scalar_color else v.lower() in value for v in spec['values'])
    if spec.get('value_rule')=='positive_ownership':
        value_match=value_match and not any(x in value for x in ('no longer','not own','sold','previously owned','unknown','uncertain'))
    return any(k in key for k in spec['attribute_contains']) and value_match


def evaluate_state(records,truth,sources):
    groups={k:[] for k in truth['entities']}; extra=[]
    for row in records:
        k=entity_key(row,truth)
        (groups[k] if k else extra).append(row)
    matched=[]; missing=[]
    for spec in truth['expected_final_facts']:
        found=any(f.get('assertion_state','ACTIVE')==spec['state'] and fact_matches(f,spec)
                  for row in groups[spec['entity']] for f in row['payload']['facts'])
        (matched if found else missing).append(spec)
    forbidden=[spec for spec in truth['forbidden_active'] if any(fact_matches(f,spec)
        for row in groups[spec['entity']] for f in project_facts(row['payload']['facts']))]
    aliases={key:all(any(a.lower() in [x.lower() for x in row['payload'].get('aliases',[])] for row in groups[key]) for a in expected)
             for key,expected in truth['expected_aliases'].items()}
    duplicate_entities=sum(max(0,len(items)-1) for items in groups.values())
    duplicates=0; unsupported=[]; contradictions=[]; total=0; checklist_hits=0
    for row in records:
        active=project_facts(row['payload']['facts']); total+=len(active)
        counts=Counter((f['attribute_name'].lower(),str(f['value']).lower()) for f in active)
        duplicates+=sum(n-1 for n in counts.values())
        key=entity_key(row,truth)
        for f in active:
            if any(spec['entity']==key and fact_matches(f,spec) for spec in truth['expected_final_facts'] if spec['state']=='ACTIVE'): checklist_hits+=1
            evidence=f.get('evidence',[])
            ids={e['source_chat_id'] for e in evidence} or {f.get('source_chat_id')}
            supplied=[sources.get(x) for x in ids]
            if any(x is None for x in supplied): unsupported.append(dict(reason='unknown_source',entity=row['entity_name'],fact=f))
            elif all(x['role']=='assistant' or x['filler'] for x in supplied): unsupported.append(dict(reason='assistant_or_filler_only_evidence',entity=row['entity_name'],fact=f))
        for attribute in {f['attribute_name'] for f in active}:
            if any(x in attribute.lower() for x in ('color','colour','purchase_year','birth_year')):
                values={str(f['value']).lower() for f in active if f['attribute_name']==attribute}
                if len(values)>1: contradictions.append(dict(entity=row['entity_name'],attribute=attribute,values=sorted(values)))
    return dict(expected_entities=len(groups),found_entities=sum(bool(v) for v in groups.values()),missing_entities=[k for k,v in groups.items() if not v],
        extra_entities=[r['entity_name'] for r in extra],duplicate_entities=duplicate_entities,expected_fact_checklist=len(matched)+len(missing),matched_facts=len(matched),
        checklist_recall=len(matched)/(len(matched)+len(missing)),active_checklist_precision=checklist_hits/total if total else 0,
        missing_facts=missing,forbidden_active=forbidden,duplicate_facts=duplicates,contradictions=contradictions,
        unsupported_provenance=unsupported,aliases=aliases,active_fact_count=total,
        caveat='Checklist precision is not semantic fact precision; extra supported facts and synonyms require human review.')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--limit',type=int);parser.add_argument('--repeat-critical',type=int,default=3)
    parser.add_argument('--trace-ids',nargs='+')
    parser.add_argument('--output-dir',default='log/real_world')
    parser.add_argument('--validate-only',action='store_true');args=parser.parse_args()
    traces,truths=load_traces()
    if args.validate_only:
        print(json.dumps(dict(traces=len(traces),users=len({t['user_id'] for t in traces}),turns=sum(len(t['turns']) for t in traces))));return
    assert Path.cwd().resolve()==ROOT
    global OUTPUT
    OUTPUT=(ROOT/args.output_dir).resolve()
    assert OUTPUT.is_relative_to(ROOT/'log'), 'Generated evidence must remain under ignored log/'
    load_dotenv(ROOT/'.env',override=False)
    if not os.environ.get('OPENAI_API_KEY'): raise RuntimeError('Credential missing; value withheld')
    for name in ('POSTGRES_URL','REDIS_URL'):
        if urlsplit(os.environ.get(name,'')).hostname not in ('127.0.0.1','localhost','::1'): raise RuntimeError('Local services required; values withheld')
    assert settings.openai.vector_dimension==1536 and not settings.enable_pii_scrubbing
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for name in ('tomte.engine','httpx','httpcore','openai'): logging.getLogger(name).setLevel(logging.CRITICAL)
    api=Counter(); requests=[]; model_calls=[]; retry_diagnostics=[]; current={}
    send=httpx.AsyncClient.send; diagnostic=openai_provider._debug_metadata
    async def observe(client,request,*a,**kw):
        route=request.url.path
        if not route.endswith(('/responses','/embeddings')): raise RuntimeError('Unexpected external route')
        body=json.loads(request.content); model=body.get('model'); api[model]+=1
        response=await send(client,request,*a,**kw)
        data=response.json(); usage=data.get('usage',{})
        requests.append(dict(trace=current.get('trace'),route=route,model=model,status=response.status_code,usage=usage))
        return response
    httpx.AsyncClient.send=observe
    openai_provider._debug_metadata=lambda item:retry_diagnostics.append(dict(trace=current.get('trace'),**item))
    live=openai_provider.OpenAILLMProvider(os.environ['OPENAI_API_KEY'])
    vector_path=OUTPUT/'embeddings.json'; vectors=json.loads(vector_path.read_text(encoding='utf-8')) if vector_path.exists() else {}
    def vector_key(role,text):return hashlib.sha256(json.dumps([settings.openai.embed_model,1536,role,text]).encode()).hexdigest()
    class RecordedProvider:
        def get_vector_dimension(self):return live.get_vector_dimension()
        def embed_text(self,text):
            key=vector_key('query',text)
            if key not in vectors:
                vectors[key]=live.embed_text(text);vector_path.write_text(json.dumps(vectors),encoding='utf-8')
            return vectors[key]
        def embed_batch(self,texts):
            missing=list(dict.fromkeys(t for t in texts if vector_key('document',t) not in vectors))
            if missing:
                for t,v in zip(missing,live.embed_batch(missing)):vectors[vector_key('document',t)]=v
                vector_path.write_text(json.dumps(vectors),encoding='utf-8')
            return [vectors[vector_key('document',t)] for t in texts]
        def is_junk(self,text):
            answer=live.is_junk(text); model_calls.append(dict(trace=current.get('trace'),stage='junk',input=json.loads(text),output=answer));return answer
        def parse_structured(self,system_prompt,user_text,schema,**kwargs):
            record=dict(trace=current.get('trace'),stage=schema.__name__,input=json.loads(user_text))
            try:
                result=live.parse_structured(system_prompt,user_text,schema,**kwargs); record['output']=result.model_dump(mode='json');return result
            except Exception as e:record['error_type']=type(e).__name__;raise
            finally:model_calls.append(record)
    llm=RecordedProvider()
    # Prefetch only user/later-query vectors, using the real unchanged query model.
    selected=[t for t in traces if not args.trace_ids or t['trace_id'] in args.trace_ids]
    if args.trace_ids:assert {t['trace_id'] for t in selected}==set(args.trace_ids)
    if args.limit:selected=selected[:args.limit]
    plan=[(t,1) for t in selected]+[(t,rep) for t in selected if t['trace_id'] in ('long-01','long-09') for rep in range(2,args.repeat_critical+1)]
    texts=sorted({turn['content'] for t,_ in plan for turn in t['turns'] if turn['role']=='user'}|{q['text'] for t,_ in plan for q in truths[t['trace_id']]['later_queries']})
    missing=[t for t in texts if vector_key('query',t) not in vectors]
    for offset in range(0,len(missing),100):
        chunk=missing[offset:offset+100];result=live.embedder.embed_query_sync(chunk)
        for t,v in zip(chunk,result.embeddings):vectors[vector_key('query',t)]=list(v)
        vector_path.write_text(json.dumps(vectors),encoding='utf-8')
    db=PostgresProvider(os.environ['POSTGRES_URL']);cache=RedisCacheProvider(os.environ['REDIS_URL'])
    observer=psycopg.connect(os.environ['POSTGRES_URL'],autocommit=True,options='-c default_transaction_read_only=on')
    raw=redis.Redis.from_url(os.environ['REDIS_URL']);before=state(observer,raw)
    if any(before['postgres'][name]['count'] for name in ('tomte_semantic_anchors','tomte_keyword_registry','tomte_keyword_audit_log')):
        raise RuntimeError('An empty dictionary is required; existing dictionary was not modified')
    prefix='world-eval-'+uuid4().hex;anchors=[];sources={};runs=[]; reliability=[]
    create=db.create_dictionary_anchor
    def track(vector):
        anchor=create(vector);anchors.append(str(anchor));return anchor
    db.create_dictionary_anchor=track
    old_start=threading.Thread.start;old_clock=redis_provider.time
    virtual=[time.time()];redis_provider.time=SimpleNamespace(time=lambda:virtual[0])
    try:
        Tomte._db=Tomte._cache=Tomte._llm=Tomte._worker=None;service._dictionary_service_instance=None
        Tomte.init(db,cache,llm,start_daemon_worker=False)
        def records(tenant,user):
            return db.conn.execute('SELECT id::text,id AS native_id,tenant_id,user_id,entity_id,entity_name,entity_domain,entity_type,payload FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s ORDER BY entity_id',(tenant,user)).fetchall()
        for index,(trace,rep) in enumerate(plan):
            tid=trace['trace_id'];current['trace']=tid+'-rep'+str(rep)
            tenant=prefix+'-'+trace['tenant_id']; user=trace['user_id'] if rep==1 else trace['user_id']+'-repeat-'+str(rep)+'-'+tid
            truth=truths[tid];checkpoints={c['turn_id']:c for c in truth['retrieval_checkpoints']};annotations={a['turn_id']:a for a in truth['turn_annotations']}
            hook=ChatHookWorkflow();workflow=ConsolidationWorkflow();captured={};retrieval=[];batch_results=[];turn_results=[];filler=[];previous=None;prior_topics=set();session=None;queued=0
            original=hook._build_knowledge_context
            def capture(persona,recent,semantic,session_entities=None):
                captured.update(persona=persona,recent=recent,raw=semantic,session=session_entities or [])
                return original(persona,recent,semantic,session_entities)
            hook._build_knowledge_context=capture
            def consolidate(session):
                items=cache.pop_session_chat_log_4_consolidation(tenant,user,session)
                if not items:return
                assert all(i['tenant_id']==tenant and i['user_id']==user for i in items)
                initial=deepcopy(records(tenant,user));calls=len(requests)
                result=dict(items=[i['item_id'] for i in items],messages=len(items))
                try:
                    result['success']=workflow.run_consolidation_cycle(tenant,user,items)
                    if result['success']:
                        committed=deepcopy(records(tenant,user));n=len(requests)
                        assert workflow.run_consolidation_cycle(tenant,user,items)
                        assert records(tenant,user)==committed and len(requests)==n
                        result['replay_no_api_or_row_change']=True
                        assert cache.clear_session_4_consolidation(tenant,user,session,items)
                    else:result['retained_for_retry']=True
                except Exception as e:
                    result.update(error_type=type(e).__name__,unchanged_rows=initial==records(tenant,user),retained_for_retry=True)
                result['requests']=len(requests)-calls;batch_results.append(result)
                print(current['trace'],'batch',len(batch_results),'success',result.get('success',False),'error',result.get('error_type','none'),flush=True)
            for turn in trace['turns']:
                new_session=tid+'-'+turn['session_id']+'-rep'+str(rep)
                if session and new_session!=session:
                    consolidate(session)
                    with db.transaction():db.conn.execute("UPDATE tomte_memory_engine SET last_mentioned_at=last_mentioned_at-interval '2 days' WHERE tenant_id=%s AND user_id=%s",(tenant,user))
                    queued=0;prior_topics=set();previous=None
                session=new_session;virtual[0]=time.time()+index*259200+turn['relative_seconds']
                before_mru=cache.get_session_entity_ids(tenant,user,session,10)
                role=ChatSenderRole.USER if turn['role']=='user' else ChatSenderRole.ASSISTANT
                payload=hook.on_chat(tenant,session,turn['content'],role,user)
                key=generate_consolidation_key(tenant,user,session)
                item=json.loads(cache.client.lindex(key,-1));sources[item['item_id']]=dict(turn_id=turn['turn_id'],role=turn['role'],filler=annotations.get(turn['turn_id'],{}).get('filler',False),content=turn['content'])
                after_mru=cache.get_session_entity_ids(tenant,user,session,10)
                if role==ChatSenderRole.ASSISTANT:assert before_mru==after_mru
                else:
                    merged=hook.cog.merge_entities(captured['recent'],captured['raw'],captured['session'])
                    entities=[e for e in merged if e['entity_domain']!='identity_and_demographics'];meta={};byid={e['entity_id']:e for e in entities}
                    for e in entities:
                        assert e['tenant_id']==tenant and e['user_id']==user,'Isolation failure'
                        pos={s:next((j for j,x in enumerate(captured[s]) if x['id']==e['id']),None) for s in ('session','recent','raw')}
                        similarity=next((x.get('raw_cosine') for x in captured['raw'] if x['id']==e['id']),None)
                        meta[e['entity_id']]=dict(positions=pos,similarity=similarity,full_similarity=similarity or 0,age_seconds=0,sources=[s for s,p in pos.items() if p is not None])
                    orders,cpu=rank_candidates(entities,meta)
                    labels={eid:entity_key(dict(entity_name=e['entity_name'],payload=e),truth) for eid,e in byid.items()}
                    cp=checkpoints.get(turn['turn_id']);required=set(cp['required']) if cp else set();acceptable=required|prior_topics|(set(cp['acceptable']) if cp else set())
                    row=dict(turn_id=turn['turn_id'],required=sorted(required),acceptable=sorted(acceptable),orders={p:[labels[eid] or 'extra:'+eid for eid in orders[p]] for p in POLICIES},candidate_ids=list(byid),candidate_meta=meta,mru_before=before_mru,mru_after=after_mru,semantic_ids=[e['entity_id'] for e in captured['raw']],cpu_us={p:cpu[p] for p in POLICIES},checkpoint=cp,context_text=payload['context_text'])
                    turn_results.append(row)
                    if cp:retrieval.append(row);prior_topics|=required
                    if annotations.get(turn['turn_id'],{}).get('filler') and previous:
                        shared=set(previous['orders']['current'])&set(row['orders']['current'])
                        filler.append(dict(turn_id=turn['turn_id'],mru_changed=before_mru!=after_mru,semantic_jaccard=len(set(previous['semantic_ids'])&set(row['semantic_ids']))/max(1,len(set(previous['semantic_ids'])|set(row['semantic_ids']))),rank_distance=statistics.mean(abs(previous['orders']['current'].index(x)-row['orders']['current'].index(x)) for x in shared) if shared else None))
                    previous=row;queued+=1
                # Freeze a real mid-conversation batch, then leave later messages live.
                if turn['turn_id']=='a14':consolidate(session);queued=0
            consolidate(session)
            final=records(tenant,user);evaluation=evaluate_state(final,truth,sources)
            later=[]
            for q in truth['later_queries']:
                hook.on_chat(tenant,tid+'-later-rep'+str(rep),q['text'],ChatSenderRole.USER,user)
                out=hook.cog.merge_entities(captured['recent'],captured['raw'],captured['session'])
                labels=[entity_key(dict(entity_name=e['entity_name'],payload=e),truth) for e in out if e['entity_domain']!='identity_and_demographics']
                later.append(dict(text=q['text'],required=q['required'],available=[k for k in q['required'] if k in labels],ordered=labels))
            run=dict(trace_id=tid,repetition=rep,trace_class=trace['trace_class'],tenant=tenant,user=user,turns=trace['turns'],hidden_truth=truth,retrieval=retrieval,turn_results=turn_results,filler=filler,batches=batch_results,final_records=[{k:v for k,v in r.items() if k!='native_id'} for r in final],state_score=evaluation,later=later,
                terminology_metrics=getattr(workflow,'last_terminology_metrics',None))
            runs.append(run)
            (OUTPUT/'progress.json').write_text(json.dumps(runs,default=str,indent=2),encoding='utf-8')
            print('Completed',current['trace'],'entities',len(final),'checklist',round(evaluation['checklist_recall'],3),flush=True)
        # Narrow optional-source faults, not global infrastructure disruption.
        current['trace']='fault-probes';tenant=prefix+'-fault';user='fault-user';session='fault-session';hook=ChatHookWorkflow()
        raw_method=db.get_raw_semantic_context
        db.get_raw_semantic_context=lambda *a,**kw:(_ for _ in ()).throw(TomteSessionStateUnavailable('Synthetic unavailable'))
        try:
            result=hook.on_chat(tenant,session,'yeah',ChatSenderRole.USER,user);reliability.append(dict(probe='raw_optional_unavailable',passed=not bool(result.get('error'))))
        finally:db.get_raw_semantic_context=raw_method
        redis_method=cache.get_session_entity_ids
        cache.get_session_entity_ids=lambda *a,**kw:(_ for _ in ()).throw(TomteSessionStateUnavailable('Synthetic unavailable'))
        try:
            result=hook.on_chat(tenant,session,'yeah',ChatSenderRole.USER,user);reliability.append(dict(probe='optional_redis_mru_unavailable',passed=not bool(result.get('error'))))
        finally:cache.get_session_entity_ids=redis_method
    finally:
        # Normal short transactions were used throughout; only owned fixtures removed.
        with db.transaction():
            for table in ('tomte_memory_engine','tomte_processed_consolidations','tomte_user_registry'):
                db.conn.execute('DELETE FROM '+table+' WHERE tenant_id LIKE %s',(prefix+'%',))
            if anchors:
                db.conn.execute('DELETE FROM tomte_keyword_audit_log WHERE dictionary_id=ANY(%s::uuid[])',(list(set(anchors)),))
                db.conn.execute('DELETE FROM tomte_semantic_anchors WHERE dictionary_id=ANY(%s::uuid[])',(list(set(anchors)),))
        keys=list(cache.client.scan_iter(match='*'+prefix+'*'))
        if keys:cache.client.delete(*keys)
        active=get_section_key_active_sessions();refs=[r for r,_ in cache.client.zscan_iter(active) if r.startswith(prefix)]
        if refs:cache.client.zrem(active,*refs)
        after=state(observer,raw);cleanup=before==after
        (OUTPUT/'cleanup.json').write_text(json.dumps(dict(before=before,after=after,verified=cleanup)),encoding='utf-8')
        observer.close();raw.close();cache.client.close();db.close();threading.Thread.start=old_start;redis_provider.time=old_clock
        httpx.AsyncClient.send=send;openai_provider._debug_metadata=diagnostic
        (OUTPUT/'model_calls.json').write_text(json.dumps(model_calls,default=str,indent=2),encoding='utf-8')
        (OUTPUT/'api_usage.json').write_text(json.dumps(dict(counts=dict(api),requests=requests,retries=retry_diagnostics),indent=2),encoding='utf-8')
    assert cleanup,'Cleanup digest mismatch'
    report=dict(runs=runs,reliability=reliability,cleanup_verified=cleanup,models=settings.openai.model_dump(),api=dict(api),requests=requests,retries=retry_diagnostics,
        limitations=['Deterministic fact checklist is lexical, not semantic precision; unmatched facts need human review.',
          'Two-day session gaps simulated using queue clock and only synthetic row recency; no wall-clock idle wait.',
          'Snapshots/cycles manually invoked; daemon scheduler and lock contention not exercised.',
          'Single runtime per base trace; two critical traces repeated three times.',
          'No new race/load framework; successful snapshot replay tested, full suite covers concurrency/rollback.',
          'Observed raw scores unavailable for recent/session-only candidates; additive scores use zero.',
          'Generated noisy traces are frozen seeded templates, not independently generated human transcripts.'])
    rows=[r for run in runs for r in run['retrieval']]
    ranking={}
    for policy in POLICIES:
        required=sum(len(r['required']) for r in rows)
        ranks=[r['orders'][policy].index(k)+1 for r in rows for k in r['required'] if k in r['orders'][policy]]
        s=dict(required=required,availability=len(ranks)/required if required else None,mean_rank=statistics.mean(ranks) if ranks else None)
        for k in (1,3,5):
            s['recall'+str(k)]=sum(len(set(r['orders'][policy][:k])&set(r['required'])) for r in rows)/required if required else None
            count=sum(len(r['orders'][policy][:k]) for r in rows)
            s['precision'+str(k)]=sum(len(set(r['orders'][policy][:k])&set(r['acceptable'])) for r in rows)/count if count else None
        ranking[policy]=s
    report['ranking']=ranking
    (OUTPUT/'report.json').write_text(json.dumps(report,default=str,indent=2),encoding='utf-8')
    with (OUTPUT/'ranking.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['policy',*next(iter(ranking.values()))]);writer.writeheader()
        for p,s in ranking.items():writer.writerow(dict(policy=p,**s))
    body='<h1>Real-world conversation evaluation</h1><p>Evaluation only. Strict checklist scores require review; no production changes.</p>'
    body+='<h2>Ranking</h2><pre>'+html.escape(json.dumps(ranking,indent=2))+'</pre>'
    for run in runs:
        body+='<details><summary>'+html.escape(run['trace_id']+' repetition '+str(run['repetition']))+'</summary><pre>'+html.escape(json.dumps(run,default=str,indent=2))+'</pre></details>'
    body+='<h2>Limits</h2><pre>'+html.escape(json.dumps(report['limitations'],indent=2))+'</pre>'
    (OUTPUT/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Tomte long traces</title><style>body{font:15px system-ui;margin:32px;max-width:1200px}pre{white-space:pre-wrap}summary{cursor:pointer;padding:12px}</style>'+body,encoding='utf-8')
    print('Evaluation complete; cleanup verified; ranking',json.dumps(ranking),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as e:
        print('Evaluation stopped:',type(e).__name__,'details withheld',flush=True)
        import traceback
        print([(Path(f.filename).name,f.lineno,f.name) for f in traceback.extract_tb(e.__traceback__)],flush=True)
        raise SystemExit(1)
