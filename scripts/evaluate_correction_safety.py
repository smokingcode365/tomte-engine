"""Bounded real-model correction evaluation; unique local fixtures, no daemon."""
import json,logging,os,sys,time
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
import httpx,psycopg,redis
from dotenv import load_dotenv
from evaluate_memory_retrieval import state
from tomte.engine import Tomte
from tomte.engine.core.config import settings
from tomte.engine.core.schema import Entity,Fact
from tomte.engine.core.extraction import ExtractionMemorySchema,resolve_extraction
from tomte.engine.core.assertions import adapt_assertions,project_facts
from tomte.engine.core.correction_safety import UnsupportedCorrection
from tomte.engine.dictionary import service
from tomte.engine.providers import openai_provider
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.providers.redis_provider import RedisCacheProvider
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from tomte.engine.utils.keys import get_section_key_active_sessions
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'log/correction_safety'


def main():
    assert Path.cwd().resolve()==ROOT
    load_dotenv('.env',override=False)
    for n in ('POSTGRES_URL','REDIS_URL'):
        if urlsplit(os.environ.get(n,'')).hostname not in ('localhost','127.0.0.1','::1'):raise RuntimeError('Local services required')
    for n in ('tomte.engine','httpx','httpcore','openai'):logging.getLogger(n).setLevel(logging.CRITICAL)
    OUT.mkdir(parents=True,exist_ok=True)
    fixture=json.loads((ROOT/'evals/correction_safety/honda_tesla_failure.json').read_text(encoding='utf-8'))
    cases=json.loads((ROOT/'evals/correction_safety/cases.json').read_text(encoding='utf-8'))
    calls=[];requests=[];retries=[];current={};send=httpx.AsyncClient.send;diag=openai_provider._debug_metadata
    async def record(client,request,*a,**kw):
        response=await send(client,request,*a,**kw)
        requests.append(dict(case=current.get('id'),model=json.loads(request.content).get('model'),status=response.status_code,usage=response.json().get('usage',{})))
        return response
    httpx.AsyncClient.send=record
    openai_provider._debug_metadata=lambda data:retries.append(dict(case=current.get('id'),**data))
    live=openai_provider.OpenAILLMProvider(os.environ['OPENAI_API_KEY'])
    class Recorder:
        def get_vector_dimension(self):return live.get_vector_dimension()
        def parse_structured(self,system_prompt,user_text,schema,**kw):
            c=dict(case=current.get('id'),stage=schema.__name__,input=json.loads(user_text));calls.append(c)
            try:
                result=live.parse_structured(system_prompt,user_text,schema,**kw);c['output']=result.model_dump(mode='json');return result
            except Exception as e:c['error_type']=type(e).__name__;raise
    db=PostgresProvider(os.environ['POSTGRES_URL']);cache=RedisCacheProvider(os.environ['REDIS_URL'])
    observer=psycopg.connect(os.environ['POSTGRES_URL'],autocommit=True,options='-c default_transaction_read_only=on');raw=redis.Redis.from_url(os.environ['REDIS_URL'])
    before=state(observer,raw);prefix='correction-eval-'+uuid4().hex;results=[]
    vectors=json.loads((ROOT/'log/real_world/embeddings.json').read_text(encoding='utf-8'))
    # Reuse actual real-model embeddings for fixture-only candidate search.
    vector=next(iter(vectors.values()))
    def seed(user,name,eid,domain,kind,facts,aliases=()):
        payload=Entity(entity_id=eid,entity_name=name,entity_domain=domain,entity_type=kind,aliases=list(aliases),facts=[Fact(**{'confidence':1,**f}) for f in facts]).model_dump(mode='json')
        payload['facts']=adapt_assertions(payload['facts'],prefix+user+eid)
        assert db.commit_graph_updates([dict(mode='INSERT',tenant_id=prefix,user_id=user,**{k:payload[k] for k in ('entity_id','entity_name','entity_domain','entity_type')},payload=payload,embedding=vector)])
    def rows(user):return db.conn.execute('SELECT id::text,id AS native_id,tenant_id,user_id,entity_id,entity_name,entity_domain,entity_type,payload FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s ORDER BY entity_id',(prefix,user)).fetchall()
    try:
        Tomte._db=Tomte._cache=Tomte._llm=Tomte._worker=None;service._dictionary_service_instance=None
        Tomte.init(db,cache,Recorder(),start_daemon_worker=False)
        for case in cases:
            current['id']=user=case['case_id']
            for eid,name,old in [('target',case['target'],case['old']),('other',case['other'],'2024' if case['attribute'].endswith('year') else 'other value')]:
                seed(user,name,eid,case['domain'],case['kind'],[dict(attribute_name=case['attribute'],value=old,source_chat_id='original',last_mentioned_at='2026-10-09T00:00:00Z')])
            existing=rows(user)
            item=dict(item_id='source-'+user,role='user',content=case['text'],timestamp=time.time())
            data=dict(CURRENT_BATCH=[item],RECENT_CONTEXT_REFERENCE_ONLY=[dict(role='user',content=case['reference'])])
            # Deliberately inject the erroneous incoming assignment to test planner+gate,
            # not extraction accuracy. Neither model receives expected_correction.
            incoming=dict(entity_id='temporary',entity_name=case['target'],entity_domain=case['domain'],entity_type=case['kind'],aliases=[],facts=[dict(attribute_name=case['attribute'],value=case['new'],source_chat_id=item['item_id'],confidence=.9,status='active')])
            op=ConsolidationWorkflow._prepare_operation(prefix,user,incoming,existing[1] if existing[1]['entity_id']=='target' else existing[0])
            workflow=ConsolidationWorkflow();blocked=False;error=None
            try:
                prepared=workflow._plan_fact_mutations([op],data,'batch-'+user,existing)
                for operation in prepared:operation['embedding']=vector
                workflow._commit_refined_operations(prepared,prefix,user,'batch-'+user)
            except UnsupportedCorrection:blocked=True
            except Exception as e:error=type(e).__name__
            final=next(r for r in rows(user) if r['entity_id']=='target')
            active=project_facts(final['payload']['facts']);old=next(f for f in final['payload']['facts'] if f['value']==case['old'])
            corrected=old['assertion_state']=='CORRECTED_INVALID' and any(f['value']==case['new'] for f in active)
            invalidated=old['assertion_state']!='ACTIVE'
            results.append(dict(case_id=user,expected_correction=case['expected_correction'],blocked=blocked,error=error,corrected=corrected,old_invalidated=invalidated,plan=getattr(workflow,'last_mutation_plan',None),final=final['payload']))
            print(user,'corrected',corrected,'blocked',blocked,'error',error,flush=True)
        for rep in range(1,11):
            current['id']=user='failure-repeat-'+str(rep)
            call=fixture['stages']['MutationPlanResponse']
            for target in call['input']['TARGETS']:
                identity=target['identity'];record=next(r for r in fixture['final_records'] if r['entity_id']==identity['entity_id']);facts=[]
                for f in target['assertions']:
                    original=deepcopy(next(x for x in record['payload']['facts'] if x['fact_id']==f['fact_id']));original.update(f);facts.append(original)
                seed(user,identity['entity_name'],identity['entity_id'],identity['entity_domain'],identity['entity_type'],facts,record['payload'].get('aliases',[]))
            existing=rows(user);data=deepcopy(fixture['stages']['ExtractionMemorySchema']['input'])
            extracted=Tomte.get_llm().parse_structured(None,json.dumps(data),ExtractionMemorySchema)
            normalized=resolve_extraction(extracted,data['CURRENT_BATCH']);workflow=ConsolidationWorkflow();entities=workflow._flatten_schema(normalized)
            op=workflow._reconcile_with_golden_records(prefix,user,entities,existing,reference_context=data['RECENT_CONTEXT_REFERENCE_ONLY'],current_batch=data['CURRENT_BATCH'])
            misassigned=any(e['entity_name']=='Honda Civic' and any(f['value']=='2024' for f in e['facts']) for e in entities)
            blocked=False;error=None
            try:
                prepared=workflow._plan_fact_mutations(op,data,'batch-'+user,existing)
                for operation in prepared:operation['embedding']=vector
                workflow._commit_refined_operations(prepared,prefix,user,'batch-'+user)
            except UnsupportedCorrection:blocked=True
            except Exception as e:error=type(e).__name__
            final=next(r for r in rows(user) if r['entity_name']=='Honda Civic')
            active_years=[f['value'] for f in project_facts(final['payload']['facts']) if f['attribute_name']=='purchase_year']
            corrupted='2024' in active_years or '2021' not in active_years
            results.append(dict(case_id=user,repetition=rep,extraction_misassigned=misassigned,blocked=blocked,error=error,corrupted=corrupted,old_2021_preserved='2021' in active_years,plan=getattr(workflow,'last_mutation_plan',None),final=final['payload']))
            print(user,'misassigned',misassigned,'corrupted',corrupted,'blocked',blocked,flush=True)
    finally:
        with db.transaction():
            for table in ('tomte_memory_engine','tomte_processed_consolidations','tomte_user_registry'):db.conn.execute('DELETE FROM '+table+' WHERE tenant_id=%s',(prefix,))
        keys=list(cache.client.scan_iter(match='*'+prefix+'*'))
        if keys:cache.client.delete(*keys)
        active=get_section_key_active_sessions();refs=[ref for ref,_ in cache.client.zscan_iter(active) if ref.startswith(prefix)]
        if refs:cache.client.zrem(active,*refs)
        after=state(observer,raw);assert before==after,'Cleanup mismatch'
        (OUT/'cleanup.json').write_text(json.dumps(dict(before=before,after=after,verified=before==after)),encoding='utf-8')
        (OUT/'results.json').write_text(json.dumps(results,default=str,indent=2),encoding='utf-8')
        (OUT/'model_calls.json').write_text(json.dumps(calls,default=str,indent=2),encoding='utf-8')
        (OUT/'usage.json').write_text(json.dumps(dict(requests=requests,retries=retries),indent=2),encoding='utf-8')
        httpx.AsyncClient.send=send;openai_provider._debug_metadata=diag;observer.close();raw.close();cache.client.close();db.close()

if __name__=='__main__':
    try:main()
    except Exception as e:print('Evaluation stopped:',type(e).__name__,'details withheld');raise SystemExit(1)
