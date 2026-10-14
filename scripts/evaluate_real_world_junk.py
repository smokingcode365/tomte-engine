"""Bounded real-provider junk/idle probe. Calls job synchronously, never a daemon."""
import hashlib,json,logging,os,time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
from dotenv import load_dotenv
import httpx,psycopg,redis
from evaluate_memory_retrieval import state
from tomte.engine import Tomte
from tomte.engine.core.config import settings
from tomte.engine.dictionary import service
from tomte.engine.providers.openai_provider import OpenAILLMProvider
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.providers.redis_provider import RedisCacheProvider
from tomte.engine.utils.keys import get_section_key_active_sessions,generate_session_ref,generate_consolidation_batch_key
from tomte.engine.workers.dictionary_maintenance_worker import run_consolidation_job
ROOT=Path(__file__).resolve().parents[1]


def main():
    assert Path.cwd().resolve()==ROOT;load_dotenv('.env',override=False)
    if any(urlsplit(os.environ.get(n,'')).hostname not in ('localhost','127.0.0.1','::1') for n in ('POSTGRES_URL','REDIS_URL')):raise RuntimeError('Local services required')
    for n in ('tomte.engine','openai','httpx','httpcore'):logging.getLogger(n).setLevel(logging.CRITICAL)
    db=PostgresProvider(os.environ['POSTGRES_URL']);cache=RedisCacheProvider(os.environ['REDIS_URL']);llm=OpenAILLMProvider(os.environ['OPENAI_API_KEY'])
    observer=psycopg.connect(os.environ['POSTGRES_URL'],autocommit=True,options='-c default_transaction_read_only=on');raw=redis.Redis.from_url(os.environ['REDIS_URL']);before=state(observer,raw)
    if before['redis']['count'] or before['postgres']['tomte_semantic_anchors']['count']:raise RuntimeError('Otherwise idle empty local fixtures required; no existing data removed')
    tag='world-junk-'+uuid4().hex;tenant,user,session=tag,'synthetic-user','synthetic-session';anchors=[];answers=[];polls=[];requests=[]
    create=db.create_dictionary_anchor
    def track(v):
        a=create(v);anchors.append(str(a));return a
    db.create_dictionary_anchor=track
    send=httpx.AsyncClient.send
    async def record(client,request,*a,**kw):
        response=await send(client,request,*a,**kw)
        requests.append(dict(model=json.loads(request.content).get('model'),status=response.status_code,usage=response.json().get('usage',{})))
        return response
    httpx.AsyncClient.send=record
    junk=llm.is_junk
    def observe(text):
        answer=junk(text);answers.append(answer);return answer
    llm.is_junk=observe
    result={}
    try:
        Tomte._db=Tomte._cache=Tomte._llm=Tomte._worker=None;service._dictionary_service_instance=None
        Tomte.init(db,cache,llm,start_daemon_worker=False)
        vectors=json.loads((ROOT/'log/real_world/embeddings.json').read_text())
        for text in ('yeah','okay'):
            key=hashlib.sha256(json.dumps([settings.openai.embed_model,1536,'query',text]).encode()).hexdigest()
            cache.enqueue_for_consolidation(tenant,user,session,text,'user',vector=vectors[key])
        cache.client.zadd(get_section_key_active_sessions(),{generate_session_ref(tenant,session,user):time.time()-600})
        select=cache.get_next_idle_session_4_consolidation
        def bounded(*a,**kw):
            if len(polls)>=2:return None
            target=select(*a,**kw)
            if target is not None:assert tuple(target)==(tenant,user,session)
            polls.append(tuple(target) if target else None);return target
        cache.get_next_idle_session_4_consolidation=bounded
        run_consolidation_job()
        frozen=cache.pop_session_chat_log_4_consolidation(tenant,user,session)
        result=dict(junk_answers=answers,polls=polls,requests=requests,frozen_messages_remaining=len(frozen),
            repeated_junk_batch=answers==[True,True] and polls==[(tenant,user,session)]*2 and len(frozen)==2,
            note='Stopped after two selections using a test hook; no daemon or infinite loop was started.')
    finally:
        with db.transaction():
            for table in ('tomte_memory_engine','tomte_processed_consolidations','tomte_user_registry'):db.conn.execute('DELETE FROM '+table+' WHERE tenant_id=%s',(tenant,))
            if anchors:db.conn.execute('DELETE FROM tomte_semantic_anchors WHERE dictionary_id=ANY(%s::uuid[])',(anchors,))
        keys=list(cache.client.scan_iter(match='*'+tag+'*'))
        if keys:cache.client.delete(*keys)
        cache.client.zrem(get_section_key_active_sessions(),generate_session_ref(tenant,session,user))
        after=state(observer,raw);result['cleanup_verified']=before==after
        observer.close();raw.close();cache.client.close();db.close();httpx.AsyncClient.send=send
        (ROOT/'log/real_world/junk_probe.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    assert result['cleanup_verified'];print(json.dumps(result))

if __name__=='__main__':
    try:main()
    except Exception as e:print('Junk probe failed:',type(e).__name__,'details withheld');raise SystemExit(1)
