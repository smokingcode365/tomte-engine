"""Small real-Redis snapshot/ack probe. Synthetic namespace only; no model calls."""
import json,os,time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4
from dotenv import load_dotenv
import redis
from tomte.engine.providers.redis_provider import RedisCacheProvider
from tomte.engine.utils.keys import generate_consolidation_key,get_section_key_active_sessions,generate_session_ref
ROOT=Path(__file__).resolve().parents[1]


def main():
    assert Path.cwd().resolve()==ROOT
    load_dotenv('.env',override=False)
    url=os.environ.get('REDIS_URL','')
    if urlsplit(url).hostname not in ('localhost','127.0.0.1','::1'):raise RuntimeError('Local Redis required')
    cache=RedisCacheProvider(url);before=set(cache.client.scan_iter());tag='world-queue-'+uuid4().hex
    tenant,user,session=tag,'synthetic-user','synthetic-session';active=get_section_key_active_sessions();ref=generate_session_ref(tenant,session,user)
    report={}
    try:
        for _ in range(2):cache.enqueue_for_consolidation(tenant,user,session,'yeah','user',vector=[1.0]+[0.0]*1535)
        cache.client.zadd(active,{ref:time.time()-600})
        idle=cache.get_next_idle_session_4_consolidation(idle_seconds=300)
        report['idle_selection']=tuple(idle or ())==(tenant,user,session)
        report['idle_return_type']=type(idle).__name__
        assert report['idle_selection']
        first=cache.pop_session_chat_log_4_consolidation(tenant,user,session)
        assert len(first)==2 and len({x['item_id'] for x in first})==2
        report['duplicate_delivery_distinct_ids']=True
        assert cache.pop_session_chat_log_4_consolidation(tenant,user,session)==first
        report['frozen_retry_stable']=True
        cache.enqueue_for_consolidation(tenant,user,session,'A later synthetic message.','user',vector=[1.0]+[0.0]*1535)
        assert cache.pop_session_chat_log_4_consolidation(tenant,user,session)==first
        assert cache.clear_session_4_consolidation(tenant,user,session,first)
        second=cache.pop_session_chat_log_4_consolidation(tenant,user,session)
        assert len(second)==1 and second[0]['content']=='A later synthetic message.'
        assert not cache.clear_session_4_consolidation(tenant,user,session,first)
        assert cache.clear_session_4_consolidation(tenant,user,session,second)
        report.update(late_arrival_preserved=True,stale_ack_rejected=True,both_batches_acknowledged=True,
            caveat='Sequential interleaving simulates arrival during processing; no simultaneous clients or model processing.')
    finally:
        keys=list(cache.client.scan_iter(match='*'+tag+'*'))
        if keys:cache.client.delete(*keys)
        cache.client.zrem(active,ref)
        after=set(cache.client.scan_iter());report['cleanup_key_count_and_digest_equivalent']=before==after
        cache.client.close()
        out=ROOT/'log/real_world/queue_probe.json';out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert report['cleanup_key_count_and_digest_equivalent']
    print(json.dumps(report))

if __name__=='__main__':
    try:main()
    except Exception as e:print('Queue probe failed:',type(e).__name__,'details withheld');raise SystemExit(1)
