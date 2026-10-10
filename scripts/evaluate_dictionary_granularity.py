"""Evaluation-only semantic dictionary granularity study.

Run from tomte-engine: uv run python scripts/evaluate_dictionary_granularity.py
Uses synthetic gold raw extraction outputs, configured embeddings, and temporary
PostgreSQL dictionary rows. Never calls extraction/identity/mutation/normalization
models or modifies production policy. Matching representations are used on both
sides except the explicit current-summary/chat baseline. Local cached embeddings
make the same study repeatable without more API calls.
"""
import hashlib
import json
import logging
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import numpy as np
import redis
from dotenv import load_dotenv

from evaluate_dictionary_representations import dictionary_state, RollbackEvaluation
from tomte.engine.core.config import settings
from tomte.engine.dictionary.service import DictionaryService
from tomte.engine.llm.embedder import get_entity_summary
from tomte.engine.providers.openai_provider import OpenAILLMProvider
from tomte.engine.providers.postgres_provider import PostgresProvider

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'scripts'/'dictionary_granularity_cases.json'
OUTPUT=ROOT/'log'/'dictionary_granularity_evaluation.json'
CACHE=ROOT/'log'/'dictionary_granularity_embeddings.json'
VEHICLE_FACTS=[('ownership_status','owned'),('nickname','Little Dog'),('purchase_year','2024'),('mileage','30000'),
 ('charging_location','home'),('insurance_provider','Example Insurance'),('tire_type','winter'),('service_date','2026-10-01'),
 ('software_version','15.2'),('battery_health','95 percent'),('parking_location','garage'),('preferred_charger','home charger'),
 ('registration_expiry','2027-01-01'),('wheel_size','19 inches'),('paint_color','red')]
OTHER_FACTS=[('hobby','basketball'),('favorite_food','pasta'),('preferred_tool','laptop'),('working_hours','daytime'),
 ('favorite_book','Example Book'),('travel_goal','Lisbon'),('pet_preference','dogs'),('favorite_music','jazz'),
 ('meeting_day','Monday'),('communication_style','concise'),('fitness_activity','walking'),('favorite_season','spring'),
 ('preferred_drink','tea'),('learning_goal','design'),('weekend_activity','hiking')]


def facts(record, raw_attribute=None, value=None, size=10):
    target=record['candidate'];background=VEHICLE_FACTS if record['entity_type']=='vehicle' else OTHER_FACTS
    return [(raw_attribute or target, record['value'] if value is None else value)]+[
        (k,v) for k,v in background if k != target][:size-1]


def whole(record, raw_type=None, raw_attribute=None, value=None, size=10):
    return json.dumps({'domain':record['domain'],'entity_name':record['entity_name'],
        'entity_type':raw_type or record['entity_type'],
        'facts':[{'attribute':k,'value':v} for k,v in facts(record,raw_attribute,value,size)]},ensure_ascii=False)


def current_summary(record):
    return get_entity_summary({'entity_domain':record['domain'],'entity_name':record['entity_name'],
        'entity_type':record['entity_type'],'aliases':[],
        'facts':[{'attribute_name':k,'value':v,'importance':3,'assertion_state':'ACTIVE'} for k,v in facts(record)]})


def fact_text(record, attribute, entity_type, value=None):
    text=f"domain: {record['domain']}\nentity_type: {entity_type}\nattribute: {attribute}"
    return text if value is None else text+f"\nvalue: {value}"


def entity_text(record, entity_type, with_name=False):
    text=f"domain: {record['domain']}\nentity_type: {entity_type}"
    return text+f"\nentity_name: {record['entity_name']}" if with_name else text


def metrics(scores, queries, candidate_ids, threshold):
    ranks=[];correct=[];wrong=[];negative=[];details=[]
    for row,q in zip(scores,queries):
        index=candidate_ids.index(q['target']);order=np.argsort(-row,kind='stable').tolist()
        rank=order.index(index)+1;best=max(s for j,s in enumerate(row) if j!=index)
        ranks.append(rank);correct.append(float(row[index]));wrong.append(float(best))
        hard=[float(row[candidate_ids.index(k)]) for k in q['hard_negatives']]
        negative.extend(hard)
        details.append({'id':q['id'],'target':q['target'],'raw_term':q['raw_term'],'value':q.get('value'),
            'rank':rank,'correct_similarity':float(row[index]),'best_wrong_similarity':float(best),
            'margin':float(row[index]-best),'top3':[{'candidate':candidate_ids[j],'similarity':float(row[j])} for j in order[:3]],
            'hard_negatives':dict(zip(q['hard_negatives'],hard))})
    return {'queries':len(queries),'top1_accuracy':sum(r==1 for r in ranks)/len(ranks),
        'top3_accuracy':sum(r<=3 for r in ranks)/len(ranks),
        'threshold_recall':sum(s>threshold for s in correct)/len(correct),
        'mean_correct_similarity':float(np.mean(correct)),'minimum_correct_similarity':min(correct),
        'mean_best_wrong_similarity':float(np.mean(wrong)),'mean_margin':float(np.mean(np.array(correct)-wrong)),
        'hard_negative_pairs':len(negative),'hard_negative_false_positive_rate':sum(s>threshold for s in negative)/len(negative),
        'details':details}


def main():
    if Path.cwd().resolve()!=ROOT:raise RuntimeError('Run from tomte-engine')
    load_dotenv(ROOT/'.env',override=False)
    for key in ('OPENAI_API_KEY','POSTGRES_URL','REDIS_URL'):
        if not os.environ.get(key):raise RuntimeError('Required environment variable is missing')
    if any(urlsplit(os.environ[k]).hostname not in ('localhost','127.0.0.1','::1') for k in ('POSTGRES_URL','REDIS_URL')):
        raise RuntimeError('Only local infrastructure is allowed')
    for name in ('openai','httpx','httpcore'):logging.getLogger(name).setLevel(logging.CRITICAL)
    dataset=json.loads(DATA.read_text());candidates=dataset['attribute_candidates'];entity_candidates=dataset['entity_candidates']
    attribute_ids=[c['candidate'] for c in candidates];entity_ids=[c['candidate'] for c in entity_candidates]
    assert len(attribute_ids)==len(set(attribute_ids)) and len(entity_ids)==len(set(entity_ids))
    texts={}
    def add(text):
        if text not in texts:texts[text]=len(texts)
        return texts[text]
    strategies=['current_summary_chat','chat_chat','whole_entity','fact_with_value','fact_without_value','raw_attribute']
    anchors={s:[] for s in strategies};query_texts={s:[] for s in strategies};queries=[]
    for c in candidates:
        for s,t in zip(strategies,[current_summary(c),c['originating_chat'],whole(c),
            fact_text(c,c['candidate'],c['entity_type'],c['value']),fact_text(c,c['candidate'],c['entity_type']),c['candidate']]):
            anchors[s].append(add(t))
    for f in dataset['attribute_families']:
        for i,q in enumerate(f['queries']):
            queries.append({'id':f"{f['candidate']}-{i+1}",'target':f['candidate'],'raw_term':q['raw_attribute'],
                            'value':q['value'],'hard_negatives':f['hard_negatives']})
            raw=q['raw_attribute'];v=q['value'];typ=f['raw_entity_type']
            for s,t in zip(strategies,[q['followup'],q['followup'],whole(f,typ,raw,v),
                fact_text(f,raw,typ,v),fact_text(f,raw,typ),raw]):query_texts[s].append(add(t))
    assert len(queries)>=75 and sum(len(q['hard_negatives']) for q in queries)>=75
    ent_strategies=['current_summary_chat','chat_chat','whole_entity','compact_type','compact_type_name']
    ent_anchors={s:[] for s in ent_strategies};ent_queries_text={s:[] for s in ent_strategies};ent_queries=[]
    for e in entity_candidates:
        record={**e,'entity_type':e['candidate'],'value':'synthetic','candidate':'example_property'}
        for s,t in zip(ent_strategies,[current_summary(record),f"This {e['entity_name']} is a {e['candidate']}.",
            whole(record),entity_text(e,e['candidate']),entity_text(e,e['candidate'],True)]):ent_anchors[s].append(add(t))
        for i,raw in enumerate(e['raw_types']):
            ent_queries.append({'id':f"{e['candidate']}-{i+1}",'target':e['candidate'],'raw_term':raw,'hard_negatives':e['hard_negatives']})
            chat=f"My {e['entity_name']} is a {raw}."
            for s,t in zip(ent_strategies,[chat,chat,whole(record,raw_type=raw),entity_text(e,raw),entity_text(e,raw,True)]):
                ent_queries_text[s].append(add(t))
    assert len(ent_queries)>=20
    busy=[];family_by={f['candidate']:f for f in dataset['attribute_families']}
    for target in ['paint_color','ownership_status']:
        f=family_by[target];raw='exterior_color' if target=='paint_color' else 'ownership'
        for size in [1,2,4,6,8,10,15]:
            busy.append({'target':target,'facts':size,'whole_query':add(whole(f,f['raw_entity_type'],raw,size=size)),
                         'fact_query':add(fact_text(f,raw,f['raw_entity_type'],f['value']))})
    paint=family_by['paint_color'];topics=['The outside of my Tesla is red.','My son joined basketball.',
        'I am learning Python.','I bought an aquarium for my fish.','I booked a trip to Lisbon.']
    noise=[{'topics':n,'chat_query':add(' '.join(topics[:n])),'fact_query':add(fact_text(paint,'exterior_color','car','red'))} for n in [1,2,3,5]]
    controls=[]
    for target,raw,value in [('ownership_status','status','sold'),('marital_status','status','married'),('employment_status','status','employed'),
        ('programming_skill_level','level','advanced'),('vehicle_type','type','SUV'),('legal_name','name','Casey Example'),
        ('current_home_address','location','123 Example Street'),('paint_color','color','red')]:
        c=next(c for c in candidates if c['candidate']==target)
        typ=c.get('raw_entity_type',c['entity_type'])
        controls.append({'target':target,'raw':raw,'value':value,'with':add(fact_text(c,raw,typ,value)),
                         'without':add(fact_text(c,raw,typ))})
    unseen_queries=[]
    unseen_names={'vehicle':'Toyota Corolla','employment':'Research engineering role','residence':'789 New Avenue',
        'skill':'Python scripting practice','person':'Jordan Sample','organization':'Sample Industries','pet':'Nova','project':'Budget tracker'}
    for e in entity_candidates:
        for raw in e['raw_types']:
            unseen_queries.append(add(entity_text({**e,'entity_name':unseen_names[e['candidate']]},raw,True)))
    heldout_queries=[];heldout_with=[];heldout_without=[];seen_comparison=[]
    new_values={'paint_color':'beige','ownership_status':'not owned','job_title':'Data Scientist',
        'current_home_address':'789 Sample Avenue','programming_language':'Rust','nickname':'Red Fox',
        'purchase_year':'2021','service_date':'2027-06-10','start_date':'2025-01-01','marital_status':'single',
        'programming_skill_level':'beginner','vehicle_type':'sedan','legal_name':'Jordan Sample',
        'current_location':'Lisbon','interior_color':'red'}
    for family_index,f in enumerate(dataset['attribute_families']):
        for query_index in (0,5):
            q=f['queries'][query_index];raw=q['raw_attribute'];v=new_values[f['candidate']]
            heldout_queries.append({'id':f"heldout-{f['candidate']}-{query_index}",'target':f['candidate'],
                'raw_term':raw,'value':v,'hard_negatives':f['hard_negatives']})
            heldout_with.append(add(fact_text(f,raw,f['raw_entity_type'],v)))
            heldout_without.append(add(fact_text(f,raw,f['raw_entity_type'])))
            seen_comparison.append(family_index*6+query_index)
    payloads=list(texts);signature=hashlib.sha256(json.dumps([settings.openai.embed_model,settings.openai.vector_dimension,payloads]).encode()).hexdigest()
    usage={'api_requests':0,'batch_calls':0,'texts':len(payloads),'input_tokens':0,'cached_texts':0}
    stored=json.loads(CACHE.read_text()) if CACHE.exists() else {}
    cached=stored.get('vectors',[])
    prefix_signature=hashlib.sha256(json.dumps([settings.openai.embed_model,settings.openai.vector_dimension,payloads[:len(cached)]]).encode()).hexdigest()
    if prefix_signature!=stored.get('signature'):cached=[]
    vectors=cached.copy();usage['cached_texts']=len(cached)
    if len(cached)<len(payloads):
        original_send=httpx.AsyncClient.send
        async def embeddings_only(self,request,*args,**kwargs):
            if not request.url.path.endswith('/embeddings'):raise RuntimeError('Non-embedding model calls are prohibited')
            usage['api_requests']+=1
            return await original_send(self,request,*args,**kwargs)
        httpx.AsyncClient.send=embeddings_only
        try:
            provider=OpenAILLMProvider(os.environ['OPENAI_API_KEY'])
            for start in range(len(cached),len(payloads),200):
                response=provider.embedder.embed_documents_sync(payloads[start:start+200]);usage['batch_calls']+=1
                usage['input_tokens']+=response.usage.input_tokens;vectors.extend([list(v) for v in response.embeddings])
            previous=stored.get('study_totals',stored.get('usage',{})) if cached else {}
            totals={'api_requests':previous.get('api_requests',0)+usage['api_requests'],
                'batch_calls':previous.get('batch_calls',0)+usage['batch_calls'],
                'input_tokens':previous.get('input_tokens',0)+usage['input_tokens'],'texts':len(payloads)}
            CACHE.parent.mkdir(exist_ok=True);CACHE.write_text(json.dumps({'signature':signature,'vectors':vectors,'study_totals':totals}),encoding='utf-8')
        finally:httpx.AsyncClient.send=original_send
    vectors=np.array(vectors)
    usage['study_totals']=totals if usage['api_requests'] else stored.get('study_totals',stored.get('usage',{}))
    assert vectors.shape==(len(payloads),settings.openai.vector_dimension)
    normalized=vectors/np.linalg.norm(vectors,axis=1,keepdims=True)
    def score(q,a):return normalized[q]@normalized[a].T
    threshold=settings.llm_threshold_low
    report={'dataset':{'attribute_queries':len(queries),'attribute_candidates':len(candidates),'hard_negative_pairs':sum(len(q['hard_negatives']) for q in queries),
        'entity_queries':len(ent_queries),'entity_candidates':len(entity_candidates),'entity_hard_negative_pairs':sum(len(q['hard_negatives']) for q in ent_queries)},
        'threshold':threshold,'embedding_usage':usage,'limitations':dataset['limitations'],
        'attribute':{s:metrics(score(query_texts[s],anchors[s]),queries,attribute_ids,threshold) for s in strategies},
        'entity':{s:metrics(score(ent_queries_text[s],ent_anchors[s]),ent_queries,entity_ids,threshold) for s in ent_strategies},'busy':[],'topic_noise':[],'value_controls':[]}
    report['dataset']['heldout_value_queries']=len(heldout_queries)
    report['value_generalization']={
        'with_value_seen_subset':metrics(score([query_texts['fact_with_value'][i] for i in seen_comparison],anchors['fact_with_value']),[queries[i] for i in seen_comparison],attribute_ids,threshold),
        'with_value_heldout':metrics(score(heldout_with,anchors['fact_with_value']),heldout_queries,attribute_ids,threshold),
        'without_value':metrics(score(heldout_without,anchors['fact_without_value']),heldout_queries,attribute_ids,threshold)}
    report['dataset']['entity_name_swap_queries']=len(unseen_queries)
    report['entity']['compact_type_name_unseen']=metrics(score(unseen_queries,ent_anchors['compact_type_name']),ent_queries,entity_ids,threshold)
    report['limitations'].append('Primary entity-name ablation shares exemplar names; separate unseen-name queries test this confound.')
    report['limitations'].append('Whole-entity prototypes have ten facts and share background facts with queries; busy curves measure ranking instability, not an unbiased universal dilution effect.')
    ent_strategies.append('compact_type_name_unseen')
    ent_anchors['compact_type_name_unseen']=ent_anchors['compact_type_name']
    ent_queries_text['compact_type_name_unseen']=unseen_queries
    report['input_collisions']={}
    for strategy in ['fact_without_value','raw_attribute','fact_with_value']:
        indices=query_texts[strategy]+(heldout_with if strategy=='fact_with_value' else [])
        metadata=queries+(heldout_queries if strategy=='fact_with_value' else [])
        grouped={}
        for index,q in zip(indices,metadata):grouped.setdefault(index,[]).append(q)
        report['input_collisions'][strategy]=[{'targets':sorted({q['target'] for q in rows}),
            'raw_terms':sorted({q['raw_term'] for q in rows}),'cases':[q['id'] for q in rows]}
            for rows in grouped.values() if len({q['target'] for q in rows})>1]
    def ranked(index,bank,ids,target):
        row=score(index,bank);j=ids.index(target);order=np.argsort(-row,kind='stable').tolist()
        return {'similarity':float(row[j]),'rank':order.index(j)+1,'top1':ids[order[0]],'passes_threshold':bool(row[j]>threshold)}
    for b in busy:
        report['busy'].append({'target':b['target'],'facts':b['facts'],
            'whole':ranked(b['whole_query'],anchors['whole_entity'],attribute_ids,b['target']),
            'fact':ranked(b['fact_query'],anchors['fact_with_value'],attribute_ids,b['target'])})
    for n in noise:
        report['topic_noise'].append({'topics':n['topics'],'chat':ranked(n['chat_query'],anchors['chat_chat'],attribute_ids,'paint_color'),
            'fact':ranked(n['fact_query'],anchors['fact_with_value'],attribute_ids,'paint_color')})
    for c in controls:
        report['value_controls'].append({k:c[k] for k in ['target','raw','value']}|{
            'with':ranked(c['with'],anchors['fact_with_value'],attribute_ids,c['target']),
            'without':ranked(c['without'],anchors['fact_without_value'],attribute_ids,c['target'])})
    db=PostgresProvider(os.environ['POSTGRES_URL']);cache=redis.Redis.from_url(os.environ['REDIS_URL'])
    service=object.__new__(DictionaryService);service.db=db
    before=dictionary_state(db);redis_before=cache.dbsize();report['postgres_checks']=[]
    try:
        # Full catalog per representation, representative queries only. All rows
        # are temporary and rollback also occurs on any diagnostic assertion failure.
        for kind,strategies_to_check,bank,qt,qs,ids in [('ATTRIBUTE',strategies,anchors,query_texts,queries,attribute_ids),
            ('ENTITY',ent_strategies,ent_anchors,ent_queries_text,ent_queries,entity_ids)]:
            for s in strategies_to_check:
                try:
                    with db.transaction():
                        own={}
                        for j,a in enumerate(bank[s]):
                            ident=db.create_dictionary_anchor(vectors[a].tolist());own[str(ident)]=j
                            db.upsert_dictionary_keyword(ident,ids[j],kind)
                        check_inputs=[(qt[s][i],qs[i]) for i in range(0,len(qs),6 if kind=='ATTRIBUTE' else 3)]
                        if kind=='ATTRIBUTE' and s in ['fact_with_value','fact_without_value']:
                            additional=heldout_with if s=='fact_with_value' else heldout_without
                            check_inputs.extend((additional[i],heldout_queries[i]) for i in [0,1,4,5,10,11,28,29])
                        for query_index,metadata in check_inputs:
                            query=vectors[query_index].tolist();memory=score(query_index,bank[s])
                            raw=db.get_semantic_keywords(query,threshold,id_limit=len(ids)+before['tomte_semantic_anchors']['count'])
                            rows=[r for r in raw if str(r['dictionary_id']) in own]
                            expected={ids[j] for j,x in enumerate(memory) if x>threshold}
                            if expected!={r['keyword'] for r in rows}:raise AssertionError('pgvector threshold mismatch')
                            error=max((abs(r['score']-float(memory[own[str(r['dictionary_id'])]])) for r in rows),default=0)
                            if error>1e-5:raise AssertionError('pgvector cosine mismatch')
                            # For empty original dictionaries, bootstrap is an exact
                            # comparison; otherwise do not confuse pre-existing words.
                            bootstrap=service.bootstrap([query]);words=bootstrap[next(k for k in bootstrap if k.value==kind)]
                            if before['tomte_semantic_anchors']['count']==0 and set(words)!=expected:raise AssertionError('bootstrap mismatch')
                            report['postgres_checks'].append({'kind':kind,'strategy':s,'query_id':metadata['id'],
                                'matched_words':sorted(expected),'max_cosine_error':error,'bootstrap_words':words})
                        raise RollbackEvaluation()
                except RollbackEvaluation:pass
    finally:
        after=dictionary_state(db);redis_after=cache.dbsize()
        report['cleanup']={'before':before,'after':after,'exact_dictionary_restore':before==after,
            'redis_before':redis_before,'redis_after':redis_after}
        db.close();cache.close()
        OUTPUT.write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert before==after and redis_before==redis_after
    print(json.dumps({k:v for k,v in report.items() if k not in ('attribute','entity','postgres_checks')},indent=2))
    for kind in ('attribute','entity'):
        print(kind,json.dumps({s:{k:v for k,v in m.items() if k!='details'} for s,m in report[kind].items()},indent=2))
    print('pgvector_checks',len(report['postgres_checks']))


if __name__=='__main__':
    try:main()
    except Exception as exc:
        # Never render service exceptions containing connection strings/credentials.
        raise SystemExit('Granularity evaluation failed ('+type(exc).__name__+'). Review local metadata; no credentials printed.')
