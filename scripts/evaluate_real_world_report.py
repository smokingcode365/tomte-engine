"""Offline scoring/reporting of real-world harness evidence. No service/API calls."""
import argparse,csv,html,json,statistics
from collections import Counter,defaultdict
from pathlib import Path
from evaluate_memory_ordering import rank_candidates
from evaluate_real_world_conversations import evaluate_state, fact_matches
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'log/real_world'
DOMAINS={'tesla':'assets_and_possessions','honda':'assets_and_possessions','terry':'relationships','bryan':'relationships','luna':'relationships','tomte':'activities_and_skills'}
POLICIES=('current','session_recent_semantic','session_semantic_recent','semantic_session_recent','max_overlap','add_medium')


def identify(row,truth):
    domain=row.get('entity_domain',row.get('payload',{}).get('entity_domain'))
    names=[row['entity_name'],*row['payload'].get('aliases',[])]
    keys=[k for k,spec in truth['entities'].items() if domain==DOMAINS[k] and any(n.lower() in ' '.join(names).lower() for n in spec['names'])]
    return keys[0] if len(keys)==1 else None


def jaccard(left,right):
    a,b=set(left),set(right)
    return len(a&b)/len(a|b) if a|b else 1.0


def metrics(rows,policy):
    gold=sum(len(r['required']) for r in rows)
    ranks=[r['orders'][policy].index(e)+1 for r in rows for e in r['required'] if e in r['orders'][policy]]
    result=dict(turns=len(rows),required=gold,availability=len(ranks)/gold if gold else None,mean_rank=statistics.mean(ranks) if ranks else None)
    for k in (1,3,5):
        total=sum(len(r['orders'][policy][:k]) for r in rows)
        result['recall'+str(k)]=sum(len(set(r['orders'][policy][:k])&set(r['required'])) for r in rows)/gold if gold else None
        result['precision'+str(k)]=sum(len(set(r['orders'][policy][:k])&set(r['acceptable'])) for r in rows)/total if total else None
        result['irrelevant'+str(k)]=(total-sum(len(set(r['orders'][policy][:k])&set(r['acceptable'])) for r in rows))/len(rows) if rows else None
        eligible=[r for r in rows if r['required']]
        # Previous-topic displacement includes acceptable continuity; association, not irrelevance or causation.
        result['stale_displacement'+str(k)]=sum(bool(set(r['required'])-set(r['orders'][policy][:k])) and bool((set(r.get('prior_topics',[]))-set(r['required']))&set(r['orders'][policy][:k])) for r in eligible)/len(eligible) if eligible else None
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output-dir',default='log/real_world');args=parser.parse_args()
    global OUT
    OUT=(ROOT/args.output_dir).resolve()
    assert OUT.is_relative_to(ROOT/'log')
    original=json.loads((OUT/'report.json').read_text(encoding='utf-8'))
    calls=json.loads((OUT/'model_calls.json').read_text(encoding='utf-8'))
    usage=json.loads((OUT/'api_usage.json').read_text(encoding='utf-8'))
    runs=original['runs'];sources={};severity=[];resolution=[];mutation=Counter();dictionary=Counter();value_mutation_fields=0
    filler_texts={'yeah','maybe','okay','i see','not sure',"i don't know",'anyway...','oh, one more thing','i think i said this already.'}
    # This map is only used by scoring after model calls finish.
    for call in calls:
        for item in call.get('input',{}).get('CURRENT_BATCH',[]):
            sources[item['item_id']]=dict(role=item['role'],content=item['content'],filler=item['content'].lower() in filler_texts)
        if call['stage']=='MutationPlanResponse':
            mutation.update(m['operation'] for p in call.get('output',{}).get('plans',[]) for m in p['operations'])
        if call['stage']=='TerminologyAlignmentResponse':
            dictionary.update(d['decision'] for field in ('entity_type_decisions','attribute_decisions') for d in call.get('output',{}).get(field,[]))
            value_mutation_fields+=sum('value' in d for field in ('entity_type_decisions','attribute_decisions') for d in call.get('output',{}).get(field,[]))
    # Domain-qualified identity prevents an event named Honda Civic sale from
    # being scored as a duplicate Honda vehicle. Events remain legitimate extras.
    import evaluate_real_world_conversations as harness
    old=harness.entity_key;harness.entity_key=identify
    try:
        for run in runs:
            truth=next(t for t in json.loads((ROOT/'evals/real_world/truth.json').read_text(encoding='utf-8')) if t['trace_id']==run['trace_id']);run['hidden_truth']=truth;final=run['final_records'];byid={r['entity_id']:r for r in final}
            run['state_score']=evaluate_state(final,truth,sources)
            previous=None;prior=set();prior_session=None;new_filler=[];annotations={a['turn_id']:a for a in truth['turn_annotations']}
            turns={t['turn_id']:t for t in run['turns']}
            for row in run['turn_results']:
                row['checkpoint']=next((c for c in truth['retrieval_checkpoints'] if c['turn_id']==row['turn_id']),None)
                if row['checkpoint']:
                    row['required']=row['checkpoint']['required'];row['acceptable']=sorted(set(row['acceptable'])|set(row['checkpoint']['acceptable']))
                session=turns[row['turn_id']]['session_id']
                if session!=prior_session:prior=set();previous=None;prior_session=session
                ordered,cpu=rank_candidates([{'entity_id':eid} for eid in row['candidate_ids']],row['candidate_meta'])
                # Do not let multiple rows mapped to one gold label inflate recall/rank.
                row['orders']={p:[identify(byid[eid],truth) or 'extra:'+eid for eid in ordered[p]] for p in POLICIES}
                row['prior_topics']=sorted(prior)
                # Retain prior topics as acceptable, matching the original label policy.
                if row['checkpoint']:prior.update(row['required'])
                if annotations.get(row['turn_id'],{}).get('filler') and previous:
                    rank_change={}
                    for p in POLICIES:
                        shared=set(previous['orders'][p])&set(row['orders'][p]);needed=set(previous['required']) or prior
                        rank_change[p]=dict(distance=statistics.mean(abs(previous['orders'][p].index(e)-row['orders'][p].index(e)) for e in shared) if shared else None,
                            required_rank_changes={e:row['orders'][p].index(e)-previous['orders'][p].index(e) for e in needed if e in row['orders'][p] and e in previous['orders'][p]})
                    new_filler.append(dict(turn_id=row['turn_id'],mru_changed=row['mru_before']!=row['mru_after'],semantic_jaccard=jaccard(previous['semantic_ids'],row['semantic_ids']),ranking=rank_change))
                previous=row
            run['filler']=new_filler
            run['retrieval']=[r for r in run['turn_results'] if r['checkpoint']]
            # Later lists use final known scope identity, excluding unrelated event rows.
            for later in run['later']:
                # Original string labels lacked domains; recalc from final record names in context is unavailable.
                later['limitation']='Recorded live availability uses initial name matching; entity domains verified in final state, not later-context snapshot.'
            score=run['state_score']
            def add(level,code,detail):severity.append(dict(trace=run['trace_id'],rep=run['repetition'],severity=level,code=code,detail=detail))
            if score['contradictions']:add('CRITICAL','contradictory_active_checklist',score['contradictions'])
            if score['duplicate_entities']:add('HIGH','duplicate_identity_candidate',score['duplicate_entities'])
            if score['forbidden_active']:add('HIGH','missed_correction_or_retraction',score['forbidden_active'])
            if score['unsupported_provenance']:add('HIGH','unsupported_provenance',score['unsupported_provenance'])
            destructive=[]
            for spec in score['missing_facts']:
                if spec['state']!='ACTIVE':continue
                for record in final:
                    if identify(record,truth)!=spec['entity']:continue
                    facts=record['payload']['facts']
                    invalid=[f for f in facts if f.get('assertion_state')=='CORRECTED_INVALID' and fact_matches(f,spec)]
                    replacements=[f for f in facts if f.get('assertion_state')=='ACTIVE' and any(k in f['attribute_name'].lower() for k in spec['attribute_contains']) and not fact_matches(f,spec)]
                    if invalid and replacements:
                        detail=dict(entity=record['entity_name'],expected=spec,invalidated=invalid,replacements=replacements)
                        destructive.append(spec);add('CRITICAL','destructive_incorrect_correction',detail)
            ordinary=[s for s in score['missing_facts'] if s not in destructive]
            if ordinary:add('MEDIUM','missing_checklist_facts',ordinary)
            processed={item for b in run['batches'] if b.get('success') for item in b['items']}
            retained={item for b in run['batches'] if not b.get('success') for item in b['items']}-processed
            run['processing_coverage']=dict(successfully_processed_unique_messages=len(processed),input_messages=len(run['turns']),retained_unprocessed_messages=len(retained))
            # Literal first-mention latency includes consolidation scheduling;
            # it is not a causal estimate of the ordering policy's speed.
            positions={t['turn_id']:i for i,t in enumerate(run['turns'])}
            rises=[]
            for key,spec in truth['entities'].items():
                mentions=[positions[t['turn_id']] for t in run['turns'] if t['role']=='user' and any(n.lower() in t['content'].lower() for n in spec['names'])]
                if not mentions:continue
                start=min(mentions)
                hit=next((row for row in run['turn_results'] if positions[row['turn_id']]>=start and key in row['orders']['current']),None)
                rises.append(dict(entity=key,input_turn_lag=positions[hit['turn_id']]-start if hit else None,first_available_turn=hit['turn_id'] if hit else None))
            run['new_topic_rise']=rises
            for batch in run['batches']:
                if batch.get('error_type'):add('HIGH','consolidation_error',batch['error_type'])
            # Semantic identity resolution is assessed only for gold-identifiable objects.
            for call in [c for c in calls if c['trace']==run['trace_id']+'-rep'+str(run['repetition']) and c['stage']=='ResolutionResponse']:
                objects={e['id']:e for e in call['input']['EXTRACTED_SEMANTIC_INFORMATION']}
                for response in call.get('output',{}).get('resolutions',[]):
                    e=objects[response['extracted_entity_id']];source=identify(dict(entity_name=e['name'],entity_domain=e['domain'],payload={'aliases':e['aliases']}),truth)
                    offered={c['id']:c for c in call['input']['CANDIDATES_BY_OBJECT'][e['id']]}
                    targets=[identify(dict(entity_name=offered[t['db_id']]['name'],entity_domain=offered[t['db_id']]['domain'],payload={'aliases':offered[t['db_id']]['aliases']}),truth) for t in response['targets']]
                    if source and targets:
                        correct=all(x==source for x in targets);resolution.append(dict(trace=run['trace_id'],source=source,targets=targets,correct=correct))
                        if not correct:add('HIGH','wrong_resolution_candidate',dict(source=source,targets=targets,note='Proposed model mapping, not proof of persisted destructive mutation.'))
    finally:harness.entity_key=old
    base=[r for r in runs if r['repetition']==1];checkpoint_rows=[row for r in base for row in r['retrieval']]
    extraction=[]
    for run in runs:
        extracted=[e for c in calls if c['trace']==run['trace_id']+'-rep'+str(run['repetition']) and c['stage']=='ExtractionMemorySchema' for entities in c.get('output',{}).values() for e in entities]
        gold=run['hidden_truth']['expected_final_facts'];matched=sum(any(identify(dict(entity_name=e['entity_name'],entity_domain=e['entity_domain'],payload=e),run['hidden_truth'])==spec['entity'] and any(fact_matches(f,spec) for f in e['facts']) for e in extracted) for spec in gold)
        extraction.append(dict(trace=run['trace_id'],rep=run['repetition'],expected_checklist=len(gold),matched=matched,recall=matched/len(gold),scope='Cumulative raw extraction coverage, not per-source semantic extraction precision.'))
    repeat={tid:[r['state_score'] for r in runs if r['trace_id']==tid] for tid in ('long-01','long-09')}
    summaries=dict(base_traces=len(base),executions=len(runs),total_input_turns=sum(len(r['turns']) for r in runs),base_input_turns=sum(len(r['turns']) for r in base),users=len({(r['tenant'],r['user']) for r in base}),sessions=sum(len({t['session_id'] for t in r['turns']}) for r in base),
        retrieval={p:metrics(checkpoint_rows,p) for p in POLICIES},old_topic_recovery={p:metrics([r for r in checkpoint_rows if r['checkpoint']['kind']=='old_return'],p) for p in POLICIES},cross_session={p:metrics([row for r in base for row in r['retrieval'] if next(t['session_id'] for t in r['turns'] if t['turn_id']==row['turn_id'])=='b'],p) for p in POLICIES},
        filler=dict(probes=sum(len(r['filler']) for r in base),mru_changed=sum(f['mru_changed'] for r in base for f in r['filler']),mean_semantic_jaccard=statistics.mean(f['semantic_jaccard'] for r in base for f in r['filler']),mean_distance={p:statistics.mean(f['ranking'][p]['distance'] for r in base for f in r['filler'] if f['ranking'][p]['distance'] is not None) for p in POLICIES}),
        state=dict(expected_entity_slots=sum(r['state_score']['expected_entities'] for r in base),found_entity_slots=sum(r['state_score']['found_entities'] for r in base),duplicate_identity_candidates=sum(r['state_score']['duplicate_entities'] for r in base),missing_checklist_facts=sum(len(r['state_score']['missing_facts']) for r in base),matched_checklist_facts=sum(r['state_score']['matched_facts'] for r in base),forbidden_active=sum(len(r['state_score']['forbidden_active']) for r in base),duplicate_facts=sum(r['state_score']['duplicate_facts'] for r in base),contradictions=sum(len(r['state_score']['contradictions']) for r in base),unsupported_provenance=sum(len(r['state_score']['unsupported_provenance']) for r in base)),
        resolution=dict(scored_mappings=len(resolution),correct=sum(r['correct'] for r in resolution)),mutation_operations=dict(mutation),dictionary=dict(decisions=dict(dictionary),value_fields_returned=value_mutation_fields),
        reliability=dict(batches=sum(len(r['batches']) for r in runs),successes=sum(bool(b.get('success')) for r in runs for b in r['batches']),verified_replays=sum(bool(b.get('replay_no_api_or_row_change')) for r in runs for b in r['batches'])),
        severity=dict(Counter(s['severity'] for s in severity)),cleanup_verified=original['cleanup_verified'],usage=dict(counts=usage['counts'],input_tokens=sum(r['usage'].get('input_tokens',r['usage'].get('prompt_tokens',0)) for r in usage['requests']),output_tokens=sum(r['usage'].get('output_tokens',r['usage'].get('completion_tokens',0)) for r in usage['requests']),native_diagnostics=usage['retries']))
    source_profiles=Counter()
    for run in base:
        byid={r['entity_id']:r for r in run['final_records']}
        for row in run['retrieval']:
            for eid,m in row['candidate_meta'].items():
                if identify(byid[eid],run['hidden_truth']) in row['required']:source_profiles['+'.join(m['sources'])]+=1
    summaries['required_source_profiles']=dict(source_profiles)
    rises=[x['input_turn_lag'] for r in base for x in r['new_topic_rise'] if x['input_turn_lag'] is not None]
    summaries['new_topic_rise']=dict(observations=len(rises),mean_input_turn_lag=statistics.mean(rises) if rises else None,max_input_turn_lag=max(rises) if rises else None,limitation='Literal first mention to first returned candidate; includes manual batching and warmed shared users, not ranking-only latency.')
    summaries['topic_switch']={p:metrics([row for r in base for row in r['retrieval'] if next(a for a in r['hidden_truth']['turn_annotations'] if a['turn_id']==row['turn_id']).get('topic_switch')],p) for p in POLICIES}
    summaries['processing_coverage']=[dict(trace=r['trace_id'],rep=r['repetition'],**r['processing_coverage']) for r in runs]
    summaries['aliases_preserved_executions']=sum(all(r['state_score']['aliases'].values()) for r in runs)
    summaries['later_retrieval']=dict(queries=sum(len(r['later']) for r in runs),available=sum(set(q['required'])<=set(q['available']) for r in runs for q in r['later']),limitation='Name-based live availability, not answer generation or complete fact correctness.')
    raw_provenance=Counter()
    for call in calls:
        if call['stage']!='ExtractionMemorySchema':continue
        for entities in call.get('output',{}).values():
            for entity in entities:
                for fact in entity['facts']:
                    source=sources.get(fact.get('source_chat_id'))
                    raw_provenance['facts']+=1
                    raw_provenance['unknown_source']+=source is None
                    raw_provenance['assistant_source']+=bool(source and source['role']=='assistant')
                    raw_provenance['filler_source']+=bool(source and source['filler'])
    summaries['raw_extraction_provenance']=dict(raw_provenance)
    summaries['extraction_checklist']=dict(expected=sum(e['expected_checklist'] for e in extraction),matched=sum(e['matched'] for e in extraction),limitation='Cumulative coverage ignores lifecycle state and does not establish semantic precision.')
    if (OUT/'queue_probe.json').exists():summaries['queue_probe']=json.loads((OUT/'queue_probe.json').read_text(encoding='utf-8'))
    if (OUT/'junk_probe.json').exists():
        probe=json.loads((OUT/'junk_probe.json').read_text(encoding='utf-8'));summaries['junk_probe']=probe
        if probe.get('repeated_junk_batch'):severity.append(dict(severity='HIGH',code='junk_batch_reselected_without_acknowledgment',detail='Bounded real worker job selected the same junk batch twice; two real Luna calls, frozen messages retained.'))
    summaries['severity']=dict(Counter(s['severity'] for s in severity))
    report=dict(summary=summaries,runs=runs,extraction=extraction,resolution=resolution,repeatability=repeat,severity=severity,reliability_probes=original['reliability'],limitations=original['limitations']+['Entity matching is domain-qualified; event memories are not duplicate vehicles.','Final-state identity used to adjudicate prior retrieved canonical IDs offline.','Precise semantic hallucination rates, genuinely concurrent arrivals, and monetary cost are not established.','Hidden truth/scoring was refined during this initial run; this is not a blinded held-out benchmark.','All required ranking targets also had session evidence: semantic-only target behavior remains untested.','Conditional identity mapping agreement is not end-to-end semantic resolution accuracy.','New-topic latency includes manual batching; dictionary false-reuse precision requires richer semantic adjudication.'],raw_report='report.json',model_calls='model_calls.json')
    (OUT/'analysis.json').write_text(json.dumps(report,default=str,indent=2),encoding='utf-8')
    with (OUT/'summary.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['policy',*next(iter(summaries['retrieval'].values()))]);writer.writeheader()
        for p,m in summaries['retrieval'].items():writer.writerow(dict(policy=p,**m))
    body='<h1>Tomte realistic synthetic conversation evaluation</h1><p>Domain-qualified offline scoring. Production unchanged. Checklist scores are not universal semantic accuracy.</p><h2>Subsystem summaries</h2><pre>'+html.escape(json.dumps(summaries,indent=2))+'</pre>'
    for run in runs:
        body+='<details><summary>'+html.escape(run['trace_id']+' repetition '+str(run['repetition']))+'</summary><pre>'+html.escape(json.dumps(run,default=str,indent=2))+'</pre></details>'
    body+='<h2>Findings</h2><pre>'+html.escape(json.dumps(severity,indent=2))+'</pre><h2>Limitations</h2><pre>'+html.escape(json.dumps(report['limitations'],indent=2))+'</pre>'
    (OUT/'analysis.html').write_text('<!doctype html><meta charset="utf-8"><title>Tomte long-trace report</title><style>body{font:15px system-ui;max-width:1200px;margin:32px}pre{white-space:pre-wrap}summary{cursor:pointer;padding:12px}</style>'+body,encoding='utf-8')
    print(json.dumps(summaries,indent=2))

if __name__=='__main__':main()
