"""Structural projection/budgets only. Language relevance belongs to the models."""
from copy import deepcopy
import json
from uuid import UUID

from tomte.engine.core.memory import public_value
from tomte.engine.exceptions import TomteSecurityViolationError


def byte_size(value):
    return len(json.dumps(value,ensure_ascii=False).encode('utf-8'))


def packet_key(packet):
    return (packet['entity']['entity_id'],packet['fact']['fact_id'],packet['history_index'])


def checked_packets(result, tenant, user):
    packets=[]
    for row in result['facts']:
        if row.get('tenant_id')!=tenant or row.get('user_id')!=user:
            raise TomteSecurityViolationError('Answer fact namespace mismatch')
        p=deepcopy(row['packet']);e=p['entity'];f=p['fact'];hi=p['history_index']
        if any(str(UUID(x))!=x for x in (e['entity_id'],f['fact_id'])):
            raise ValueError('Invalid answer fact identity')
        if type(hi) is not int or hi < -1 or p['value_kind']!=('current' if hi==-1 else 'history'):
            raise ValueError('Invalid answer fact snapshot')
        # An explicit public projection; optional future internal fields stay private.
        entity={k:e[k] for k in ('entity_id','entity_name','entity_type','entity_domain','aliases')}
        if 'first_mentioned_at' in e:entity['first_mentioned_at']=e['first_mentioned_at']
        value=public_value(f['value'])
        if not value or any(not isinstance(v['text'],str) or not v['text'].strip()
                or v['fact_confidence'] not in ('high','medium','low') for v in value.values()):
            raise ValueError('Invalid public answer value')
        packets.append({'entity':entity,'fact':{'fact_id':f['fact_id'],
            'attribute_name':f['attribute_name'],'value':value},
            'value_kind':p['value_kind'],'history_index':hi})
    return packets


def fit_packets(packets, budget):
    """Keep whole snapshots; if all fit, no score or count threshold drops one."""
    result=[];used=2;omitted=0;seen=set()
    for p in packets:
        key=packet_key(p)
        if key in seen:continue
        seen.add(key)
        size=byte_size(p)+2
        if used+size>budget:
            omitted+=1
            continue
        result.append(p);used+=size
    return result,omitted


def render_fact_evidence(packets, output_format):
    """Group identity once, preserving distinct fact UUIDs and history positions."""
    entities={}
    for p in packets:
        eid=p['entity']['entity_id']
        entities.setdefault(eid,{'entity':p['entity'],'facts':[]})['facts'].append(
            {**p['fact'],'value_kind':p['value_kind'],'history_index':p['history_index']})
    if output_format=='compact_json':
        return json.dumps(list(entities.values()),ensure_ascii=False,separators=(',',':'))
    lines=['RETRIEVED FACTS WITH ENTITY CONTEXT (other entity facts may be omitted)']
    for item in entities.values():
        lines.append('ENTITY '+json.dumps(item['entity'],ensure_ascii=False))
        for fact in item['facts']:
            lines.append('FACT '+json.dumps({k:v for k,v in fact.items() if k!='value'},ensure_ascii=False))
            for field,value in fact['value'].items():
                lines.append('  '+json.dumps(field,ensure_ascii=False)+': '+json.dumps(value,ensure_ascii=False))
    return '\n'.join(lines)
