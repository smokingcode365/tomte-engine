"""Lossless labelled rows for the fact reader; no language interpretation."""
import json


def render_filter_input(payload):
    quote = lambda value: json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    previous, candidates = payload['previously_kept'], payload['candidates']
    entities = {}
    for item in [*previous, *candidates]:
        # UUIDs group internally, never on the wire. Equal names must not merge
        # different subjects; different metadata for one UUID must also survive.
        identity = (item['_entity_id'], quote(item['entity']))
        if identity not in entities:
            entities[identity] = (f'E{len(entities) + 1}', item['entity'])
    lines = ['Need ' + quote(payload['information_needed'])]
    if payload.get('memory_owner_name') is not None:
        lines.append('Owner ' + quote(payload['memory_owner_name']))
    lines.append('Entities')
    for label, entity in entities.values():
        row = label + ' ' + ' '.join(quote(entity[k]) for k in ('entity_name', 'entity_type', 'entity_domain'))
        if entity.get('aliases'):
            row += ' aliases ' + quote(entity['aliases'])
        if entity.get('first_mentioned_at') is not None:
            row += ' first_mentioned_at ' + quote(entity['first_mentioned_at'])
        lines.append(row)
    for heading, items in [('Previous', previous), ('Candidates', candidates)]:
        if heading == 'Previous' and not items and not payload.get('previously_kept_is_partial'):
            continue
        lines.append(heading + (' (partial)' if heading == 'Previous' and payload.get('previously_kept_is_partial') else ''))
        for item in items:
            label = entities[(item['_entity_id'], quote(item['entity']))][0]
            row = f"{int(item['handle'][1:])} {label} {quote(item['attribute'])}"
            if item['value_kind'] == 'history':
                row += ' history'
            if heading == 'Previous':
                row += ' ' + item['category']
            lines.append(row)
            for name, value in item['value'].items():
                row = '  ' + quote(name) + ' ' + quote(value['text'])
                if value['fact_confidence'] != 'high':
                    row += ' ' + value['fact_confidence']
                if value.get('context'):
                    row += ' context ' + quote(value['context'])
                lines.append(row)
    return '\n'.join(lines)
