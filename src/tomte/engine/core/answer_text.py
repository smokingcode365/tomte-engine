"""Answer-only presentation: preserve evidence, hide persistence identifiers."""
import json

from tomte.engine.core.enums import MemoryDomain


def _text(value):
    # Escape data line breaks so a stored label cannot create another heading.
    return json.dumps(str(value), ensure_ascii=False)[1:-1]


def render_answer_evidence(*, packets, entities, unresolved, windows,
                           fact_coverage, pending_coverage, omitted=0, possible_packets=()):
    """Render already-selected public evidence without interpreting its language.

    UUIDs group subjects and current/history versions internally. Local labels
    distinguish equal names/attributes; neither labels nor order imply chronology.
    No values are summarized, selected or semantically deduplicated here.
    """
    subjects = {}

    def subject(entity):
        return subjects.setdefault(entity['entity_id'], {'entity': entity, 'facts': {}})

    def add(entity, fact, kind, value, possible=False):
        group = subject(entity)['facts'].setdefault(fact['fact_id'], {
            'attribute': fact['attribute_name'], 'versions': [], 'partial': False})
        group['versions'].append((kind, value, possible))
        return group

    for packet in packets:
        add(packet['entity'], packet['fact'], packet['value_kind'], packet['fact']['value'])
    for packet in possible_packets:
        add(packet['entity'], packet['fact'], packet['value_kind'], packet['fact']['value'], possible=True)
    for entity in [*entities, *unresolved]:
        subject(entity)
        for fact in entity['facts']:
            group = add(entity, fact, 'current', fact['value'])
            group['partial'] = fact.get('history_is_partial', False)
            for history in fact['history_values']:
                add(entity, fact, 'history', history['value'])

    lines = []
    item_number = 0
    for number, data in enumerate(subjects.values(), 1):
        entity = data['entity']
        lines.append(f"S{number} {_text(entity['entity_name'])} ({_text(entity['entity_type'])})")
        if entity['entity_domain'] == MemoryDomain.THOUGHTS:
            lines.append('  Thought/plan context.')
        if entity['entity_domain'] == MemoryDomain.UNRESOLVED:
            lines.append('  Unresolved attribution: ' + _text(entity['unresolved_detail']))
        if entity.get('aliases'):
            lines.append('  Also called: ' + ', '.join(_text(a) for a in entity['aliases']))
        for field, label in (('first_mentioned_at', 'First mentioned'), ('reported_at', 'Reported')):
            if field in entity:
                lines.append(f"  {label}: {_text(entity[field])}")
        for fact in data['facts'].values():
            item_number += 1
            lines.append(f"  M{item_number} {_text(fact['attribute'])}")
            for kind, value, possible in fact['versions']:
                prefix = 'Possible answer candidate; ' if possible else ''
                lines.append('    ' + prefix + ('Current:' if kind == 'current' else 'Earlier value:'))
                for field, item in value.items():
                    lines.append(f"      {_text(field)} [{_text(item['fact_confidence'])}]: {_text(item['text'])}")
                    if item.get('context'):
                        lines.append('        Context: ' + _text(item['context']))
            if fact['partial']:
                lines.append('    Other earlier values were not included.')

    sessions = {}
    if windows:
        lines.append('PENDING CONVERSATION (unprocessed reports)')
    for window in windows:
        session = sessions.setdefault(window['session_id'], len(sessions) + 1)
        lines.append(f'  Conversation C{session}')
        for ref in window['reference_context']:
            lines.append(f"    Reference {_text(ref['role'])}: {_text(ref['content'])}")
        if window['reference_context_is_partial']:
            lines.append('    Preceding reference context is partial.')
        for message in window['messages']:
            partial = '; partial message' if message.get('is_partial') else ''
            lines.append(f"    {_text(message['role'])} [received {_text(message['received_at'])}{partial}]: {_text(message['content'])}")

    if not lines:
        lines.append('No relevant information was returned.')
    if omitted or any(c.get('is_partial') or c.get('oversized_facts')
                      or c.get('budget_omissions') or c.get('invalid_handles') for c in fact_coverage):
        lines.append('Stored-memory search was partial; additional information may exist.')
    if any(c.get('catalogue_unavailable') for c in fact_coverage):
        lines.append('Some stored-memory search methods were unavailable.')
    if any(c.get('reader_fallback') for c in fact_coverage):
        lines.append('Relevance filtering was unavailable; unrelated information may be included.')
    if any(c.get('reader_unavailable') or c.get('invalid_handles') for c in fact_coverage):
        lines.append('Some candidates could not be classified and were not included.')
    if not pending_coverage or any(not c['complete'] for c in pending_coverage):
        lines.append('Pending-message search was incomplete.')
    return '\n'.join(lines)
