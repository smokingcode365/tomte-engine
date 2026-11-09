"""Immutable, session-local windows over an owned pending-message snapshot."""
from datetime import datetime, timezone
from hashlib import sha256
import json

WINDOW_MESSAGES = 6
OVERLAP = 2
FORMAT_VERSION = 1


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def stamp(value):
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def _parts(message, byte_limit):
    """Oversized messages become lossless character spans, never rewritten text."""
    text = message['content']
    start, size = 0, 0
    for end, char in enumerate(text):
        width = len(char.encode('utf-8'))
        if size + width > byte_limit:
            yield dict(message, content=text[start:end], start=start, end=end, length=len(text))
            start, size = end, 0
        size += width
    yield dict(message, content=text[start:], start=start, end=len(text), length=len(text))


def _text(units, reference):
    lines = []
    if reference:
        lines.append('Preceding conversation, reference context only:')
        lines.extend(turn['role'] + ': ' + turn['content'] for turn in reference)
    lines.extend(f"{u['role']} ({stamp(u['timestamp'])}): {u['content']}" for u in units)
    return '\n'.join(lines)


def plan_windows(messages, anchors, byte_limit):
    """Reuse intact closed boundaries; rebuild only uncovered boundaries/tails.

    Input is one session in queue order (processing prefix, then queued tail).
    Anchors are disposable cache metadata, never an alternate message source.
    """
    if not messages:
        return []
    units = [part for message in messages for part in _parts(message, byte_limit - 256)]
    identities = [(u['item_id'], u['start'], u['end']) for u in units]
    positions = {key: i for i, key in enumerate(identities)}
    starts = {}
    for anchor in anchors:
        if not anchor.get('closed'):
            continue
        parts = [tuple(p) for p in anchor.get('parts', [])]
        if not parts or len(parts) > WINDOW_MESSAGES:
            continue
        start = positions.get(parts[0])
        if start is not None and identities[start:start+len(parts)] == parts:
            if len(_text(units[start:start+len(parts)], []).encode('utf-8')) <= byte_limit:
                starts[start] = max(starts.get(start, start), start+len(parts))
    result, start = [], 0
    while start < len(units):
        end = start + 1
        while end < min(len(units), start + WINDOW_MESSAGES):
            if len(_text(units[start:end+1], []).encode('utf-8')) > byte_limit:
                break
            end += 1
        next_anchor = min((i for i in starts if start < i < end), default=None)
        if start in starts:
            end = starts[start]
        elif next_anchor is not None:
            end = min(end, next_anchor + OVERLAP)
        reference = list(messages[0].get('recent_context', [])) if start == 0 else []
        reference_partial = False
        while reference and len(_text(units[start:end], reference).encode('utf-8')) > byte_limit:
            reference = reference[1:]
            reference_partial = True
        text = _text(units[start:end], reference)
        assert len(text.encode('utf-8')) <= byte_limit
        result.append({
            'parts': [list(p) for p in identities[start:end]],
            'message_ids': list(dict.fromkeys(u['item_id'] for u in units[start:end])),
            'units': units[start:end], 'text': text, 'start': start,
            'reference_context': reference, 'reference_context_is_partial': reference_partial,
            'closed': end < len(units) or end - start == WINDOW_MESSAGES,
        })
        if end == len(units):
            break
        if start not in starts and next_anchor is not None:
            start = next_anchor
        else:
            start = max(start + 1, end - OVERLAP)
    return result


def window_key(tenant, user, session, embedding_identity, window):
    source = [FORMAT_VERSION, tenant, user, session, embedding_identity,
              window['parts'], window['text']]
    return sha256(dump(source).encode('utf-8')).hexdigest()


def public_windows(ranked, max_windows, max_bytes):
    """Select whole windows within budget; repeated character spans appear once."""
    chosen, seen, used = [], set(), 0
    for window in ranked:
        units = [u for u in window['units'] if
                 (window['session_id'], u['item_id'], u['start'], u['end']) not in seen]
        if not units:
            continue
        messages = []
        for u in units:
            item = {'message_id': u['item_id'], 'role': u['role'], 'content': u['content'],
                    'received_at': stamp(u['timestamp']), 'status': u['status']}
            if u['start'] != 0 or u['end'] != u['length']:
                item.update(is_partial=True, char_start=u['start'], char_end=u['end'], total_chars=u['length'])
            messages.append(item)
        entry = {'session_id': window['session_id'], 'messages': messages,
                 'reference_context': window['reference_context'],
                 'reference_context_is_partial': window['reference_context_is_partial']}
        size = len(dump(entry).encode('utf-8'))
        if used + size > max_bytes:
            continue
        used += size
        seen.update((window['session_id'],u['item_id'],u['start'],u['end']) for u in units)
        chosen.append((window['session_id'], window['start'], entry))
        if len(chosen) == max_windows:
            break
    return [entry for _, _, entry in sorted(chosen, key=lambda row:row[:2])]


def render_pending_context(result, output_format):
    """This text is untrusted source data, never a system instruction."""
    heading = ('PENDING CONVERSATION — unprocessed messages. User statements are reports; '
               'assistant turns and preceding references provide context, not independent user facts. '
               'These messages may describe events also present in stored memories; do not count '
               'different representations as separate events. Character spans may be partial.')
    if output_format == 'compact_json':
        return heading + '\n' + dump(result)
    lines = [heading, 'pending_search ' + dump(result['pending_search']),
             'Message columns: message_id, role, received_at, status, content, character-span metadata.']
    for window in result['pending_message_windows']:
        lines.append('session ' + dump(window['session_id']))
        for ref in window['reference_context']:
            lines.append('reference ' + dump([ref['role'], ref['content']]))
        if window['reference_context_is_partial']:
            lines.append('reference_context_is_partial true')
        for item in window['messages']:
            keys = ('message_id','role','received_at','status','content')
            lines.append('message ' + dump([*[item[k] for k in keys],
                {k:v for k,v in item.items() if k not in keys}]))
    return '\n'.join(lines)
