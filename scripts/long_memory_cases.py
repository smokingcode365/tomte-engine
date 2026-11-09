"""Structural validation/planning for authored long cases; no language scoring."""
from collections import Counter
from datetime import datetime, timedelta
import re


def validate_case(case, gold):
    assert case['split'] in ('development', 'validation')
    assert case['repetitions'] == 3 and len(case['batches']) == 16
    assert len(case['batch_metadata']) == 16
    messages = {}
    previous = None
    sessions = []
    for index, (batch, meta) in enumerate(zip(case['batches'], case['batch_metadata'])):
        assert set(meta) == {'session_id', 'at'}
        assert re.fullmatch(r'session_\d{2}', meta['session_id'])
        at = datetime.fromisoformat(meta['at'].replace('Z', '+00:00'))
        assert at.utcoffset() == timedelta(0)
        assert previous is None or at - previous >= timedelta(minutes=15)
        previous = at
        sessions.append(meta['session_id'])
        assert sum(t['role'] == 'user' for t in batch) == (3 if index % 2 == 0 else 2)
        assert any(t['role'] == 'assistant' for t in batch)
        for turn in batch:
            key = turn['message_id']
            assert re.fullmatch(r'[ua]\d{2}', key) and key not in messages
            assert key.startswith('u' if turn['role'] == 'user' else 'a')
            messages[key] = (index, turn['role'])
    assert len(set(sessions)) == 8
    assert all(sessions[i] == sessions[i + 1] for i in range(0, 16, 2))
    assert len(case['queries']) == 8
    query_ids = set()
    for query in case['queries']:
        assert set(query) == {'query_id', 'after_batch', 'text'}
        assert re.fullmatch(r'q\d{2}', query['query_id']) and query['query_id'] not in query_ids
        query_ids.add(query['query_id'])
        assert type(query['after_batch']) is int and query['after_batch'] in range(1, 16, 2)
        assert isinstance(query['text'], str) and query['text'].strip()
    counts = Counter(q['after_batch'] for q in case['queries'])
    assert counts[15] == 4 and sum(v for k, v in counts.items() if k < 15) == 4
    assert len(counts) == 5
    assert [c['after_batch'] for c in gold['checkpoints']] == list(range(1, 16, 2))
    assert gold['structure']

    def supported(entry, boundary):
        assert isinstance(entry['meaning'], str) and entry['meaning'].strip()
        assert entry['source_turns']
        for source in entry['source_turns']:
            assert source in messages and messages[source][1] == 'user'
            assert messages[source][0] <= boundary, 'Gold uses future information'

    for checkpoint in gold['checkpoints']:
        assert checkpoint['required'] and checkpoint['forbidden']
        ids = [entry['id'] for entry in checkpoint['required']]
        assert len(set(ids)) == len(ids)
        for entry in checkpoint['required']:
            supported(entry, checkpoint['after_batch'])
    queries = {q['query_id']: q for q in case['queries']}
    assert len(gold['retrieval']) == len(queries)
    assert {q['query_id'] for q in gold['retrieval']} == query_ids
    for entry in gold['retrieval']:
        supported(entry, queries[entry['query_id']]['after_batch'])


def build_long_plan(data, phase='all'):
    """Smoke runs are part of, not additional to, the declared 18-run campaign."""
    assert phase in ('all', 'smoke', 'development', 'validation')
    development = [c for c in data['cases'] if c['split'] == 'development']
    validation = [c for c in data['cases'] if c['split'] == 'validation']
    assert len(development) == len(validation) == 3
    smoke = [(case, 1) for case in development[:2]]
    remaining = [(case, repeat) for repeat in range(1, 4) for case in development
                 if repeat != 1 or case not in development[:2]]
    held_out = [(case, repeat) for repeat in range(1, 4) for case in validation]
    return {'all': smoke + remaining + held_out, 'smoke': smoke,
            'development': remaining, 'validation': held_out}[phase]


def dataset_statistics(data):
    """Measure text length without model calls or adding tokenizer dependencies."""
    result = []
    for case in data['cases']:
        users = [t['content'] for b in case['batches'] for t in b if t['role'] == 'user']
        english = [t for t in users if not re.search(r'[\u3400-\u9fff]', t)]
        lengths = [len(re.findall(r"\b[\w]+(?:['’-][\w]+)*\b", t)) for t in english]
        result.append({'case_id': case['case_id'], 'split': case['split'],
                       'user_messages': len(users), 'assistant_messages': sum(t['role'] == 'assistant' for b in case['batches'] for t in b),
                       'user_characters': sum(map(len, users)), 'user_utf8_bytes': sum(len(t.encode('utf-8')) for t in users),
                       'english_messages': len(english), 'english_words': sum(lengths),
                       'english_words_min': min(lengths), 'english_words_max': max(lengths),
                       'english_words_mean': round(sum(lengths) / len(lengths), 2)})
    return result
