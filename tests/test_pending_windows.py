"""Window planning is mechanical; no language classification or model calls."""
from tomte.engine.core.pending_windows import plan_windows


def messages(count, start=1, text=None):
    return [dict(item_id=f'm{i}', tenant_id='t', user_id='u', session_id='s',
                 role='user', content=text or f'Synthetic message {i}', timestamp=1700000000+i,
                 recent_context=[], status='queued') for i in range(start,start+count)]


def test_overlapping_windows_preserve_all_messages():
    windows = plan_windows(messages(14), [], 6000)
    assert [w['message_ids'] for w in windows] == [
        [f'm{i}' for i in range(1,7)], [f'm{i}' for i in range(5,11)], [f'm{i}' for i in range(9,15)]]


def test_prefix_removal_reuses_intact_window_boundaries():
    old = plan_windows(messages(100), [], 6000)
    new = plan_windows(messages(99,start=2), old, 6000)
    assert [w['message_ids'] for w in new[1:]] == [w['message_ids'] for w in old[1:]]
    assert set(mid for w in new for mid in w['message_ids']) == {f'm{i}' for i in range(2,101)}


def test_long_unicode_message_is_covered_without_splitting_characters():
    source = messages(1,text='你好，世界。'*1000)
    windows = plan_windows(source, [], 1000)
    pieces = {tuple(p): unit['content'] for w in windows for p,unit in zip(w['parts'],w['units'])}
    assert ''.join(pieces[p] for p in sorted(pieces,key=lambda p:p[1])) == source[0]['content']
    assert all(len(w['text'].encode('utf-8')) <= 1000 for w in windows)
