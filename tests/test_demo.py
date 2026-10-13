"""Demo orchestration contracts with deterministic local doubles, never paid APIs."""
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock
import json
import threading
import socket
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import HTTPServer

import pytest
from demo.app import Demo, DemoError, active_entity, handler_for
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


def entity(value='blue'):
    return dict(id='11111111-1111-4111-8111-111111111111', entity_id='veh_01',
                entity_name='Willow bicycle', entity_type='bicycle', aliases=['Willow'],
                facts=[dict(fact_id='current', attribute_name='color', value=value, assertion_state='ACTIVE'),
                       dict(fact_id='old', attribute_name='color', value='white', assertion_state='HISTORICAL')])


@pytest.fixture
def wired():
    # Doubles model provider interfaces only. No alternative production storage path.
    db = SimpleNamespace(rows=[], markers=set())
    db.get_user_graph_skeleton = Mock(side_effect=lambda t, u: deepcopy(db.rows))
    db.is_consolidation_processed = Mock(side_effect=lambda t, u, b: b in db.markers)
    cache = SimpleNamespace(history={}, queue={}, frozen={})
    cache.acquire_lock = Mock(return_value=True)
    cache.release_lock = Mock()
    cache.get_chat_history = Mock(side_effect=lambda t, s, user_id, limit: deepcopy(cache.history.get(s, [])[-limit:]))
    def freeze(t, u, s):
        if s not in cache.frozen: cache.frozen[s] = cache.queue.pop(s, [])
        return deepcopy(cache.frozen[s])
    def ack(t, u, s, batch=None):
        assert batch == cache.frozen.get(s) or not batch
        cache.frozen.pop(s, None)
        return True
    cache.pop_session_chat_log_4_consolidation = Mock(side_effect=freeze)
    cache.clear_session_4_consolidation = Mock(side_effect=ack)
    tenant, user = 'tomte-demo-test', 'demo-user'
    client = SimpleNamespace()
    def record(s, text, user_id, role):
        assert user_id == user
        cache.history.setdefault(s, []).append(dict(role=role, content=text))
        cache.queue.setdefault(s, []).append(dict(tenant_id=tenant, user_id=user, session_id=s,
            item_id=str(sum(len(x) for x in cache.history.values())), role=role, content=text, timestamp=1791859643))
    def send(s, text, user_id):
        history = deepcopy(cache.history.get(s, []))
        record(s, text, user_id, 'user')
        return dict(clean_message=text.strip(), chat_history=history, context_text='current context',
                    query_relevant_memories=deepcopy(db.rows), background_memories=[dict(entity_id='profile', entity_name='Demo profile', facts=[])])
    client.send_user_message = Mock(side_effect=send)
    client.log_assistant_response = Mock(side_effect=lambda s, a, user_id: record(s, a, user_id, 'assistant') or {})
    host = Mock(return_value='A fictional host reply, not user evidence.')
    workflow = SimpleNamespace(_batch_id=ConsolidationWorkflow._batch_id)
    def run(t, u, batch):
        assert (t, u) == (tenant, user)
        db.markers.add(workflow._batch_id(t, u, batch))
        return True
    workflow.run_consolidation_cycle = Mock(side_effect=run)
    app = Demo(client, db, cache, workflow, host, tenant, user)
    app.new_session()
    return app


def test_three_session_demo_wiring_with_explicit_stubbed_memory_updates(wired):
    app = wired
    first = app.sessions[0]
    app.send(first, 'I own a blue bicycle named Willow.')
    assert app.host.call_args.args[0]['conversation_history'] == []
    assert [m['role'] for m in app.cache.queue[first]] == ['user', 'assistant']
    # Fixture prescribes durable results; it is not an LLM semantic evaluation.
    def persist_blue(t, u, batch):
        app.db.rows = [entity()]
        app.db.markers.add(app.workflow._batch_id(t, u, batch))
        return True
    app.workflow.run_consolidation_cycle.side_effect = persist_blue
    assert app.consolidate(first)['status'] == 'committed'
    second = app.new_session()
    app.send(second, 'What color is Willow?')
    request = app.host.call_args.args[0]
    assert request['conversation_history'] == []
    assert request['query_relevant_memories'][0]['facts'][0]['value'] == 'blue'
    assert request['background_memories'][0]['name'] == 'Demo profile'
    assert len(request['query_relevant_memories'][0]['facts']) == 1
    app.send(second, 'I painted Willow red.')
    def persist_red(t, u, batch):
        app.db.rows = [entity('red')]
        app.db.markers.add(app.workflow._batch_id(t, u, batch))
        return True
    app.workflow.run_consolidation_cycle.side_effect = persist_red
    assert app.consolidate(second)['status'] == 'committed'
    third = app.new_session()
    app.send(third, 'What color is Willow now?')
    assert app.inspect()[0]['facts'][0]['value'] == 'red'
    assert app.inspect()[0]['id'] == entity()['id']
    assert app.state(first)['history'][0]['content'].startswith('I own')
    assert app.host.call_args.args[0]['conversation_history'] == []
    app.db.get_user_graph_skeleton.assert_called_with(app.tenant, app.user)


def test_host_failure_never_logs_fabricated_assistant(wired):
    wired.host.side_effect = RuntimeError('private provider details')
    with pytest.raises(DemoError, match='Host provider failed') as caught:
        wired.send(wired.sessions[0], 'Synthetic user fact')
    assert 'private' not in str(caught.value)
    wired.client.log_assistant_response.assert_not_called()
    assert len(wired.cache.history[wired.sessions[0]]) == 1


def test_assistant_logging_failure_keeps_generated_reply_distinct(wired):
    wired.client.log_assistant_response.side_effect = None
    wired.client.log_assistant_response.return_value = {'error': 'secret-like internal details'}
    with pytest.raises(DemoError) as caught: wired.send(wired.sessions[0], 'Synthetic fact')
    assert caught.value.generated_reply == wired.host.return_value
    assert 'internal details' not in str(caught.value)


@pytest.mark.parametrize('failure', ['false', 'exception', 'missing-marker', 'ack'])
def test_consolidation_failures_do_not_ack_unconfirmed_commit(wired, failure):
    s = wired.sessions[0]
    wired.send(s, 'Synthetic fact')
    if failure == 'exception':
        wired.workflow.run_consolidation_cycle.side_effect = RuntimeError('private')
        with pytest.raises(DemoError): wired.consolidate(s)
    else:
        if failure in ('false', 'missing-marker'):
            wired.workflow.run_consolidation_cycle.side_effect = None
            wired.workflow.run_consolidation_cycle.return_value = failure != 'false'
        if failure == 'ack':
            wired.cache.clear_session_4_consolidation.side_effect = None
            wired.cache.clear_session_4_consolidation.return_value = False
        result = wired.consolidate(s)
        assert result['status'] == ('committed_ack_pending' if failure == 'ack' else 'retry')
    assert s in wired.cache.frozen
    if failure != 'ack': wired.cache.clear_session_4_consolidation.assert_not_called()
    wired.cache.release_lock.assert_called_once()


def test_noop_marker_and_exact_ack_preserve_later_queue(wired):
    s = wired.sessions[0]
    wired.send(s, 'Hello')
    def commit(t, u, batch):
        wired.db.markers.add(wired.workflow._batch_id(t, u, batch))
        wired.cache.queue[s] = [dict(content='later message')]
        return True
    wired.workflow.run_consolidation_cycle.side_effect = commit
    assert wired.consolidate(s)['status'] == 'no_memory_change'
    assert wired.cache.queue[s][0]['content'] == 'later message'
    assert wired.cache.clear_session_4_consolidation.call_args.args[3][0]['role'] == 'user'


def test_foreign_session_and_batch_rejected_before_consolidation(wired):
    with pytest.raises(DemoError): wired.send('foreign', 'hello')
    wired.client.send_user_message.assert_not_called()
    wired.cache.pop_session_chat_log_4_consolidation.side_effect = None
    wired.cache.pop_session_chat_log_4_consolidation.return_value = [dict(tenant_id='other', user_id=wired.user, session_id=wired.sessions[0])]
    with pytest.raises(DemoError, match='scope mismatch'): wired.consolidate(wired.sessions[0])
    wired.workflow.run_consolidation_cycle.assert_not_called()
    wired.cache.clear_session_4_consolidation.assert_not_called()


def test_inspector_projects_only_active_assertions(wired):
    wired.db.rows = [dict(id=entity()['id'], payload=entity())]
    assert active_entity(wired.db.rows[0])['facts'][0]['value'] == 'blue'
    assert len(wired.inspect()[0]['facts']) == 1
    wired.db.get_user_graph_skeleton.assert_called_once_with(wired.tenant, wired.user)


def test_local_http_page_and_origin_token_boundary(wired):
    server = HTTPServer(('127.0.0.1', 0), Mock())
    port = server.server_address[1]
    server.RequestHandlerClass = handler_for(wired, 'test-token', port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    origin = f'http://127.0.0.1:{port}'
    try:
        with urlopen(origin + '/', timeout=5) as reply:
            assert 'Tomte V1' in reply.read().decode()
        raw = json.dumps(dict(session=wired.sessions[0])).encode()
        with pytest.raises(HTTPError) as caught:
            urlopen(Request(origin + '/api/inspect', data=raw), timeout=5)
        assert caught.value.code == 403
        request = Request(origin + '/api/inspect', data=raw,
                          headers={'Origin': origin, 'X-Demo-Token': 'test-token'})
        with urlopen(request, timeout=5) as reply: assert json.load(reply) == {'entities': []}
    finally:
        server.shutdown(); server.server_close(); thread.join(timeout=5)


def test_idle_browser_preconnection_does_not_block_next_request(wired):
    server = HTTPServer(('127.0.0.1', 0), Mock())
    port = server.server_address[1]
    base = handler_for(wired, 'test-token', port)
    accepted = threading.Event()
    class Handler(base):
        def setup(self):
            super().setup()
            accepted.set()
    server.RequestHandlerClass = Handler
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    idle = socket.create_connection(('127.0.0.1', port), timeout=5)
    try:
        assert accepted.wait(2)
        with urlopen(f'http://127.0.0.1:{port}/', timeout=5) as reply:
            assert 'Tomte V1' in reply.read().decode()
        assert idle.recv(1) == b''
        wired.client.send_user_message.assert_not_called()
    finally:
        idle.close(); server.shutdown(); server.server_close(); thread.join(timeout=5)
