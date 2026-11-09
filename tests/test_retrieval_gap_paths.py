"""Real SQL coverage for formerly labelled caller-facing field searches."""
import pytest

from memory_fixtures import simple_entity, simple_fact, prepare_write
from test_postgres_transactions import local_db
from test_plain_retrieval import vector
from test_unresolved_memories import pending_payload


@pytest.mark.parametrize('route', ['weighted', 'session', 'unresolved'])
def test_caller_routes_rank_plain_fields_and_preserve_owner_scope(local_db, route):
    c = local_db
    ids = {}
    for name, plain, labelled, user in [
        ('plain winner', .99, .05, c.users[0]),
        ('labelled winner', .1, 1., c.users[0]),
        ('foreign user', 1., 1., c.users[1]),
    ]:
        payload = (pending_payload() if route == 'unresolved' else
                   simple_entity(name, facts=[simple_fact('detail', name)]))
        op = prepare_write(c.tenant, user, payload, vector=vector(.01))
        for entry in op['search_entries']: entry['embedding'] = vector(labelled)
        for entry in op['retrieval_entries']: entry['embedding'] = vector(plain)
        assert c.db.commit_graph_updates([op])
        ids[name] = payload['entity_id']

    def read(tenant, user):
        if route == 'weighted':
            return c.db.semantic_retrieval(tenant, user, vector(1.), 1)
        if route == 'session':
            return c.db.discover_session_entities(tenant, user, vector(1.), .9, 1)
        return c.db.get_unresolved_memories(tenant, user, vector(1.), 1)

    assert [row['entity_id'] for row in read(c.tenant, c.users[0])] == [ids['plain winner']]
    assert read(c.tenant + '-other', c.users[0]) == []
