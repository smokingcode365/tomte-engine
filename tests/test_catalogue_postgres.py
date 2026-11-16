"""Progressive index reads and exact fact paths against isolated PostgreSQL."""
from copy import deepcopy
import pytest

from memory_fixtures import simple_entity, simple_fact, prepare_write, STAMP
from test_answer_facts_postgres import seed
from test_plain_retrieval import vector
from test_postgres_transactions import local_db


def test_progressive_catalogue_reads_names_aliases_and_historical_fields(local_db):
    c = local_db
    fact = simple_fact('acquisition', 'Got Lucky in March 2022.')
    fact['history_values'] = [{'_archived_at': STAMP,
        'value': {'old_location': deepcopy(fact['value']['detail'])}}]
    lucky = seed(c, 'Lucky', facts=[fact, simple_fact('acquisition', 'Another distinct acquisition fact')])
    coco = seed(c, 'Coco')
    with c.db.transaction():
        c.db.conn.execute("UPDATE tomte_memory_engine SET payload=jsonb_set(payload,'{aliases}','[\"Lucky dog\"]'::jsonb) "
                         'WHERE tenant_id=%s AND user_id=%s AND entity_id=%s',
                         (c.tenant, c.users[0], lucky['entity_id']))
    domains = c.db.get_catalogue_domains(c.tenant, c.users[0], 400)
    assert [r['entity_domain'] for r in domains['rows']] == ['relationships']
    entities = c.db.get_catalogue_entities(c.tenant, c.users[0], ['relationships'], 400)
    assert {r['entity_name'] for r in entities['rows']} == {'Lucky', 'Coco'}
    assert next(r for r in entities['rows'] if r['entity_name'] == 'Lucky')['aliases'] == ['Lucky dog']
    facts = c.db.get_catalogue_facts(c.tenant, c.users[0], [lucky['entity_id']], 400)
    assert len(facts['rows']) == 2
    assert {r['attribute_name'] for r in facts['rows']} == {'acquisition'}
    earlier = next(r for r in facts['rows'] if r['fact_id'] == fact['fact_id'])
    assert set(earlier['field_names']) == {'detail', 'old_location'}
    assert all(r['entity_id'] == lucky['entity_id'] for r in facts['rows'])
    assert 'Got Lucky' not in str(facts)
    assert not facts['is_partial']


def test_concrete_path_selects_only_requested_fact_and_its_history(local_db):
    c = local_db
    fact = simple_fact('acquisition', 'Got Lucky in March 2022.')
    fact['history_values'] = [{'_archived_at': STAMP, 'value': deepcopy(fact['value'])}]
    lucky = seed(c, 'Lucky', facts=[fact, simple_fact('acquisition', 'A different instance'), simple_fact('breed', 'Lab')])
    seed(c, 'Coco')
    path = [{'entity_id': lucky['entity_id'], 'fact_id': fact['fact_id']}]
    found = c.db.get_answer_facts(c.tenant, c.users[0], vector(1.0), path, 200, 24000)
    assert found['total_facts'] == 2
    assert {r['packet']['fact']['fact_id'] for r in found['facts']} == {fact['fact_id']}
    assert {r['packet']['value_kind'] for r in found['facts']} == {'current', 'history'}
    broad = c.db.get_answer_facts(c.tenant, c.users[0], vector(1.0), [], 200, 24000)
    assert broad['total_facts'] == 5


@pytest.mark.parametrize('domains', [None, ['relationships']])
def test_catalogue_and_concrete_paths_remain_user_and_tenant_scoped(local_db, domains):
    c = local_db
    own = seed(c, 'Lucky')
    other = seed(c, 'Other user', user=c.users[1])
    foreign_tenant = c.tenant + '-foreign'
    try:
        foreign = seed(c, 'Other tenant', tenant=foreign_tenant)
        entities = c.db.get_catalogue_entities(c.tenant, c.users[0], domains, 400)
        assert [r['entity_id'] for r in entities['rows']] == [own['entity_id']]
        assert c.db.get_catalogue_facts(c.tenant, c.users[0], [other['entity_id'], foreign['entity_id']], 400)['rows'] == []
        for entity in (other, foreign):
            paths = [{'entity_id': entity['entity_id'], 'fact_id': entity['facts'][0]['fact_id']}]
            assert c.db.get_answer_facts(c.tenant, c.users[0], vector(1.0), paths, 200, 24000)['total_facts'] == 0
        assert c.db.get_catalogue_domains(c.tenant, 'absent', 400) == {'rows': [], 'is_partial': False}
        assert c.db.get_catalogue_entities(c.tenant, c.users[0], [], 400)['rows'] == []
        assert c.db.get_catalogue_facts(c.tenant, c.users[0], [], 400)['rows'] == []
    finally:
        c.db.delete_user_memory(foreign_tenant, c.users[0])


def test_catalogue_limits_report_partial_and_unresolved_stays_separate(local_db):
    c = local_db
    first = seed(c, 'Lucky', facts=[simple_fact('one', 'First'), simple_fact('two', 'Second')])
    seed(c, 'Coco')
    seed(c, 'Unresolved fixture')
    with c.db.transaction():
        # Test-only canonical-row fixture; catalogue must not expose unresolved rows.
        c.db.conn.execute("UPDATE tomte_memory_engine SET entity_domain='unresolved_memories' "
                         "WHERE tenant_id=%s AND user_id=%s AND payload->>'entity_name'='Unresolved fixture'",
                         (c.tenant, c.users[0]))
    assert c.db.get_catalogue_entities(c.tenant, c.users[0], ['relationships'], 1)['is_partial']
    assert c.db.get_catalogue_facts(c.tenant, c.users[0], [first['entity_id']], 1)['is_partial']
    assert all(r['entity_domain'] != 'unresolved_memories' for r in c.db.get_catalogue_domains(c.tenant, c.users[0], 400)['rows'])
    assert c.db.get_catalogue_entities(c.tenant, c.users[0], ['unresolved_memories'], 400)['rows'] == []
    assert all(r['entity_domain'] != 'unresolved_memories' for r in c.db.get_catalogue_entities(c.tenant, c.users[0], None, 400)['rows'])


def test_entity_catalogue_without_domain_filter_returns_all_owned_domains(local_db):
    c = local_db
    seed(c, 'Lucky')
    journal = simple_entity('My journal', facts=[simple_fact('writing', 'Keeps a journal')])
    journal.update(entity_type='journal', entity_domain='assets_and_possessions')
    assert c.db.commit_graph_updates([prepare_write(c.tenant, c.users[0], journal, vector=vector(.5))])
    rows = c.db.get_catalogue_entities(c.tenant, c.users[0], None, 400)
    assert {(r['entity_name'], r['entity_domain']) for r in rows['rows']} == {
        ('Lucky', 'relationships'), ('My journal', 'assets_and_possessions')}
    assert not rows['is_partial']
    assert c.db.get_catalogue_entities(c.tenant, c.users[0], None, 1)['is_partial']
    assert c.db.get_catalogue_entities(c.tenant, 'absent', None, 400)['rows'] == []
    assert c.db.get_catalogue_entities(c.tenant, c.users[0], [], 400)['rows'] == []


@pytest.mark.parametrize('path', [
    {'entity_id': 'not-a-uuid', 'fact_id': 'also-invalid'},
    {'entity_id': '00000000-0000-0000-0000-000000000001'},
])
def test_invalid_concrete_path_is_rejected(local_db, path):
    with pytest.raises(ValueError):
        local_db.db.get_answer_facts(local_db.tenant, local_db.users[0], vector(1.0), [path], 200, 24000)
