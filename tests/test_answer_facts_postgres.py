"""Canonical fact lookup and actual pgvector ranking in an isolated test DB."""
from copy import deepcopy
import pytest
from memory_fixtures import simple_entity,simple_fact,prepare_write,STAMP
from test_postgres_transactions import local_db
from test_plain_retrieval import vector


def seed(c,name='Panda',kind='dog',facts=None,tenant=None,user=None,score=.5):
    e=simple_entity(name,facts=facts or [simple_fact('breed','Lab mix')])
    e.update(entity_type=kind,entity_domain='relationships')
    assert c.db.commit_graph_updates([prepare_write(tenant or c.tenant,user or c.users[0],e,vector=vector(score))])
    return e


def paths():return [{'entity_domain':'relationships','entity_type':'dog','attribute_name':'breed','field_name':'detail'}]
def search(c,selected=None,limit=200,byte_limit=24000):
    return c.db.get_answer_facts(c.tenant,c.users[0],vector(1.),paths() if selected is None else selected,limit,byte_limit)


def test_context_is_returned_for_current_and_history_without_internal_metadata(local_db):
    fact = simple_fact('game', 'We played afterward.')
    fact['value']['detail']['context'] = 'The user and Sam played chess.'
    fact['history_values'] = [{'value': deepcopy(fact['value']), '_archived_at': STAMP}]
    entity = seed(local_db, facts=[fact])
    packets = [r['packet'] for r in search(local_db, [])['facts']]
    assert len(packets) == 2
    assert {p['history_index'] for p in packets} == {-1, 0}
    assert all(p['fact']['value']['detail']['context'] == 'The user and Sam played chess.' for p in packets)
    assert all('_recorded_at' not in str(p) for p in packets)
    assert local_db.db.get_answer_facts(local_db.tenant, local_db.users[1], vector(1.), [], 100, 24000)['facts'] == []


def test_catalog_and_narrow_search_keep_low_score_matching_dogs(local_db):
    c=local_db
    dogs=[seed(c,name=n,score=s) for n,s in [('Panda',-.5),('Pepper',.1),('Pixie',.9),('Precious',.8)]]
    seed(c,'Friend','person',score=1.)
    catalog=c.db.get_answer_catalog(c.tenant,c.users[0],400)
    assert not catalog['is_partial']
    dogpath=next(p for p in catalog['paths'] if p['entity_type']=='dog')
    assert dogpath['entity_count']==4
    result=search(c)
    assert result['total_facts']==4
    assert {r['packet']['entity']['entity_id'] for r in result['facts']}=={e['entity_id'] for e in dogs}
    assert all('_recorded_at' not in str(r['packet']) for r in result['facts'])


def test_lookup_scope_unknown_paths_and_empty_corpus(local_db):
    c=local_db;own=seed(c);seed(c,'Other user',user=c.users[1])
    foreign=c.tenant+'-foreign'
    try:
        seed(c,'Other tenant',tenant=foreign)
        assert [r['packet']['entity']['entity_id'] for r in search(c)['facts']]==[own['entity_id']]
        unknown=[{**paths()[0],'field_name':'not_present'}]
        assert search(c,unknown)['total_facts']==0
        assert len(search(c,[])['facts'])==1
        assert c.db.get_answer_catalog(c.tenant,'absent',400)=={'paths':[],'is_partial':False}
    finally:c.db.delete_user_memory(foreign,c.users[0])


def test_snapshots_duplicate_attributes_and_missing_vector_are_preserved(local_db):
    c=local_db;f=simple_fact('breed','New value')
    f['history_values']=[{'value':simple_fact('breed','Old value')['value'],'_archived_at':STAMP}]
    e=seed(c,facts=[f,simple_fact('breed','Another instance'),simple_fact('ownership','The neighbor owns this dog.')])
    with c.db.transaction():
        c.db.conn.execute('DELETE FROM tomte_memory_retrieval WHERE tenant_id=%s AND user_id=%s',(c.tenant,c.users[0]))
    result=search(c)
    assert result['total_facts']==4 and len(result['facts'])==4
    assert {r['packet']['value_kind'] for r in result['facts']}=={'current','history'}
    assert any(r['packet']['fact']['attribute_name']=='ownership' for r in result['facts'])


def test_large_entity_projects_only_selected_fact_snapshots_and_reports_omissions(local_db):
    c=local_db
    e=seed(c,facts=[simple_fact('breed','Lab mix'),simple_fact('biography','x'*100000)])
    result=search(c,limit=1,byte_limit=2000)
    assert result['total_facts']==2 and result['oversized_facts']==1
    assert len(result['facts'])==1 and result['facts'][0]['packet']['fact']['attribute_name']=='breed'
    assert 'biography' not in str(result['facts'][0]['packet'])
    assert c.db.get_answer_catalog(c.tenant,c.users[0],1)['is_partial']


@pytest.mark.parametrize('selected', [[], paths()], ids=['broad', 'structured'])
def test_high_score_history_is_not_displaced_by_other_entities(local_db, selected):
    c = local_db
    fact = simple_fact('breed', 'Current information')
    fact['history_values'] = [
        {'value': simple_fact('breed', 'Earlier information')['value'], '_archived_at': STAMP}
    ]
    relevant = seed(c, facts=[fact], score=.95)
    seed(c, 'Second entity', score=.8)
    seed(c, 'Third entity', score=.7)
    with c.db.transaction():
        c.db.conn.execute(
            'UPDATE tomte_memory_retrieval SET embedding=%s::vector '
            'WHERE tenant_id=%s AND user_id=%s AND fact_id=%s AND history_index=0',
            (vector(.9), c.tenant, c.users[0], fact['fact_id']),
        )

    result = search(c, selected, limit=2)

    assert result['total_facts'] == 4
    assert [(r['packet']['entity']['entity_id'], r['packet']['history_index'])
            for r in result['facts']] == [(relevant['entity_id'], -1), (relevant['entity_id'], 0)]
