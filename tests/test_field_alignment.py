"""Parent-scoped field vocabulary; language meaning is supplied by model stubs."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

import pytest
from memory_fixtures import value
from tomte.engine.dictionary.alignment import align_terms, align_value_fields, field_context, FieldAlignmentResponse
from tomte.engine.dictionary.service import DictionaryService
from test_postgres_transactions import local_db


def setup(fields=None, decision='REUSE', candidate='employer_name'):
    raw=[{'entity_id':'tmp_bob','entity_name':'Bob','entity_type':'person','entity_domain':'relationships',
          'aliases':[],'facts':[{'attribute_name':'employment','source_chat_id':'u1','value':fields or {
              'company':{'text':'Probably Cedar Labs; user is unsure','fact_confidence':'low'}}}]}]
    batch=[{'item_id':'u1','role':'user','content':'Bob probably works at Cedar Labs.','timestamp':1791547200}]
    context=field_context(raw[0],raw[0]['facts'][0]);anchor=str(uuid4())
    def retrieve(*args,**kw):
        assert kw['keyword_type']=='FIELD' and kw['keyword_context']==context
        return [{'dictionary_id':anchor,'keyword':candidate,'keyword_type':'FIELD','keyword_context':context,'score':.99}]
    def model(**kw):
        assert kw['schema'] is FieldAlignmentResponse
        views=json.loads(kw['user_text'])['fields']
        return FieldAlignmentResponse.model_validate({'field_decisions':[
            {**{k:v[k] for k in ('entity_handle','fact_handle','field_handle')},'decision':decision,
             'candidate_handle':v['candidates'][0]['candidate_handle'] if decision=='REUSE' else None}
            for v in views]})
    db=SimpleNamespace(get_semantic_keywords=Mock(side_effect=retrieve))
    llm=SimpleNamespace(get_vector_dimension=lambda:2,embed_batch=Mock(side_effect=lambda texts:[[1.,0.] for _ in texts]),
                        parse_structured=Mock(side_effect=model))
    return raw,batch,db,llm,context,anchor


def test_model_field_reuse_keeps_values_qualifications_and_confidence_unchanged():
    raw,batch,db,llm,context,anchor=setup();before=deepcopy(raw)
    aligned,evidence=align_value_fields(raw,batch,llm,db)
    assert aligned[0]['facts'][0]['value']=={'employer_name':raw[0]['facts'][0]['value']['company']}
    assert raw==before
    item,=evidence[0]
    assert item['keyword_context']==context and item['dictionary_id']==anchor and item['keyword']=='employer_name'
    query,=llm.embed_batch.call_args.args[0]
    assert 'Cedar' not in query and 'Bob' not in query and 'company' in query
    view=json.loads(llm.parse_structured.call_args.kwargs['user_text'])['fields'][0]
    assert view['source_text']==batch[0]['content'] and view['parent_context']==context


def test_same_attribute_and_field_names_share_vocabulary_without_merging_facts():
    raw,batch,db,llm,context,anchor=setup()
    second=deepcopy(raw[0]['facts'][0])
    second['value']['company']={'text':'Pine Labs','fact_confidence':'high'}
    raw[0]['facts'].append(second)
    aligned,evidence=align_value_fields(raw,batch,llm,db)
    assert len(aligned[0]['facts'])==2
    assert [f['value']['employer_name'] for f in aligned[0]['facts']]==[
        f['value']['company'] for f in raw[0]['facts']]
    views=json.loads(llm.parse_structured.call_args.kwargs['user_text'])['fields']
    assert {v['fact_handle'] for v in views}=={'fact_0_0','fact_0_1'}
    assert {v['parent_context'] for v in views}=={context}
    assert len(evidence[0])==2


def test_keep_raw_preserves_language_and_exact_spelling():
    raw,batch,db,llm,context,anchor=setup({'公司':{'text':'主要在家，每周一去办公室','fact_confidence':'high'}},decision='KEEP_RAW')
    aligned,evidence=align_value_fields(raw,batch,llm,db)
    assert aligned==raw and evidence[0][0]['keyword']=='公司' and evidence[0][0]['dictionary_id'] is None


def test_field_collision_rejects_whole_alignment_without_dropping_information():
    raw,batch,db,llm,context,anchor=setup({'company':value('Cedar')['detail'],'employer':value('Pine')['detail']})
    before=deepcopy(raw)
    with pytest.raises(ValueError,match='collapse'):align_value_fields(raw,batch,llm,db)
    assert raw==before


@pytest.mark.parametrize('defect',['parent','role','invented','other_field','duplicate'])
def test_field_reuse_cannot_borrow_an_unoffered_candidate(defect):
    raw,batch,db,llm,context,anchor=setup();before=deepcopy(raw)
    if defect in {'parent','role'}:
        db.get_semantic_keywords.side_effect=lambda *a,**kw:[{'dictionary_id':anchor,'keyword':'employer_name',
            'keyword_type':'ATTRIBUTE' if defect=='role' else 'FIELD','keyword_context':'other' if defect=='parent' else context,'score':.99}]
    def invalid(**kw):
        d={'entity_handle':'entity_0','fact_handle':'fact_0_0','field_handle':'field_0','decision':'REUSE',
           'candidate_handle':'invented' if defect=='invented' else 'candidate_1_0' if defect=='other_field' else 'candidate_0_0'}
        return {'field_decisions':[d,d] if defect=='duplicate' else [d]}
    llm.parse_structured.side_effect=invalid
    with pytest.raises(ValueError):align_value_fields(raw,batch,llm,db)
    assert raw==before


def test_failed_field_stage_restores_raw_parent_terms_and_discards_all_feedback():
    from test_terminology_alignment import setup_stage,reuse_response
    from tomte.engine.dictionary.alignment import TerminologyAlignmentResponse
    raw,batch,db,llm=setup_stage(TerminologyAlignmentResponse.model_validate(reuse_response()))
    parent_model=llm.parse_structured.side_effect
    def model(**kw):
        if kw['schema'] is FieldAlignmentResponse:raise RuntimeError('synthetic field provider failure')
        return parent_model(**kw)
    llm.parse_structured.side_effect=model
    result=align_terms(raw,batch,llm,db)
    assert result.entities==raw and result.evidence==[[]] and result.metrics['raw_fallback']
    assert result.metrics['embedding_calls']==2 and result.metrics['embedding_texts']==3
    assert result.metrics['candidate_lookups']==3 and result.metrics['terminology_calls']==2


def test_embedding_metrics_count_bounded_requests_in_both_phases():
    from test_terminology_alignment import setup_stage
    raw,batch,db,llm=setup_stage()
    raw[0]['facts'][0]['value']={f'field_{i}':value(str(i))['detail'] for i in range(35)}
    result=align_terms(raw,batch,llm,db)
    assert not result.metrics['raw_fallback']
    assert result.metrics['embedding_texts']==37 and result.metrics['embedding_calls']==4
    assert [len(c.args[0]) for c in llm.embed_batch.call_args_list]==[2,16,16,3]
    assert result.metrics['candidate_lookups']==37


def test_cross_anchor_maintenance_never_compares_different_field_parents():
    service=DictionaryService.__new__(DictionaryService);service.db=Mock();llm=Mock();service._llm=lambda:llm
    a,b=uuid4(),uuid4()
    service.db.get_cross_anchor_merge_candidates.return_value=[{'id_a':a,'id_b':b}]
    service.db.get_anchor_keywords.side_effect=[
        [{'keyword':'name','keyword_type':'FIELD','keyword_context':'employment'}],
        [{'keyword':'name','keyword_type':'FIELD','keyword_context':'education'}]]
    assert service.cross_anchor_merge()==set()
    llm.parse_structured.assert_not_called();service.db.consolidate_duplicate_anchors.assert_not_called()


def test_real_field_dictionary_scope_and_internal_merge_preserve_other_parent(local_db):
    db=local_db.db;vector=[1.]+[0.]*1535
    class Rollback(Exception):pass
    with pytest.raises(Rollback):
        with db.transaction():
            anchor=db.create_dictionary_anchor(vector)
            for context in ('employment','education'):
                for word in ('name','company'):
                    db.upsert_dictionary_keyword(anchor,word,'FIELD',context)
            selected=db.get_semantic_keywords(vector,.75,1,keyword_type='FIELD',keyword_limit=5,keyword_context='employment')
            assert {r['keyword'] for r in selected}=={'name','company'} and all(r['keyword_context']=='employment' for r in selected)
            db.apply_internal_merges(str(anchor),[{'kw_type':'FIELD','keyword_context':'employment','survivor':'name','victims':['company'],'total_hits':2}])
            rows=db.get_anchor_keywords(str(anchor))
            assert {(r['keyword'],r['keyword_context']) for r in rows}=={('name','employment'),('name','education'),('company','education')}
            raise Rollback()
    assert db.get_dictionary_anchor_details(anchor) is None
