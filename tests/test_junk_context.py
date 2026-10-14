"""Offline junk-input/gating contracts. Model decisions are prescribed, not AI-quality claims."""
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import Mock
import json
import pytest
from tomte.engine.core.extraction import RECENT_CONTEXT_TURNS
from tomte.engine.core.enums import DictionaryKeywordType
from tomte.engine.providers import openai_provider
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from tomte.engine.utils.keys import generate_common_key
from test_consolidation_mechanics import mechanics

CASES = [
    ('I know you sold your Tesla.', "No, that's wrong. You misunderstood me.", False),
    ('Your brother lives in Toronto.', "That's my cousin, not my brother.", False),
    ('Your car is blue.', 'No, I meant the other car.', False),
    ("You're going to Japan.", "That's not happening anymore.", False),
    ('Your Tesla is sold.', "I didn't say that.", False),
    ('Your Tesla is sold.', 'thanks', True),
    ('Your Tesla is sold.', 'okay', True),
    ('Your Tesla is sold.', 'lol', True),
    ('Your Tesla is sold.', 'got it', True),
    ('Your Tesla is sold.', 'cool', True),
    (None, 'No.', True),
    ('Did you sell your Tesla?', 'No.', False),
]

class ExtractionReached(Exception):
    pass

@pytest.mark.parametrize('reference,message,junk', CASES)
def test_frozen_context_reaches_provider_and_classification_gates_extraction(monkeypatch,reference,message,junk):
    provider=object.__new__(openai_provider.OpenAILLMProvider)
    provider.cheap_model='openai-responses:gpt-6-luna'
    received=[]
    def classify(text,**kwargs):
        data=json.loads(text);received.append(data)
        assert data['CURRENT_BATCH'][0]['content']==message
        assert data['RECENT_CONTEXT_REFERENCE_ONLY']==([] if reference is None else [{'role':'assistant','content':reference}])
        return SimpleNamespace(output=junk)
    monkeypatch.setattr(openai_provider.junk_filter_agent,'run_sync',classify)
    provider.parse_structured=Mock(side_effect=ExtractionReached)
    workflow=object.__new__(ConsolidationWorkflow)
    workflow.db=Mock();workflow.db.is_consolidation_processed.return_value=False
    workflow.db.transaction.return_value=nullcontext()
    workflow.llm=provider;workflow.dict_service=Mock()
    workflow.dict_service.bootstrap.return_value={DictionaryKeywordType.ENTITY:[],DictionaryKeywordType.ATTRIBUTE:[]}
    batch=[{'item_id':'current-message','tenant_id':'tenant','user_id':'user','session_id':'session',
        'timestamp':1791547200,'role':'user','content':message,'vector':[1.0],
        'recent_context':[] if reference is None else [{'role':'assistant','content':reference}]}]
    if junk:
        assert workflow.run_consolidation_cycle('tenant','user',batch) is True
        provider.parse_structured.assert_not_called();workflow.dict_service.bootstrap.assert_not_called()
        workflow.db.claim_consolidation_batch.assert_called_once()
        workflow.db.commit_graph_updates.assert_called_once_with([])
        workflow.dict_service.apply_feedback.assert_called_once_with([])
    else:
        with pytest.raises(ExtractionReached):workflow.run_consolidation_cycle('tenant','user',batch)
        provider.parse_structured.assert_called_once()
        assert json.loads(provider.parse_structured.call_args.kwargs['user_text'])==received[0]
        workflow.db.claim_consolidation_batch.assert_not_called()


def test_real_frozen_junk_input_is_bounded_isolated_and_retry_stable(mechanics,monkeypatch):
    case=mechanics;keys=[];seen=[]
    other_scopes=[(case.tenant+'-other',case.user,case.session),
                  (case.tenant,case.user+'-other',case.session),
                  (case.tenant,case.user,case.session+'-other')]
    try:
        for tenant,user,session in other_scopes:
            keys.append(generate_common_key(tenant,session,user))
            case.cache.append_chat_history(tenant,session,'Foreign context','assistant',user)
        history=[{'role':'assistant','content':f'prior claim {i}'} for i in range(7)]
        for turn in history:case.cache.append_chat_history(case.tenant,case.session,turn['content'],turn['role'],case.user)
        captured=case.cache.get_chat_history(case.tenant,case.session,case.user)
        case.cache.enqueue_for_consolidation(case.tenant,case.user,case.session,"No, that's wrong.",'user',case.llm.vector(),captured)
        frozen=case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)
        def classify(text):
            seen.append(text)
            assert json.loads(text)['RECENT_CONTEXT_REFERENCE_ONLY']==history[-RECENT_CONTEXT_TURNS:]
            return True  # Only the processed marker may persist; no extraction/model work.
        monkeypatch.setattr(case.llm,'is_junk',classify)
        workflow=ConsolidationWorkflow()
        assert workflow.run_consolidation_cycle(case.tenant,case.user,frozen) is True
        case.cache.append_chat_history(case.tenant,case.session,'Later history','assistant',case.user)
        case.cache.enqueue_for_consolidation(case.tenant,case.user,case.session,'Future message','user',case.llm.vector(),[{'role':'assistant','content':'new claim'}])
        retry=case.cache.pop_session_chat_log_4_consolidation(case.tenant,case.user,case.session)
        assert retry==frozen
        monkeypatch.setattr(case.cache,'get_chat_history',Mock(side_effect=AssertionError('Must not read live history on retry')))
        assert workflow.run_consolidation_cycle(case.tenant,case.user,retry) is True
        assert len(seen)==1  # Processed replay does not repeat classification.
    finally:
        case.cache.client.delete(*keys)
