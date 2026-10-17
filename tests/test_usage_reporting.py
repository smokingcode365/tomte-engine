from types import SimpleNamespace
from decimal import Decimal
from threading import Event
import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import RequestUsage
from pydantic_ai.models.function import FunctionModel
from tomte.engine.usage import UsageReporter, _observation
from test_openai_provider import provider


def test_saved_responses_cache_writes_are_unknown_after_sdk_loss():
    """Replay numeric provider evidence through the SDK's public usage extractor."""
    import json
    from pathlib import Path
    samples = json.loads((Path(__file__).parent / 'fixtures' / 'responses_usage_samples.json').read_text())
    assert sum(s['usage']['input_tokens_details']['cache_write_tokens'] for s in samples) == 20703
    events = []
    reporter = UsageReporter(events.append)
    try:
        for sample in samples:
            usage = RequestUsage.extract(sample, provider='openai', provider_url='https://api.openai.com/v1',
                                         provider_fallback='openai', api_flavor='responses')
            response = ModelResponse(parts=[], model_name=sample['model'], provider_name='openai', usage=usage)
            observation = _observation(response)
            # Installed adapter loses this field: do not present its default zero as measured.
            assert observation['cache_write_tokens'] is None
            assert observation['input_tokens'] == sample['usage']['input_tokens']
            assert observation['output_tokens'] == sample['usage']['output_tokens']
            reporter.record({'tenant_id':'synthetic','user_id':sample['scope'],'kind':'consolidation'},
                            {**observation, 'response_observed':True})
    finally:
        assert reporter.close(2)
    assert len(events) == 10
    assert all(e['cache_write_tokens'] is None and not e['usage_complete'] for e in events)
    for user, expected_input, expected_output in [('a',13639,927),('b',13635,882)]:
        row = reporter.query('synthetic',user)['totals'][0]
        assert row['cache_write_tokens'] is None and row['known_cache_write_tokens'] == 0
        assert row['unknown_cache_write_tokens'] == 5 and not row['usage_complete']
        assert row['input_tokens'] == expected_input and row['output_tokens'] == expected_output
        assert row['estimated_dollars'] is None


def test_unknown_usage_and_pricing_are_not_zero():
    unknown=_observation(ModelResponse(parts=[]))
    assert unknown['input_tokens'] is None and unknown['estimated_dollars'] is None
    r=UsageReporter()
    with r.scope('t','u','foreground'):
        r.record({'tenant_id':'t','user_id':'u','kind':'foreground'},
                 {**unknown,'response_observed':False})
    row=r.query('t','u')['totals'][0]
    assert not row['usage_complete'] and row['estimated_dollars'] is None
    assert row['observed_responses']==0


def test_cache_completeness_preserves_known_fields_and_partial_totals():
    reporter = UsageReporter()
    scope = {'tenant_id':'t','user_id':'u','kind':'consolidation'}
    for writes in (5, 0):
        response = ModelResponse(parts=[], provider_name='openai', model_name='synthetic',
                                 usage=RequestUsage(input_tokens=20,output_tokens=3,
                                                    cache_read_tokens=2,cache_write_tokens=writes))
        event = _observation(response)
        assert event['cache_write_tokens'] == (5 if writes else None)
        assert event['usage_complete'] is bool(writes)
        reporter.record(scope,{**event,'response_observed':True})
    row = reporter.query('t','u')['totals'][0]
    assert row['input_tokens'] == 40 and row['output_tokens'] == 6
    assert row['cache_read_tokens'] == 4
    assert row['cache_write_tokens'] is None and row['known_cache_write_tokens'] == 5
    assert row['unknown_usage'] == row['unknown_cache_write_tokens'] == 1
    assert row['unknown_input_tokens'] == row['unknown_output_tokens'] == 0
    assert not row['usage_complete']


def test_unknown_cache_writes_do_not_expose_incomplete_sdk_price(monkeypatch):
    monkeypatch.setattr(ModelResponse,'cost',lambda self:SimpleNamespace(total_price=Decimal('.1')))
    response = ModelResponse(parts=[],provider_name='openai',usage=RequestUsage(input_tokens=20,output_tokens=3))
    assert _observation(response)['estimated_dollars'] is None
    response.usage.cache_write_tokens = 5
    assert _observation(response)['estimated_dollars'] == .1


def test_pending_starts_after_first_usable_observation_and_scopes_isolate():
    r=UsageReporter()
    assert r.query('t','u',20)['pending_estimate'] is None
    from tomte.engine.usage import _scope
    event={'model':'synthetic', 'input_tokens':40,'output_tokens':10,
           'estimated_dollars':.002,'outcome':'success','response_observed':True,
           'cache_read_tokens':0,'cache_write_tokens':0}
    with r.scope('t','u','consolidation',100):
        _scope.get()['events'].append(event)
        r.record(_scope.get(),event)
    q=r.query('t','u',200)
    assert q['estimate_samples']==1 and q['pending_estimate']=={
        'input_tokens':80,'output_tokens':20,'estimated_dollars':.004}
    assert r.query('other','u',20)['pending_estimate'] is None
    assert r.query('other','u')['totals']==[]


def test_listener_failure_and_blocking_do_not_hold_processing():
    called=Event(); release=Event()
    def listener(event):
        called.set(); release.wait(3); raise ValueError('listener failure')
    r=UsageReporter(listener)
    event={**_observation(None),'response_observed':False}
    try:
        for _ in range(300):
            r.record({'tenant_id':'t','user_id':'u','kind':'foreground'},event)
        assert called.wait(1) and r.dropped_notifications>0
        assert r.query('t','u')['totals'][0]['observations']==300
    finally:
        release.set(); assert r.close(3)
    assert r.listener_errors>0


def test_sdk_text_usage_is_captured_without_diagnostics(provider,monkeypatch):
    from tomte.engine.core.config import settings
    monkeypatch.setattr(settings,'provider_diagnostics_enabled',False)
    def respond(messages,info):
        from pydantic_ai.messages import ToolCallPart
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,{'response':True})],
                             usage=RequestUsage(input_tokens=13,output_tokens=3))
    provider.cheap_model=FunctionModel(respond)
    r=UsageReporter()
    with r.scope('t','u','foreground'):
        assert provider.is_junk('synthetic') is True
    row=r.query('t','u')['totals'][0]
    assert row['observed_responses']==1 and row['input_tokens']==13 and row['output_tokens']==3


def test_embedding_result_tokens_and_sdk_price(provider):
    provider.embedder.embed_query_sync.return_value=SimpleNamespace(
        embeddings=[[1.]],model_name='synthetic',usage=RequestUsage(input_tokens=7),
        cost=lambda:SimpleNamespace(total_price=Decimal('.0002')))
    r=UsageReporter()
    with r.scope('t','u','foreground'):
        assert provider.embed_text('synthetic')==[1.]
    row=r.query('t','u')['totals'][0]
    assert row['input_tokens']==7 and row['estimated_dollars']==.0002


def test_failed_structured_response_usage_is_retained_without_double_count(provider,monkeypatch):
    from tomte.engine.core.config import settings
    from tomte.engine.core.assertions import MutationPlanResponse
    from pydantic_ai.exceptions import UnexpectedModelBehavior
    from pydantic_ai.messages import ToolCallPart
    monkeypatch.setattr(settings,'provider_diagnostics_enabled',True)
    calls=[]
    def respond(messages,info):
        calls.append(True)
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name,{'plans':'invalid'})],
                             usage=RequestUsage(input_tokens=19,output_tokens=4))
    provider.strong_model=FunctionModel(respond)
    provider.agent_retries=0
    r=UsageReporter()
    with r.scope('t','u','consolidation',100):
        with pytest.raises(UnexpectedModelBehavior):
            provider.parse_structured('synthetic','synthetic',MutationPlanResponse)
    row=r.query('t','u')['totals'][0]
    assert len(calls)==row['observed_responses']==1
    assert row['input_tokens']==19 and row['output_tokens']==4
    assert r.query('t','u',100)['pending_estimate'] is None


def test_installed_sdk_embedding_price_and_empty_batch(provider):
    from pydantic_ai.embeddings.result import EmbeddingResult
    result=EmbeddingResult([[1.]], inputs=['synthetic'], input_type='query',
                           model_name='text-embedding-3-small',provider_name='openai',
                           usage=RequestUsage(input_tokens=7))
    provider.embedder.embed_query_sync.return_value=result
    r=UsageReporter()
    with r.scope('t','u','foreground'):
        provider.embed_text('synthetic')
        assert provider.embed_batch(texts=[])==[]
    row=r.query('t','u')['totals'][0]
    assert row['observations']==1
    assert row['estimated_dollars']==float(result.cost().total_price)>0
