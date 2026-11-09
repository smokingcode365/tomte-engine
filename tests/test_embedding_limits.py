"""Real local tokenizer and mocked provider requests; no paid model calls."""
from copy import deepcopy
import json

import httpx
import pytest
import tiktoken

from tomte.engine.core.config import OpenAISettings, settings
from tomte.engine.llm.embedder import get_entity_summary
from tomte.engine.providers.openai_provider import OpenAILLMProvider, _fit_embedding_documents
from memory_fixtures import simple_entity, simple_fact


@pytest.mark.parametrize('text', [' x' * 8192, ' x' * 8193, '中🙂' * 6000],
                         ids=['exact-limit', 'over-limit', 'unicode-over-limit'])
def test_document_limit_counts_tokens_and_preserves_unicode(text):
    encoding = tiktoken.get_encoding('cl100k_base')
    original = [text]
    result = _fit_embedding_documents(original, 'openai:text-embedding-3-large')[0]
    assert original == [text]
    assert len(encoding.encode_ordinary(result)) <= 8192
    assert text.startswith(result) and '\ufffd' not in result
    if len(encoding.encode_ordinary(text)) <= 8192:
        assert result == text
    else:
        assert result != text and len(encoding.encode_ordinary(result)) >= 8188


@pytest.mark.parametrize('managed', [False, True])
def test_actual_document_request_preserves_fitting_summary_and_bounds_oversized_one(monkeypatch, managed):
    encoding = tiktoken.get_encoding('cl100k_base')
    payload = simple_entity('Long description', facts=[simple_fact('detail', 'A useful detail. ' * 800)])
    before = deepcopy(payload)
    summary = get_entity_summary(payload)
    assert len(summary.encode('utf-8')) > 8000
    assert len(encoding.encode_ordinary(summary)) < 8192
    long_text = '中🙂 <|endoftext|> ' * 3000
    inputs = [summary, long_text, 'A short field value.']
    sent = []

    async def send(client, request, *args, **kwargs):
        body = json.loads(request.content)
        sent.append(body)
        return httpx.Response(200, request=request, json={
            'object': 'list', 'model': body['model'],
            'data': [{'object': 'embedding', 'index': i, 'embedding': [1., 0.]}
                     for i, _ in enumerate(body['input'])],
            'usage': {'prompt_tokens': 12, 'total_tokens': 12},
        })

    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-unit-test-key')
    monkeypatch.setattr(settings, 'openai', OpenAISettings(vector_dimension=2))
    monkeypatch.setattr(httpx.AsyncClient, 'send', send)
    from tomte.engine.runtime import LazyOpenAI
    provider = LazyOpenAI() if managed else OpenAILLMProvider('synthetic-unit-test-key')
    try:
        assert provider.embed_batch(inputs) == [[1., 0.]] * 3
        assert sent[0]['input'][0] == summary
        assert sent[0]['input'][2] == inputs[2]
        shortened = sent[0]['input'][1]
        assert long_text.startswith(shortened) and '\ufffd' not in shortened
        assert len(encoding.encode_ordinary(shortened)) <= 8192
        assert provider.embed_text('An unchanged query') == [1., 0.]
        assert sent[1]['input'] == ['An unchanged query']
        assert payload == before and inputs == [summary, long_text, 'A short field value.']
    finally:
        if managed:
            provider.close()


def test_unknown_deployment_does_not_inherit_an_assumed_openai_limit():
    inputs = ['large document ' * 9000]
    assert _fit_embedding_documents(inputs, 'openai:custom-deployment') is inputs
