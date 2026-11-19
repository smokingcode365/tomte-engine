"""Explicit host credentials must survive standalone and managed model calls."""
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest

from tomte.engine.providers.openai_provider import OpenAILLMProvider
from tomte.engine.providers.postgres_provider import PostgresProvider
from tomte.engine.runtime import owned_operation


@pytest.mark.parametrize('managed', [False, True])
def test_supplied_key_is_used_for_embedding_without_environment_key(monkeypatch, managed):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    seen = []

    async def send(client, request, *args, **kwargs):
        seen.append(request.headers.get('authorization'))
        return httpx.Response(200, request=request, json={
            'object': 'list', 'model': 'text-embedding-3-large',
            'data': [{'object': 'embedding', 'index': 0, 'embedding': [1.0]}],
            'usage': {'prompt_tokens': 1, 'total_tokens': 1},
        })

    monkeypatch.setattr(httpx.AsyncClient, 'send', send)
    template = OpenAILLMProvider('synthetic-explicit-key')
    if managed:
        database = object.__new__(PostgresProvider)
        child = SimpleNamespace(close=Mock())
        monkeypatch.setattr(database, 'fork', lambda: child)
        with owned_operation(database, template) as (_, model):
            assert model.embed_text('Synthetic evidence') == [1.0]
        child.close.assert_called_once()
    else:
        assert template.embed_text('Synthetic evidence') == [1.0]
    assert seen == ['Bearer synthetic-explicit-key']


def test_injected_key_takes_precedence_for_all_text_routes(monkeypatch):
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-unrelated-environment-key')
    provider = OpenAILLMProvider('synthetic-explicit-key')
    for name in ('strong_model', 'cheap_model', 'candidate_filter_model',
                 'answer_reader_model', 'retrieval_planner_model'):
        assert getattr(provider, name).client.api_key == 'synthetic-explicit-key'
