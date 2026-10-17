"""Pinned-SDK adapter contracts; fake credentials and no network transports."""
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from pydantic import BaseModel
from pydantic_ai import models
from pydantic_ai.embeddings.google import GoogleEmbeddingModel
from pydantic_ai.embeddings.voyageai import VoyageAIEmbeddingModel
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.google import GoogleModel

from tomte.engine.core.config import AnthropicSettings, GeminiSettings, settings
from tomte.engine.core.extraction import ExtractionMemorySchema
from tomte.engine.providers import anthropic_provider, gemini_provider
from tomte.engine.providers import diagnostics


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Network transport is forbidden in adapter tests")
    monkeypatch.setattr(httpx.Client, "send", forbidden)
    monkeypatch.setattr(httpx.AsyncClient, "send", forbidden)
    monkeypatch.setattr(models, "ALLOW_MODEL_REQUESTS", False)
    monkeypatch.setattr(settings, "provider_diagnostics_enabled", False)
    monkeypatch.setattr(settings, "anthropic", AnthropicSettings())
    monkeypatch.setattr(settings, "gemini", GeminiSettings())
    for name in ("ANTHROPIC_API_KEY", "VOYAGE_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.delenv(name, raising=False)


def test_anthropic_constructor_matches_installed_sdk():
    provider = anthropic_provider.AnthropicLLMProvider("synthetic-anthropic-key", "synthetic-voyage-key")
    assert isinstance(provider.strong_model, AnthropicModel)
    assert isinstance(provider.cheap_model, AnthropicModel)
    assert provider.strong_model.model_name == "claude-3-5-sonnet-latest"
    assert provider.cheap_model.model_name == "claude-3-haiku-20240307"
    assert isinstance(provider.embedder.model, VoyageAIEmbeddingModel)
    assert provider.embedder.model.model_name == "voyage-large-2"
    assert provider.get_vector_dimension() == 1536


@pytest.mark.parametrize("prefix", ["google-gla:", ""])
def test_gemini_constructor_matches_installed_sdk(monkeypatch, prefix):
    configured = GeminiSettings(strong_model=prefix + "gemini-1.5-pro",
                               cheap_model=prefix + "gemini-1.5-flash",
                               embed_model=prefix + "text-embedding-004")
    monkeypatch.setattr(settings, "gemini", configured)
    before = configured.model_dump()
    provider = gemini_provider.GeminiLLMProvider("synthetic-google-key")
    assert isinstance(provider.strong_model, GoogleModel)
    assert isinstance(provider.cheap_model, GoogleModel)
    assert provider.strong_model.model_name == "gemini-1.5-pro"
    assert provider.cheap_model.model_name == "gemini-1.5-flash"
    assert isinstance(provider.embedder.model, GoogleEmbeddingModel)
    assert provider.embedder.model.model_name == "text-embedding-004"
    assert provider.get_vector_dimension() == 768
    assert configured.model_dump() == before


@pytest.fixture(params=[anthropic_provider, gemini_provider], ids=["anthropic", "gemini"])
def adapter(request):
    module = request.param
    cls = module.AnthropicLLMProvider if module is anthropic_provider else module.GeminiLLMProvider
    provider = cls.__new__(cls)
    provider.strong_model, provider.cheap_model = "synthetic-strong", "synthetic-cheap"
    provider.embedder = Mock()
    return module, provider


@pytest.mark.parametrize("expected", [False, True])
def test_junk_returns_sdk_output(adapter, monkeypatch, expected):
    module, provider = adapter
    run = Mock(return_value=SimpleNamespace(output=expected))
    monkeypatch.setattr(module.junk_filter_agent, "run_sync", run)
    assert provider.is_junk("synthetic batch") is expected
    run.assert_called_once_with("synthetic batch", model="synthetic-cheap")


def test_extraction_returns_output_and_preserves_schema_and_dependencies(adapter, monkeypatch):
    module, provider = adapter
    expected = ExtractionMemorySchema()
    run = Mock(return_value=SimpleNamespace(output=expected))
    monkeypatch.setattr(module.tomte_extractor_agent, "run_sync", run)
    assert provider.parse_structured(None, "synthetic batch", ExtractionMemorySchema,
                                     taxonomy=["vehicle"], lexicon=["paint_color"]) is expected
    args, kwargs = run.call_args
    assert args == ("synthetic batch",)
    assert kwargs["model"] == "synthetic-strong" and kwargs["output_type"] is ExtractionMemorySchema
    assert kwargs["deps"].taxonomy == ["vehicle"] and kwargs["deps"].lexicon == ["paint_color"]
    run.assert_called_once()


def test_generic_returns_sdk_output_and_preserves_prompt(adapter, monkeypatch):
    module, provider = adapter
    class Reply(BaseModel):
        value: str
    expected = Reply(value="synthetic result")
    agent = Mock()
    agent.run_sync.return_value = SimpleNamespace(output=expected)
    construct = Mock(return_value=agent)
    monkeypatch.setattr(module, "Agent", construct)
    assert provider.parse_structured("synthetic instructions", "synthetic batch", Reply) is expected
    from tomte.engine.llm.trust import TRUST_BOUNDARY
    construct.assert_called_once_with(output_type=Reply, system_prompt=TRUST_BOUNDARY + "synthetic instructions")
    agent.run_sync.assert_called_once_with("synthetic batch", model="synthetic-strong")


def test_embedding_conversion_and_empty_batch_are_unchanged(adapter):
    _, provider = adapter
    provider.embedder.embed_query_sync.return_value = SimpleNamespace(embeddings=[(1.0, 2.0)])
    provider.embedder.embed_documents_sync.return_value = SimpleNamespace(embeddings=[(3.0, 4.0)])
    assert provider.embed_text("synthetic") == [1.0, 2.0]
    assert provider.embed_batch([]) == []
    provider.embedder.embed_documents_sync.assert_not_called()
    assert provider.embed_batch(["synthetic"]) == [[3.0, 4.0]]


@pytest.mark.parametrize("active", [False, True])
def test_diagnostics_switch_preserves_output(adapter, monkeypatch, caplog, active):
    module, provider = adapter
    monkeypatch.setattr(settings, "provider_diagnostics_enabled", active)
    caplog.set_level("INFO", logger=diagnostics.logger.name)
    monkeypatch.setattr(module.junk_filter_agent, "run_sync", Mock(return_value=SimpleNamespace(output=False)))
    assert provider.is_junk("synthetic batch") is False
    events = [r for r in caplog.records if r.name == diagnostics.logger.name]
    assert len(events) == int(active)
