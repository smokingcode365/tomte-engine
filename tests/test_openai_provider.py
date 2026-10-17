"""OpenAI result API regression tests; no credentials or network requests."""

from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock

from pydantic import BaseModel
import pytest

from pydantic_ai import Embedder
from pydantic_ai.models import infer_model
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from tomte.engine.core.config import OpenAISettings
from tomte.engine.core.schema import TomteMemorySchema
from tomte.engine.core.extraction import ExtractionMemorySchema
from tomte.engine.core.assertions import MutationPlanResponse
from tomte.engine.core.contextual import ContextualActionResponse
from tomte.engine.core.resolution import ResolutionResponse
from tomte.engine.dictionary.alignment import TerminologyAlignmentResponse
from tomte.engine.providers import openai_provider


@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr(openai_provider.settings, "openai", OpenAISettings())
    monkeypatch.setattr(openai_provider, "Embedder", Mock())
    return openai_provider.OpenAILLMProvider("synthetic-unit-test-key")


@pytest.mark.parametrize("expected", [False, True])
def test_junk_filter_returns_output_without_legacy_data(provider, monkeypatch, expected):
    result = SimpleNamespace(output=expected)
    run = Mock(return_value=result)
    monkeypatch.setattr(openai_provider.junk_filter_agent, "run_sync", run)
    assert provider.is_junk("Synthetic text") is expected
    run.assert_called_once_with("Synthetic text", model=provider.cheap_model,
                                model_settings={"openai_reasoning_effort": "none"})


@pytest.mark.parametrize("schema", [TomteMemorySchema, ExtractionMemorySchema])
def test_extraction_returns_typed_output_and_passes_dictionary_dependencies(provider, monkeypatch, schema):
    expected = schema()
    run = Mock(return_value=SimpleNamespace(output=expected))
    monkeypatch.setattr(openai_provider.tomte_extractor_agent, "run_sync", run)
    assert provider.parse_structured(None, "Synthetic extraction", schema,
                                     taxonomy=["vehicle"], lexicon=["paint_color"]) is expected
    run.assert_called_once()
    args, kwargs = run.call_args
    assert args == ("Synthetic extraction",) and kwargs["model"] == provider.strong_model
    assert kwargs["output_type"] is schema
    assert kwargs["model_settings"] == {"openai_reasoning_effort": "low"}
    assert kwargs["deps"].taxonomy == ["vehicle"] and kwargs["deps"].lexicon == ["paint_color"]


def test_generic_reconciliation_returns_typed_output(provider, monkeypatch):
    class Reconciliation(BaseModel):
        mode: Literal["INSERT", "MUTATE", "REINFORCE"]
        matched_db_id: str | None = None
    expected = Reconciliation(mode="INSERT")
    agent = Mock()
    agent.run_sync.return_value = SimpleNamespace(output=expected)
    construct = Mock(return_value=agent)
    monkeypatch.setattr(openai_provider, "Agent", construct)
    assert provider.parse_structured("Synthetic reconciliation instructions", "Synthetic input", Reconciliation) is expected
    from tomte.engine.llm.trust import TRUST_BOUNDARY
    construct.assert_called_once_with(output_type=Reconciliation, retries=1,
                                     system_prompt=TRUST_BOUNDARY + "Synthetic reconciliation instructions")
    agent.run_sync.assert_called_once_with("Synthetic input", model=provider.strong_model,
                                          model_settings={"openai_reasoning_effort": "low"})


@pytest.mark.parametrize("expected, use_cheap", [
    (TerminologyAlignmentResponse(), True),
    (ResolutionResponse(resolutions=[]), False),
    (MutationPlanResponse(plans=[]), False),
    (ContextualActionResponse(), False),
])
def test_only_terminology_uses_configured_fast_model(provider, monkeypatch, expected, use_cheap):
    # Distinct configured routes ensure this does not depend on default model IDs.
    provider.cheap_model = "openai-responses:configured-fast-model"
    provider.strong_model = "openai-responses:configured-strong-model"
    agent = Mock()
    agent.run_sync.return_value = SimpleNamespace(output=expected)
    monkeypatch.setattr(openai_provider, "Agent", Mock(return_value=agent))
    assert provider.parse_structured("Synthetic instructions", "Synthetic input", type(expected)) is expected
    agent.run_sync.assert_called_once_with(
        "Synthetic input", model=provider.cheap_model if use_cheap else provider.strong_model,
        model_settings={"openai_reasoning_effort": "low"})


@pytest.mark.parametrize("role, expected_model", [
    ("strong_model", "gpt-6.1-sol"),
    ("cheap_model", "gpt-6-luna"),
])
def test_model_identifiers_resolve_to_responses(provider, role, expected_model):
    # Resolve through the installed library without credentials or a network request.
    client = OpenAIProvider(api_key="synthetic-unit-test-key")
    model = infer_model(getattr(provider, role), provider_factory=lambda _: client)
    assert isinstance(model, OpenAIResponsesModel)
    assert model.model_name == expected_model


def test_embedding_configuration_preserves_vector_space(monkeypatch):
    monkeypatch.setattr(openai_provider.settings, "openai", OpenAISettings())
    provider = openai_provider.OpenAILLMProvider("synthetic-unit-test-key")
    assert isinstance(provider.embedder, Embedder)
    assert provider.embedder.model == "openai:text-embedding-3-small"
    assert provider.get_vector_dimension() == 1536
