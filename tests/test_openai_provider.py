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
from tomte.engine.core.candidate_filter import CandidateFilterResponse, CandidateFilterUnavailable
from tomte.engine.core.answering import AnswerDecision, AnswerFactSelection, AnswerReaderUnavailable, RequiredInformation, FactBatchJudgment
from tomte.engine.workflows.search_request import SearchEmbeddingQuery
from tomte.engine.core.catalogue import CatalogueSelection
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


@pytest.mark.parametrize("schema", [ExtractionMemorySchema])
def test_extraction_returns_typed_output_and_passes_dictionary_dependencies(provider, monkeypatch, schema):
    expected = schema()
    run = Mock(return_value=SimpleNamespace(output=expected))
    monkeypatch.setattr(openai_provider.tomte_extractor_agent, "run_sync", run)
    assert provider.parse_structured(None, "Synthetic extraction", schema,
                                     taxonomy=["vehicle"], lexicon=["paint_color"]) is expected
    run.assert_called_once()
    args, kwargs = run.call_args
    assert args == ("Synthetic extraction",) and kwargs["model"] == provider.strong_model
    assert "output_type" not in kwargs
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


@pytest.mark.parametrize("expected, model_attribute", [
    (TerminologyAlignmentResponse(), "cheap_model"),
    (CandidateFilterResponse(keep=["c1"], drop=[]), "candidate_filter_model"),
    (AnswerFactSelection(keep=["P1"]), "answer_reader_model"),
    (FactBatchJudgment(kind='collection', found=[], possible=[], related=[], unrelated=[], enough=[]), "answer_reader_model"),
    (CatalogueSelection(handles=["F1"]), "retrieval_planner_model"),
    (SearchEmbeddingQuery(embedding_text="Outdoor activities"), "retrieval_planner_model"),
    (AnswerDecision(action="answer",answer="Supported answer."), "strong_model"),
    (RequiredInformation(information_needed=["Coco's breed"]), "strong_model"),
    (ResolutionResponse(resolutions=[]), "strong_model"),
    (MutationPlanResponse(plans=[]), "strong_model"),
    (ContextualActionResponse(), "strong_model"),
], ids=["terminology", "candidate_filter", "answer_reader", "batch_reader", "retrieval_planner", "search_query", "answer", "information_planner", "resolution", "mutation", "contextual"])
def test_structured_stages_use_configured_models(provider, monkeypatch, expected, model_attribute):
    # Distinct configured routes ensure this does not depend on default model IDs.
    provider.cheap_model = "openai-responses:configured-fast-model"
    provider.candidate_filter_model = "openai-responses:configured-filter-model"
    provider.answer_reader_model = "openai-responses:configured-answer-reader"
    provider.retrieval_planner_model = "openai-responses:configured-retrieval-planner"
    provider.strong_model = "openai-responses:configured-strong-model"
    agent = Mock()
    agent.run_sync.return_value = SimpleNamespace(output=expected)
    monkeypatch.setattr(openai_provider, "Agent", Mock(return_value=agent))
    assert provider.parse_structured("Synthetic instructions", "Synthetic input", type(expected)) is expected
    agent.run_sync.assert_called_once_with(
        "Synthetic input", model=getattr(provider, model_attribute),
        model_settings={"openai_reasoning_effort": "low"})


def test_candidate_filter_has_no_format_retry_and_wraps_provider_failure(provider, monkeypatch):
    from pydantic_ai.exceptions import ModelHTTPError
    agent = Mock()
    agent.run_sync.side_effect = ModelHTTPError(503, "model", "unavailable")
    construct = Mock(return_value=agent)
    monkeypatch.setattr(openai_provider, "Agent", construct)
    with pytest.raises(CandidateFilterUnavailable):
        provider.parse_structured("filter", "input", CandidateFilterResponse)
    assert construct.call_args.kwargs["retries"] == 0
    assert provider.supports_candidate_filter is True


@pytest.mark.parametrize("role, expected_model", [
    ("strong_model", "gpt-6.1-sol"),
    ("cheap_model", "gpt-6-luna"),
    ("candidate_filter_model", "gpt-6.1-terra"),
    ("answer_reader_model", "gpt-6.1-terra"),
    ("retrieval_planner_model", "gpt-6.1-terra"),
])
def test_model_identifiers_resolve_to_responses(provider, role, expected_model):
    # Resolve through the installed library without credentials or a network request.
    client = OpenAIProvider(api_key="synthetic-unit-test-key")
    model = infer_model(getattr(provider, role), provider_factory=lambda _: client)
    assert isinstance(model, OpenAIResponsesModel)
    assert model.model_name == expected_model


@pytest.mark.parametrize('schema', [AnswerFactSelection, FactBatchJudgment])
def test_answer_reader_has_no_retry_and_wraps_only_provider_failure(provider,monkeypatch,schema):
    from pydantic_ai.exceptions import ModelHTTPError
    agent=Mock();agent.run_sync.side_effect=ModelHTTPError(503,'model','unavailable')
    construct=Mock(return_value=agent);monkeypatch.setattr(openai_provider,'Agent',construct)
    with pytest.raises(AnswerReaderUnavailable):
        provider.parse_structured('reader','input',schema)
    assert construct.call_args.kwargs['retries']==0


@pytest.mark.parametrize('managed', [False, True])
@pytest.mark.parametrize('override', [None, 'gpt-6.1-sol'])
def test_retrieval_planner_setting_applies_to_standalone_and_managed(monkeypatch, managed, override):
    from tomte.engine.core.config import TomteSettings
    from tomte.engine.runtime import LazyOpenAI
    monkeypatch.setenv('OPENAI_API_KEY', 'synthetic-unit-test-key')
    monkeypatch.delenv('OPENAI__RETRIEVAL_PLANNER_MODEL', raising=False)
    if override:
        monkeypatch.setenv('OPENAI__RETRIEVAL_PLANNER_MODEL', override)
    configured = TomteSettings(_env_file=None).openai
    monkeypatch.setattr(openai_provider.settings, 'openai', configured)
    expected = CatalogueSelection(handles=['F1'])
    agent = Mock()
    agent.run_sync.return_value = SimpleNamespace(output=expected)
    monkeypatch.setattr(openai_provider, 'Agent', Mock(return_value=agent))
    live = LazyOpenAI() if managed else openai_provider.OpenAILLMProvider('synthetic-unit-test-key')
    try:
        assert live.parse_structured('plan', 'memory request', CatalogueSelection) is expected
        model = agent.run_sync.call_args.kwargs['model']
        name = model.removeprefix('openai-responses:') if isinstance(model,str) else model.model_name
        assert name == (override or 'gpt-6.1-terra')
    finally:
        if managed:
            live.close()


@pytest.mark.parametrize("managed", [False, True])
@pytest.mark.parametrize("override", [None, "gpt-6.1-sol"])
def test_candidate_filter_setting_applies_to_standalone_and_managed_calls(monkeypatch, managed, override):
    from tomte.engine.core.config import TomteSettings
    from tomte.engine.runtime import LazyOpenAI

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unit-test-key")
    monkeypatch.setenv("OPENAI__CHEAP_MODEL", "gpt-6-luna")
    monkeypatch.setenv("OPENAI__STRONG_MODEL", "gpt-6.1-sol")
    monkeypatch.delenv("OPENAI__CANDIDATE_FILTER_MODEL", raising=False)
    if override is not None:
        monkeypatch.setenv("OPENAI__CANDIDATE_FILTER_MODEL", override)
    configured = TomteSettings(_env_file=None).openai
    expected_model = override or "gpt-6.1-terra"
    assert configured.candidate_filter_model == expected_model
    assert configured.cheap_model == "gpt-6-luna"
    assert configured.strong_model == "gpt-6.1-sol"
    monkeypatch.setattr(openai_provider.settings, "openai", configured)
    output = CandidateFilterResponse(keep=["c1"], drop=[])
    agent = Mock()
    agent.run_sync.return_value = SimpleNamespace(output=output)
    monkeypatch.setattr(openai_provider, "Agent", Mock(return_value=agent))
    live = LazyOpenAI() if managed else openai_provider.OpenAILLMProvider("synthetic-unit-test-key")
    try:
        assert live.parse_structured("filter", "synthetic input", CandidateFilterResponse) is output
        model = agent.run_sync.call_args.kwargs["model"]
        assert (model.removeprefix("openai-responses:") if isinstance(model, str) else model.model_name) == expected_model
        assert agent.run_sync.call_args.kwargs["model_settings"] == {"openai_reasoning_effort": "low"}
    finally:
        if managed:
            live.close()


def test_embedding_configuration_selects_large_model_with_storage_dimensions(monkeypatch):
    monkeypatch.setattr(openai_provider.settings, "openai", OpenAISettings())
    provider = openai_provider.OpenAILLMProvider("synthetic-unit-test-key")
    assert isinstance(provider.embedder, Embedder)
    assert provider.embedder.model.model_name == "text-embedding-3-large"
    assert provider.get_vector_dimension() == 1536


@pytest.mark.parametrize("model,dimension", [
    ("openai:text-embedding-3-large", 1536),
    ("openai:text-embedding-3-large", 1024),
    ("openai:text-embedding-3-small", 1536),
])
@pytest.mark.parametrize("managed", [False, True])
def test_query_and_document_requests_use_configured_model_and_dimensions(monkeypatch, model, dimension, managed):
    import httpx
    import json

    sent = []

    async def send(client, request, *args, **kwargs):
        payload = json.loads(request.content)
        sent.append(payload)
        texts = payload["input"]
        assert isinstance(texts, list)
        return httpx.Response(200, request=request, json={
            "object": "list", "model": payload["model"],
            "data": [{"object": "embedding", "index": index,
                      "embedding": [1.0] + [0.0] * (dimension - 1)}
                     for index, _ in enumerate(texts)],
            "usage": {"prompt_tokens": 10, "total_tokens": 10},
        })

    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-unit-test-key")
    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    monkeypatch.setattr(openai_provider.settings, "openai",
                        OpenAISettings(embed_model=model, vector_dimension=dimension))
    from tomte.engine.runtime import LazyOpenAI
    live = LazyOpenAI() if managed else openai_provider.OpenAILLMProvider("synthetic-unit-test-key")
    try:
        assert len(live.embed_text("Synthetic query")) == dimension
        assert [len(v) for v in live.embed_batch(["Synthetic first document", "Synthetic second document"])] == [dimension, dimension]
        assert len(sent) == 2
        assert all(p["model"] == model.removeprefix("openai:") and p["dimensions"] == dimension for p in sent)
        assert sent[0]["input"] == ["Synthetic query"]
        assert sent[1]["input"] == ["Synthetic first document", "Synthetic second document"]
        assert live.embed_batch([]) == [] and len(sent) == 2
    finally:
        if managed:
            live.close()
