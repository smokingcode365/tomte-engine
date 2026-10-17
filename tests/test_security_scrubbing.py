"""Optional PII failure handling using local doubles, without NLP downloads."""
import json
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest

from tomte.engine.core.config import settings
from tomte.engine.providers import redis_provider
from tomte.engine.utils.security import SecurityGuard

WITHHELD = "[TEXT WITHHELD DUE TO SCRUBBER FAILURE]"
PRIVATE = "Contact synthetic-person@example.invalid"


@pytest.fixture
def enabled_guard(monkeypatch):
    monkeypatch.setattr(settings, "enable_pii_scrubbing", True)
    guard = SecurityGuard.__new__(SecurityGuard)
    guard.analyzer = None
    guard.anonymizer = None
    return guard


def test_enabled_unavailable_scrubber_withholds_queue_and_reference_text(enabled_guard, monkeypatch):
    cache = redis_provider.RedisCacheProvider.__new__(redis_provider.RedisCacheProvider)
    cache.client = Mock()
    monkeypatch.setattr(redis_provider, "get_security_guard", lambda: enabled_guard)
    cache.enqueue_for_consolidation("tenant", "user", "session", PRIVATE, "user",
                                    recent_context=[{"role": "assistant", "content": PRIVATE}])
    serialized = cache.client.pipeline.return_value.rpush.call_args.args[1]
    payload = json.loads(serialized)
    assert payload["content"] == WITHHELD
    assert payload["recent_context"] == [{"role": "assistant", "content": WITHHELD}]
    assert PRIVATE not in serialized


def test_initialization_failure_still_withholds_text(monkeypatch):
    monkeypatch.setattr(settings, "enable_pii_scrubbing", True)
    analyzer_module = ModuleType("presidio_analyzer")
    analyzer_module.AnalyzerEngine = Mock(side_effect=RuntimeError("synthetic NLP unavailable"))
    anonymizer_module = ModuleType("presidio_anonymizer")
    anonymizer_module.AnonymizerEngine = Mock()
    monkeypatch.setitem(sys.modules, "presidio_analyzer", analyzer_module)
    monkeypatch.setitem(sys.modules, "presidio_anonymizer", anonymizer_module)
    guard = SecurityGuard()
    assert guard.scrub_pii(PRIVATE) == WITHHELD


def test_missing_anonymizer_withholds_without_analyzing(enabled_guard):
    enabled_guard.analyzer = Mock()
    assert enabled_guard.scrub_pii(PRIVATE) == WITHHELD
    enabled_guard.analyzer.analyze.assert_not_called()


def test_disabled_scrubbing_returns_original_without_initializing(monkeypatch):
    monkeypatch.setattr(settings, "enable_pii_scrubbing", False)
    initialize = Mock(side_effect=AssertionError("must not initialize"))
    monkeypatch.setattr(SecurityGuard, "_initialize_pii_scrubber", initialize)
    guard = SecurityGuard()
    assert guard.scrub_pii(PRIVATE) == PRIVATE
    initialize.assert_not_called()


def test_enabled_success_preserves_scrubber_result(enabled_guard):
    enabled_guard.analyzer = Mock()
    enabled_guard.anonymizer = Mock()
    enabled_guard.anonymizer.anonymize.return_value = SimpleNamespace(text="Contact <EMAIL_ADDRESS>")
    assert enabled_guard.scrub_pii(PRIVATE) == "Contact <EMAIL_ADDRESS>"
    enabled_guard.analyzer.analyze.assert_called_once()
    assert enabled_guard.analyzer.analyze.call_args.kwargs["text"] == PRIVATE
    enabled_guard.anonymizer.anonymize.assert_called_once_with(
        text=PRIVATE, analyzer_results=enabled_guard.analyzer.analyze.return_value)


def test_runtime_scrubber_failure_keeps_existing_withheld_behavior(enabled_guard):
    enabled_guard.analyzer = Mock()
    enabled_guard.anonymizer = Mock()
    enabled_guard.analyzer.analyze.side_effect = RuntimeError("synthetic analysis failure")
    assert enabled_guard.scrub_pii(PRIVATE) == WITHHELD


def test_empty_input_remains_empty_when_scrubber_unavailable(enabled_guard):
    assert enabled_guard.scrub_pii("") == ""
