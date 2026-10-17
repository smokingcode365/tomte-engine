"""Opt-in, metadata-only provider observation. Never changes calls or exceptions."""
from functools import wraps
from contextvars import ContextVar
import hashlib
import inspect
import json
import logging
import time
from typing import get_args
from uuid import uuid4

from pydantic import BaseModel, ValidationError
from pydantic_core import ErrorType
from pydantic_ai import capture_run_messages
from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart

from tomte.engine.core.config import settings

logger = logging.getLogger("tomte.engine.provider_diagnostics")
_result_metadata = ContextVar("provider_result_metadata", default=None)


def observe_result(result):
    """Keep only numeric usage at the embedding conversion boundary, if enabled."""
    from tomte.engine.usage import embedding_result
    embedding_result(result)
    metadata = _result_metadata.get()
    if metadata is not None:
        try:
            usage = result.usage
            metadata.update({key: _integer(getattr(usage, key, None)) for key in ("input_tokens", "output_tokens")})
        except Exception:
            pass

# Match only reviewed, constant validator messages; never serialize arbitrary msg/ctx.
_MUTATION_REASONS = {
    "Duplicate mutation sources": "duplicate_sources",
    "ADD needs an attribute and no target fact": "invalid_add_shape",
    "Mutation requires an existing target fact": "missing_target_fact",
    "Mutation requires a nonempty new value": "missing_replacement",
    "Mutation does not accept a replacement value": "replacement_forbidden",
    "Empty replacement value": "empty_replacement",
}
_GATE_REASONS = {
    "Automatic retry or split refused": "retry_refused",
    "Experiment deadline reached": "deadline_stop",
    "Spend reservation ceiling reached": "budget_stop",
    "Attempt ceiling reached": "request_limit_stop",
    "Price review expired": "price_review_stop",
    "Provider HTTP error": "http_rejection",
}


def _fingerprint(value):
    # Untrusted identifiers may themselves contain credentials or personal data.
    return hashlib.sha256(value.encode()).hexdigest()[:24] if isinstance(value, str) else None


def _name(cls):
    module = cls.__module__
    if module.startswith(("tomte.engine.", "pydantic", "openai", "httpx", "httpcore")) or module == "builtins":
        return cls.__name__[:80]
    if cls.__name__ == "GateStopped":
        return "GateStopped"
    return "external_exception"


def _chain(exc):
    result, seen, pending = [], set(), [exc]
    while pending and len(result) < 8:
        exc = pending.pop(0)
        if exc is None or id(exc) in seen:
            continue
        seen.add(id(exc)); result.append(exc)
        # Suppressed context can still contain the cause; inspect types, never strings.
        pending.append(exc.__cause__ or exc.__context__)
        if type(exc).__name__ in {"ExceptionGroup", "BaseExceptionGroup"}:
            pending.extend(exc.exceptions[:8])
    return result


def _errors(errors, schema):
    definition = schema.model_json_schema() if isinstance(schema, type) and issubclass(schema, BaseModel) else {}
    fields = {k for node in [definition, *definition.get("$defs", {}).values()] for k in node.get("properties", {}) if len(k) <= 64}
    result = []
    for error in errors[:20]:
        location = [p if type(p) is int and 0 <= p < 1000000 else p if isinstance(p, str) and p in fields else "<field>"
                    for p in error.get("loc", ())[:16]]
        code = error.get("type")
        item = {"code": code if code in get_args(ErrorType) else "custom_error", "path": location}
        # Exact allowlist comparison only. The original message/input/context is discarded.
        msg = error.get("msg")
        for text, reason in _MUTATION_REASONS.items():
            if msg in (text, "Value error, " + text):
                item["constraint"] = reason
        if item not in result:
            result.append(item)
    return result


def _integer(value):
    return value if type(value) is int and 0 <= value <= 10**12 else None


def _event(operation, schema, model, messages, result, exc, started, history_owned, result_metadata=None):
    chain = _chain(exc)
    # Do not attribute an enclosing capture's earlier run to this operation.
    history = messages if history_owned else []
    responses = [m for m in history if isinstance(m, ModelResponse)]
    retry_parts = [p for m in history for p in m.parts if isinstance(p, RetryPromptPart)]
    errors = []
    for error in chain:
        if isinstance(error, ValidationError):
            errors.extend(_errors(error.errors(include_input=False, include_context=False, include_url=False), schema))
    for part in retry_parts:
        if isinstance(part.content, list):
            errors.extend(_errors(part.content, schema))
    errors = list({json.dumps(e, sort_keys=True): e for e in errors}.values())[:10]
    names = [_name(type(e)) for e in chain]
    gate_reasons = [e.args[0] for e in chain if type(e).__name__ == "GateStopped" and e.args and isinstance(e.args[0], str)]
    stop = next((_GATE_REASONS[r] for r in gate_reasons if r in _GATE_REASONS),
                "gate_stop_unknown" if gate_reasons else None)
    category = "success"
    if exc is not None:
        if errors:
            category = "malformed_response" if any(e["code"] == "json_invalid" for e in errors) else "schema_validation"
        elif stop and stop != "gate_stop_unknown":
            category = stop
        elif any(n in {"TimeoutError", "APITimeoutError", "ReadTimeout", "ConnectTimeout", "PoolTimeout", "WriteTimeout"} for n in names):
            category = "timeout"
        elif any(n in {"APIConnectionError", "ConnectError", "NetworkError", "ReadError", "WriteError"} for n in names):
            category = "transport_network"
        elif any(n in {"ModelHTTPError", "APIStatusError", "BadRequestError", "RateLimitError", "AuthenticationError", "PermissionDeniedError", "InternalServerError"} for n in names):
            category = "http_rejection"
        elif stop:
            category = stop
        elif "UsageLimitExceeded" in names:
            category = "usage_limit_stop"
        elif any(n in {"UnexpectedModelBehavior", "IncompleteToolCall"} for n in names):
            category = "sdk_or_parse_failure"
        elif any(type(e).__module__.startswith(("pydantic_ai.", "openai.")) for e in chain):
            category = "provider_sdk_error"
        else:
            category = "engine_or_host_logic_error"
    last = responses[-1] if responses else None
    exhausted = any(type(e).__name__ == "UnexpectedModelBehavior" and e.args and isinstance(e.args[0], str)
                    and e.args[0].startswith("Exceeded maximum retries") for e in chain)
    usage = getattr(result, "usage", None)
    if callable(usage):
        usage = usage()
    usage_values = {key: sum(getattr(m.usage, key, 0) for m in responses) if responses else _integer(getattr(usage, key, None))
                    for key in ("input_tokens", "output_tokens")}
    if result_metadata:
        usage_values = result_metadata
    finish = getattr(last, "finish_reason", None)
    status = (getattr(last, "provider_details", None) or {}).get("status")
    http_status = next((_integer(getattr(e, "status_code", None)) for e in chain if getattr(e, "status_code", None) is not None), None)
    return {
        "event": "provider_diagnostic", "correlation_id": uuid4().hex,
        "operation": operation, "stage": {"MutationPlanResponse": "fact_mutation", "TerminologyAlignmentResponse": "terminology_alignment",
            "ResolutionResponse": "entity_resolution", "ExtractionMemorySchema": "extraction", "TomteMemorySchema": "extraction"}.get(getattr(schema, "__name__", ""), operation),
        "schema": _name(schema) if isinstance(schema, type) else None,
        "model_fingerprint": _fingerprint(model if isinstance(model, str) else getattr(model, "model_name", None)),
        "outcome": "failure" if exc is not None else "success", "category": category,
        "failure_stage": "output_validation" if errors else "provider_operation" if exc else None,
        "exception_chain": names, "validation_errors": errors, "stop_reason": stop,
        "error_code": errors[0]["code"] if errors else category if exc else None,
        "transport_succeeded": True if responses or result_metadata or http_status or category == "http_rejection" else None,
        "response_count": len(responses) if history_owned else None,
        "tool_output_count": sum(isinstance(p, ToolCallPart) for p in last.parts) if last else None,
        "retry_prompts": len(retry_parts) if history_owned else None,
        "retry_policy": "unchanged", "timeout_observed": category == "timeout",
        "sdk_retry_exhausted": exhausted,
        "http_status": http_status,
        "provider_request_id_fingerprint": next((_fingerprint(getattr(e, "request_id", None)) for e in chain if getattr(e, "request_id", None)), None),
        "provider_response_id_fingerprint": _fingerprint(getattr(last, "provider_response_id", None)),
        "finish_reason": finish if finish in {"stop", "length", "content_filter", "tool_call", "error"} else None,
        "response_status": status if status in {"completed", "incomplete", "failed", "cancelled"} else None,
        "usage": usage_values, "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "history_available": history_owned,
    }


def observe_provider(model_attribute):
    """Wrap a provider method; the disabled path reads one flag and delegates once."""
    def decorate(method):
        signature = inspect.signature(method)

        @wraps(method)
        def observed(self, *args, **kwargs):
            if not settings.provider_diagnostics_enabled:
                return method(self, *args, **kwargs)
            started, result, error = time.monotonic(), None, None
            bound = signature.bind(self, *args, **kwargs)
            schema = bound.arguments.get("schema")
            attribute = "cheap_model" if getattr(schema, "__name__", "") == "TerminologyAlignmentResponse" else model_attribute
            model = getattr(self, attribute, None)
            metadata = {}
            token = _result_metadata.set(metadata)
            with capture_run_messages() as messages:
                owned = not messages
                try:
                    result = method(self, *args, **kwargs)
                    return result
                except BaseException as exc:
                    error = exc
                    raise  # Identity, context and traceback propagation remain unchanged.
                finally:
                    try:
                        event = _event(method.__name__, schema, model, messages, result, error, started, owned, metadata)
                        logger.log(logging.WARNING if error else logging.INFO, json.dumps(event, sort_keys=True))
                    except Exception:
                        pass  # Observation or logging failure must not affect memory behavior.
                    finally:
                        _result_metadata.reset(token)
        from tomte.engine.usage import track_provider
        tracked = track_provider(observed)
        @wraps(method)
        def bounded(self, *args, **kwargs):
            from tomte.engine.core.capacity import check_provider_capacity
            check_provider_capacity(method.__name__, signature.bind(self, *args, **kwargs).arguments)
            return tracked(self, *args, **kwargs)
        return bounded
    return decorate
