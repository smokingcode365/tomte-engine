# Optional provider diagnostics

Set `PROVIDER_DIAGNOSTICS_ENABLED=true` before Engine startup (environment or
local `.env`). The single setting `settings.provider_diagnostics_enabled` defaults
to `False`. Remove the setting or set it to `false` to disable it and restart.
There is no raw-payload mode. Existing normal logging is independent.

The probe uses the standard Python logger `tomte.engine.provider_diagnostics`,
following the host application's logging configuration. Failures use WARNING;
successful operations use INFO. It does not write files, configure handlers,
send telemetry, or install dependencies. A host that replaces Tomte's normal
logger still needs to configure this named logger to retain diagnostic events.

## Recorded metadata

The OpenAI, Anthropic and Gemini providers' four external-operation methods have
optional decorators. Embedding conversion hooks retain only numeric usage.
Disabled calls bypass capture, event construction and diagnostic logging.
Enabled calls emit one bounded event on exit, before an outer caller can replace
an exception. Calls, return values, original exceptions and native retries remain
unchanged. The probe does not observe memory algorithms or database operations.

Events include a local correlation ID, operation, schema/stage, model fingerprint,
duration, safe exception classes, known stop reason, validation code/path and
reviewed constant mutation-constraint codes. Native message metadata supplies
response/usage counts, retry prompts, output-tool count, finish reason and response
status when available. HTTP status and request/response ID fingerprints are
recorded when exposed. Identifiers and model names are hashed because arbitrary
provider metadata can itself contain sensitive text; hashes are not anonymization.

Transport success is true only when response/usage evidence or an HTTP rejection
is available; otherwise it is unknown, not an invented failure or success. Native
response counts are not transport dispatch counts. Stop reasons are exact matches
to reviewed constant gate reasons; unknown gate errors stay unknown. Exhaustion
is separately recorded when the SDK exposes its known exhaustion exception.

No raw prompts, user turns, provider output, vectors, headers, connection strings,
exception messages, traceback text, Pydantic input/ctx or arbitrary error-message
text is emitted. Validation paths retain schema-declared keys only; other keys
are replaced with `<field>`. At most eight exception nodes, ten validation errors
and sixteen components per path are included. No original response is recoverable
from this event. Model-level custom validator text is suppressed unless it matches
a reviewed constant constraint; unknown constraints require a narrow metadata
allowlist extension and offline regression, not unrestricted raw logging.

## Security, retention and limits

The enable flag and Git ignores are not access controls. Before enabling, the
host owner must choose a private local destination, restrict filesystem ACLs to
the operator, exclude it from shared/cloud sinks, cap its rotation/size and delete
it after the investigation (recommended maximum seven days). The Engine does not
enforce host logger permissions or retention. Do not attach whole application
logs: preexisting logging outside this probe can have different content rules.
Only reviewed sanitized events belong in shared reports. No logs are created by
the probe itself, so it needs no new repository ignore exception.

Native capture may already belong to an enclosing agent run. In that case message
history metadata is marked unavailable rather than attributing another run's
history. Safe exception-chain metadata is still attempted. SDK failures before
a parsed ModelResponse may expose neither usage nor status/IDs; fields remain
null. Logger/observer failure is suppressed to preserve application behavior,
so a configured and tested logger is necessary. This is not guaranteed delivery.
An already completed run's lost exception cannot be recovered retrospectively.

## Removal map

1. Remove decorators, the two embedding `observe_result` calls and the diagnostics
   import from each of `openai_provider.py`, `anthropic_provider.py` and
   `gemini_provider.py` under `src/tomte/engine/providers/`.
2. Remove `provider_diagnostics_enabled` from `core/config.py` and local config.
3. Delete `providers/diagnostics.py`, `tests/test_provider_diagnostics.py` and this
   document. Remove any host logger configuration and retained local records.

No memory workflow, prompt, validation rule, SDK retry/budget setting or database
schema depends on the probe. Keeping this small default-off boundary observation
is preferable to repeatedly adding ad hoc raw logs during future investigations.
