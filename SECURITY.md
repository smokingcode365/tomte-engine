# Security policy

Tomte's SDK and administrative methods are trusted-host APIs. The host must
authenticate callers, authorize tenant/user access and protect service credentials.
The local demo is not an internet-facing application. Never expose its ports or
the development database/cache to untrusted networks.

Treat memory, model output and conversation text as untrusted input. Structural
validation and prompt boundaries reduce risks; they do not guarantee semantic
correctness or prevent every prompt injection. Optional PII scrubbing is not a
complete privacy or compliance guarantee. See [the specification](docs/specification.md)
and [operations](docs/operations.md) for data retention/deletion limits.

For a suspected vulnerability, use the repository's private vulnerability reporting
channel if available. If no private channel is enabled, open a public issue asking
the maintainer to establish a private contact, without exploit details, credentials
or private data. Do not post a working cross-tenant exploit or sensitive payload
in a public issue. No response-time or extended-support guarantee is offered.

Include affected version, impact and a minimal synthetic reproduction privately.
The v1 release line is the current maintenance target; older development snapshots
are not separately maintained.
