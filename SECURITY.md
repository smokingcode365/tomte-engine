# Security policy

This project is experimental; no release has a guaranteed support period or
security response time. Do not use it for sensitive or regulated data without
an independent review.

Do not post credentials, private conversations, exploit payloads containing
user data, or connection URLs in public issues. Report security concerns privately
to [smokingcode365@gmail.com](mailto:smokingcode365@gmail.com), initially with a
minimal description and no credentials or private user data. This is a documented
private contact route, not a guaranteed response-time or support commitment.

GitHub private vulnerability reporting is recommended after the public repository
is created. It has **not** been enabled or verified. Once enabled and verified,
use **Security > Report a vulnerability**; until then use the private email route.
Do not claim the GitHub reporting feature is available before checking it.

The Compose setup is trusted-local development only: loopback ports, a local
PostgreSQL superuser and passwordless Redis. It is not a public deployment.
Keep `.env` and backups private. LLM and embedding requests can send text to
external providers and incur charges; configure provider-side spending controls.

The host owns authentication and maps trusted tenant/user/session IDs. Do not
accept arbitrary visitor-supplied IDs or expose admin APIs to untrusted users.
Assistant text is context, not user-owned persistence evidence. Prompt injection
and probabilistic extraction remain risks. Current text context projects current
facts; raw structured memories include historical/invalid assertions and require
state-aware presentation. PostgreSQL and Redis do not share a transaction.
