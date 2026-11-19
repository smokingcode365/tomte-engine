# Changelog

## 1.0.0 — 2026-11-19

This version freezes existing engine functionality. No new memory feature is part
of this completion work. Release artifacts are prepared locally; no Git tag or
package publication has occurred. See [verification](docs/release-verification.md).

### Fixed

- Use the explicitly injected OpenAI API key for both standalone models and
  managed operation clients, instead of silently requiring an environment key.
- Validate admin graph-read and memory-wipe tenant/user IDs before opening
  resources or constructing a potentially colliding Redis namespace.
- Let the no-API evaluation preflight inspect the configured embedding model;
  historical live-comparison model pins remain enforced for those live runs.

### Release engineering

- Add the existing-feature specification, architecture overview, testing guide,
  operational runbook, contribution guidance and security policy.
- Add an MIT license and remove the unsupported package-description claim.
- Declare the directly used dotenv dependency and restrict source distributions
  to deliberate runtime/documentation files, excluding local caches/prototypes.
- Remove the unused `schedule` dependency (the worker owns scheduling) and the
  obsolete numeric relevance-threshold example setting and empty Makefile.

### Existing behavior retained

Canonical entity/fact UUIDs, structured values and immutable history; scoped
PostgreSQL persistence and Redis batches; semantic dictionary alignment; session
continuity; pending-memory search; opt-in lexical retrieval; LLM-assisted fact
retrieval and answering; managed workers and a local demonstration application.

Older development snapshots used package version `0.2.0`. Scalar/assertion payloads
from older memory designs are not automatically migrated. See the
[memory design](docs/memory-design.md), [read API](docs/memory-read-api.md) and
[release checklist](docs/release-checklist.md) before upgrading.
