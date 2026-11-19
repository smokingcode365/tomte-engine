# Changelog

## 2.0.0 — 2026-11-19

The [November baseline release](https://github.com/smokingcode365/tomte-engine/releases/tag/v2.0.0)
replaces the ambiguous use of package version 1.0.0 on November development commits.
The original October v1.0.0 tag/release remains unchanged. The release page names the
exact tested commit and CI run and provides wheel/source assets with checksums.
See [verification](docs/release-verification.md) and [upgrade requirements](docs/upgrading-from-1.0.md).

The major version reflects breaking changes already present between the October
release and November source. This release finalization adds no new engine feature
or SAR implementation and performs no application-data migration.

### Compatibility changes since 1.0.0

- Returned public entities/facts now use canonical UUIDs, structured text/confidence
  value items and in-fact history. Consumers of the old assertion/scalar shapes
  must adapt; old stored payloads and queues are not automatically converted.
- Custom managed providers require an `operation_factory` supplying independent
  resources. Replacing initialized providers requires successful shutdown first.
- Embedding defaults and derived indexes changed. Existing vectors require a
  deliberate model/index rollout, even when dimensions match.
- Additional storage, pending/read, LLM-assisted retrieval/answering and managed
  runtime capabilities are documented in the current SDK contract.

### Fixed

- Use the explicitly injected OpenAI API key for both standalone models and
  managed operation clients, instead of silently requiring an environment key.
- Validate admin graph-read and memory-wipe tenant/user IDs before opening
  resources or constructing a potentially colliding Redis namespace.
- Let the no-API evaluation preflight inspect the configured embedding model;
  historical live-comparison model pins remain enforced for those live runs.

### Release engineering

- Expand documentation with the specification, architecture overview, testing
  guide and operational runbook; update contribution and security guidance.
- Preserve the MIT license/public attribution and use an evidence-based package description.
- Declare the directly used dotenv dependency and restrict source distributions
  to deliberate runtime/documentation files, excluding local caches/prototypes.
- Remove the unused `schedule` dependency (the worker owns scheduling) and the
  obsolete numeric relevance-threshold example setting and empty Makefile.

### Existing behavior retained

Canonical entity/fact UUIDs, structured values and immutable history; scoped
PostgreSQL persistence and Redis batches; semantic dictionary alignment; session
continuity; pending-memory search; opt-in lexical retrieval; LLM-assisted fact
retrieval and answering; managed workers and a local demonstration application.

### Known limitations

Semantic evidence remains a small development smoke, not a broad accuracy estimate.
Memory wipe remains partial erasure. Production readiness is not established.
See the [release checklist](docs/release-checklist.md) and verification record.

## 1.0.0 — 2026-10-15

Initial public experimental release, preserved at commit
[`393f34f6086abcb6713fc60bb680476947047ad5`](https://github.com/smokingcode365/tomte-engine/commit/393f34f6086abcb6713fc60bb680476947047ad5).
Includes the original Python chat SDK, assertion-based memory, PostgreSQL/Redis
persistence, terminology alignment, synthetic evaluation tooling and local setup.
The [original tag and assets](https://github.com/smokingcode365/tomte-engine/releases/tag/v1.0.0)
remain unchanged. Its source and original changelog remain available at that tag.
Internal development snapshots used version 0.2.0; they are not a separate public
compatibility baseline for the November release.
