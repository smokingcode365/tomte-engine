# Release candidate verification (2026-10-14)

This is the unpublished v1.0.0 candidate, derived from the approved private
source snapshot with fresh public history planned separately. The owner approved
version 1.0.0, MIT, copyright (c) 2026 Sheng Guo, and public contact
smokingcode365@gmail.com. Historical results below describe their own earlier
candidate states and are not claims that earlier artifacts were published.

## TCX-041 package and offline verification

Windows, CPython 3.12.13, uv 0.11.2; a new environment was installed using the
unchanged lockfile. Offline suite: **567 collected, 402 passed, 165 skipped,
0 failed, 0 errors, 1 warning**. Skips: 110 combined PostgreSQL/Redis cases,
38 PostgreSQL cases and 17 Redis cases, all explicitly opt-in. The warning is
Pydantic-AI's event-loop deprecation warning. Five new evaluator context fixtures
passed as part of that suite; no frozen labels or retrieval metrics were retuned.

The first sandboxed test attempt reported failures/errors and stalled before
complete diagnostics; it was interrupted. The successful rerun used writable
isolated temporary files and loopback permission, without code/assertion changes.
The initial failure causes were not fully captured and are not proven.

Wheel and source distribution were built with isolated Hatchling. The wheel
includes Python prompt modules and the MIT notice. The sdist includes source,
tests, docs, synthetic evaluation fixtures, the then-included HTTP demo, and setup/CI files.
Neither artifact includes private audit records, `.env`, virtual environments,
runtime logs, bytecode or original Git history. The source logging package is
included deliberately.

A second fresh environment installed locked runtime dependencies and the wheel
non-editably. An isolated-mode import and guest SDK ingestion smoke passed from
installed site-packages with fake infrastructure providers, no path override,
no model calls and no worker threads. Dependency consistency check passed.

## TCX-042 fresh-service verification

On Windows CPython 3.12.13, uniquely named disposable PostgreSQL/Redis projects
bootstrapped from empty volumes. PostgreSQL 17.11 supplied vector 0.8.7 and
pgcrypto 1.3; all six Tomte tables and both vector(1536) columns were verified.
Repeating bootstrap preserved table identities and a synthetic registry canary.
Redis 7.4.11 was healthy with AOF everysec and periodic snapshots disabled.
Existing Docker containers, volumes and data were not used or changed.

Service-enabled suite: **567 collected, 564 passed, 3 failed, 0 skipped,
0 errors, 1 warning**. Two fixtures lacked explicit current-USER alias evidence;
one test still expected background-first text. These were test-contract failures,
not an approved green release result. TCX-043 corrects those fixtures/expectations
without changing alias authorization or retrieval behavior.

The noneditable wheel also passed a registered-user SDK smoke on separate real
disposable services with a local fake LLM. It verified message ownership/source
IDs, retrieval of an explicitly seeded synthetic memory, tenant/user isolation
and fresh-session history. This was not real-model extraction/consolidation.
Scoped cleanup restored DB/Redis fingerprints; both test projects were removed.

Installed CPython 3.13.12 additionally passed the offline suite: 402 passed,
165 service-gated skips, no failures, one warning. No new runtime was installed.
The fixture validator loaded 27 dev/13 validation cases; conversation validation
loaded 12 synthetic traces / 6 users / 648 turns, without paid calls.

## TCX-043 test-contract corrections and verification

Alias fixtures now explicitly approve aliases with current USER source IDs and
exact naming quotes. The helper forwards only supplied alias citations; no
citation is inferred. The retrieval regression checks query-first ordering,
both section bodies, no overlap and full fixture coverage. Production executable
logic and alias authorization are unchanged; two source docstrings were corrected.

Focused service-enabled regressions: **50 passed, 0 failed, 0 skipped**.
The complete service-enabled suite ran twice on fresh disposable projects:
**567 collected, 566 passed, 1 failed, 0 skipped, 0 errors, 1 warning** each time.
All three TCX-042 test-contract failures passed in both runs. The remaining failure
is the unchanged demo HTTP origin/token test: Windows error 10053 while expecting
an HTTP 403 response. Its isolated recheck passed (1 passed). Root cause is not
confirmed; this is a separate release blocker, not a green full-suite result.
No assertion was weakened or failure marked as expected to hide it.

Disposable projects were removed. Existing containers and volumes were preserved.
The first run observed a default-bridge ID change; its cause was not
established. The second run's original-resource metadata matched before/after.
No existing service was started, reset or used for testing; prior evidence remains
private and unchanged. No paid APIs or background worker were used.

The existing TCX-041 wheel/sdist are preserved historical evidence and predate
these test/documentation changes. TCX-045 rebuilds final-version artifacts
separately; do not publish the old source archive
as though it includes TCX-043.

## Remaining verification and release decisions

- Remote Linux GitHub Actions and Python 3.10 have not run; 3.10 is not installed.
- No real model evaluation, optional security extras or alternative-provider
  compatibility checks were run as part of release hardening.
- Historical TCX-027/028 reproduction remains limited by undistributed evidence;
  see [evaluation report](evaluation.md). No universal accuracy is claimed.
- Attribution/provenance owner review remains necessary before publication.
  TCX-045 now sets the approved release version and documents the private email
  reporting route; GitHub private reporting remains unconfigured/unverified.

Passing deterministic tests does not establish production readiness. No public
repository, release, commit or tag has been created by release hardening.

## TCX-044 repository scope

The optional local HTTP application, its adapter/UI-only tests and instructions
are excluded from the engine release. Its prior HTTP failure was not fixed by
this scope decision. Engine persistence, authorization, concurrency, worker ACK,
lifecycle, current-fact projection and public SDK verification remain.
Four existing evaluation-contract tests now live under `tests/` so the existing
CI command collects them. Empty placeholders were removed; package initializers
and uncertain internal APIs were retained. An executable SDK example replaces
the duplicated README snippet. Final cleanup verification is recorded below.

### TCX-044 verification results

The pre-cleanup offline baseline collected 571 cases when the four evaluation
contracts outside `tests/` were explicitly included: 406 passed, 165 skipped,
zero failures/errors. After removing the 12 HTTP-app-only cases and moving the
four evaluation contracts into standard collection, the offline result is
394 passed, 165 skipped. The complete service-enabled `tests/` suite collected
559 and passed all 559, with no skips or failures. Both runs report one existing
Pydantic-AI event-loop deprecation warning. No engine assertion was weakened;
these counts describe coverage scope, not an improvement from deleting tests.
TCX-043's earlier HTTP-only failure remains historical evidence, not a fixed bug.

A uniquely named disposable PostgreSQL 17/pgvector and Redis 7.4 project verified
empty-database bootstrap, vector(1536), and repeat bootstrap preserving schema
object IDs and a synthetic canary. The installed noneditable wheel and executable
SDK example passed ingestion, query/background retrieval, user/tenant isolation,
and a new session using deterministic local embeddings. Durable memory was an
explicit provider fixture; this was not a new live semantic/consolidation study.
No external model request or daemon was used. Scoped SDK cleanup restored all
six table row fingerprints and Redis value fingerprints.

The disposable containers, volumes and network were removed after run-label and
ownership checks. Existing container/volume/image inventories were unchanged;
existing custom networks were unchanged. Docker's built-in `bridge` network ID
and creation time changed during the run, as was also observed in TCX-043. The
cause is unconfirmed; strict all-network preservation verification therefore
failed. No command targeted that built-in network or existing services/data.

Fresh wheel and sdist builds succeed from the cleaned candidate using the locked,
cached toolchain. The wheel includes engine prompts/logging/license resources;
the sdist includes the example, retained evaluations and moved tests. Neither
contains the HTTP application, private evidence, environment secrets or caches.
These new private TCX-044 artifacts supersede the old source contents for review,
but at TCX-044 still carried version 0.2.0 pending approval (now superseded by
TCX-045). Local verification
uses Windows Python 3.12; remote CI and other declared Python versions remain
unverified. The private development repository and earlier release evidence are
preserved. Nothing has been committed, pushed, tagged or published.

## TCX-045 final v1.0.0 candidate

The owner approved version **1.0.0**, MIT, copyright (c) 2026 Sheng Guo,
and the public identity `smokingcode365/tomte-engine` with contact
smokingcode365@gmail.com. That future repository has not been created by this
work. Public Git history will be initialized independently only after separate
authorization; earlier internal 0.2.0 history is not a public ancestor. Private
checkpoint IDs have been removed from publication-facing evidence descriptions,
while study IDs, metrics and reproducibility limits remain unchanged.

Only release metadata and documentation changed. All engine modules, prompts,
tests, examples, infrastructure, evaluation cases and dependency constraints are
unchanged from TCX-044. The lockfile changes only the local package version;
`uv lock --check --offline` and `uv sync --locked --offline` pass without changing
any dependency version. An initial offline lock refresh lacked uncached metadata
for other Python-version branches; retaining the existing dependency graph and
updating the local version avoided unrelated resolution or upgrades.

The final-version offline suite collected **559**: **394 passed, 165 skipped,
0 failed, 0 errors**, with one existing Pydantic-AI event-loop deprecation warning
(64.19 seconds). Service-dependent cases are opt-in; TCX-044's **559/559** isolated
service result is prior evidence, not a repeated TCX-045 run. No Docker command,
service creation, database operation or paid model evaluation was required here.
Remote Linux/Python CI matrix runs still have not been verified.

Fresh **tomte_engine-1.0.0-py3-none-any.whl** and
**tomte_engine-1.0.0.tar.gz** build from the candidate. A new noneditable wheel
environment passes isolated imports, metadata/license/resource checks,
`uv pip check`, and the executable SDK example's help. A network-forbidden smoke
uses deterministic in-memory providers to verify initialization with no worker,
registered context, ordered user/assistant history and queue roles, tenant
isolation and trusted admin inspection. Three local embedding stubs, no external
model calls. This is a packaging/SDK contract check, not fresh real persistence
or semantic evaluation. An initial private fake had incompatible keyword names;
correcting that test fake resolved the smoke failure without engine changes.

README/CHANGELOG define the host SDK, internal storage/workflow boundary, fresh
public-history distinction, partial wipe behavior and explicit migration policy.
SECURITY documents the approved private email route. GitHub private vulnerability
reporting is recommended after repository creation, but remains unconfigured and
unverified. Dependency notices and synthetic-only fixture attribution remain;
automated checks do not establish universal provenance, legal or security safety.

Final archive hashes, the full 146-file manifest and preservation evidence are
recorded in the private TCX-045 report. Archives exclude private evidence, Git
metadata, real environment files, caches and the removed HTTP application. Only
the owner-approved public email appears. Original private source, refs, untracked
prototypes and earlier evidence remain unchanged. No Git repository was
initialized, and nothing was committed, pushed, tagged, created remotely or
published. This candidate awaits owner review and separate publication approval.
