# Managed execution and usage reporting

The supported default managed runtime uses PostgreSQL, Redis and the existing
OpenAI Responses/embedding models. This is a synchronous library: a host with an
async event loop should invoke the synchronous client from a host worker thread.

## Initialization and ownership

`Tomte.init(db, cache, llm, max_workers=1, usage_listener=callback)` starts one
background execution slot by default. Set `max_workers=2` to permit two users to
process concurrently. The coordinator submits at most that many futures; the
executor does not accumulate an unbounded work queue. There is no process-global
`schedule` registration. Daily dictionary maintenance starts after local work drains
and uses a database-wide dictionary lock. Subsequent consolidation may overlap it:
dictionary feedback is optional and cannot hold back a valid memory commit.

Managed database operations and background attempts create an independent
PostgreSQL connection. Model agents, SDK client and event loop are created lazily,
only when model resources are used. Database reads and marker-only ACK recovery
therefore do not require SDK construction or valid LLM credentials. Guest messages
and assistant history/queue writes use only Redis, without opening DB/model resources.
Owned SDK sockets close on their owning loop, including partial-construction failure.
The initialized database/provider objects are
templates owned by the caller; shutdown does not close caller-owned templates.
Redis's normal connection pool is shared, but pipelines are operation-local.

Repeated initialization with the same providers and options is idempotent.
Replacing providers/options requires successful shutdown first. Workflows built
directly, or explicitly assigned to `client._workflow`, are caller-owned legacy
entry points and do not receive managed concurrency guarantees.

Custom providers, including the alternate Anthropic/Voyage and Gemini adapters,
must supply `operation_factory(control)`: a context manager yielding `(db, llm)`.
It must create independent resources, close them on their owning thread/loop,
provide database transaction/marker/user-guard contracts, bound network waits and
SDK retries, and honor cooperative control before/after calls. A custom cache must
be safe for concurrent calls. The runtime does not silently share uncertain SDK
clients. Standalone alternate-provider behavior remains available; live parity and
automatic ownership factories for those adapters are not claimed.

## Ordering, retries and recovery

The frozen unit remains a session batch. Redis and PostgreSQL serialize execution
by `(tenant_id, user_id)`. A per-user lock **does not guarantee chronological
ordering across sessions**. Strict interleaving needs a separate queue/provenance
design; it is intentionally deferred.

Redis holds one active batch state per user, a due-time index and the existing
frozen source snapshot. Terminal outcomes contain batch ID, attempt count, safe
reason code and outcome; no conversation text. Successful terminal metadata expires
after seven days; skipped-batch tombstones do not expire and block replay of a
reintroduced snapshot. PostgreSQL commit markers retain existing durability semantics.

States are NEW, RUNNING, WAIT, RECONCILE, ACK and SKIP. Recognized transient
failures get at most three total attempts, delayed 30 and 120 seconds. No worker
thread sleeps through those delays. Later batches for the same user wait; other
users can proceed. Invalid output, exhausted retries and other definitive
failures are recorded as skipped and automatically removed from the frozen queue.
Later queued messages are preserved. There is no automatic replay of skipped
batches. Manually deleting tombstones or reconstructing sources under new IDs
invalidates replay protection and is unsupported. Redis persistence/loss remains an
operational dependency; skipped-batch tombstones are not duplicated into PostgreSQL.

Every recovery obtains an independent PostgreSQL connection and the per-user
advisory lock, then reads the scoped batch commit marker before model work.
Committed batches get only an ACK. Unknown commit outcomes remain blocked for
that user until PostgreSQL can answer. A recovered RUNNING state with no commit
marker and no recorded classified failure is conservatively skipped, not replayed.
Redis state/ACK/SKIP operations compare the lease token atomically; finalization
also compares batch ID and the exact frozen list. A replaced owner cannot ACK or
release a successor's work. The PostgreSQL lock prevents a replacement from writing
while an old execution still holds its connection, even if its Redis lease expired.

Lease TTL is 120 seconds; the coordinator renews active leases every 30 seconds.
Renewal failure stops further cooperative stages. Busy database owners and
infrastructure failures receive a short scheduling cooldown to allow other users
to run. Cross-process claims/leases are supported; integration tests cover the
underlying separate-connection fences, not process-kill fault injection.

## Timeouts, foreground fallback and shutdown

Native connection establishment is bounded to five seconds, SQL statements to
15 seconds and SQL lock waits to two seconds. Redis socket/connect waits are two
seconds. Native managed OpenAI HTTP waits use 30 seconds and SDK retries are zero;
managed structured-output validation retries are also zero. The worker owns the
bounded retry policy. Legacy standalone agents keep their existing retry setting.

Operation control checks run around model stages and immediately before the
memory transaction. Background operation deadlines default to 300 seconds;
foreground model-stage control uses 60 seconds. These are cooperative deadlines,
not a guarantee of thread termination at exactly that wall-clock instant. A current
blocking call must return or hit its own timeout. Python threads cannot safely be
forcibly killed. The foreground client returns an error-only payload on memory
failure; the host should continue the conversation using its original message.
It does not wait for background consolidation.

`Tomte.shutdown(timeout=5)` requests stop and returns a boolean. `False` means
work/resources or a listener remain alive; retain the runtime and retry shutdown.
One monotonic deadline covers the shutdown wait. Redis I/O does not hold the
coordinator state mutex; a blocked scan cannot block the shutdown caller, and a
stop is rechecked at task admission. The polling operation may remain alive after
False until its blocking call returns. No new task is admitted after stop.
The running coordinator continues renewing outstanding leases while draining.
Do not replace providers, close their resources or claim completion after False.
Executor threads can keep the Python process alive until their running calls exit.

## Dictionary safety

Maintenance holds the global dictionary advisory lock across its read/model/write
cycle, without leaving a SQL transaction open during model calls. Compliant
dictionary writes take the corresponding transaction-level lock. Validated memory
and its replay marker commit atomically first. Prepared dictionary feedback then
uses its own transaction and the existing bounded lock wait. Feedback failure
(including contention or a deleted suggested anchor) rolls back only that optional
transaction, logs a sanitized warning and sets the workflow's
`last_dictionary_feedback` to `failed`. It cannot undo memory or repeat the LLM
pipeline. Mandatory provenance, identity and memory validation still fail closed.
There is no durable feedback queue: a crash after memory commit can omit feedback;
marker reconciliation ACKs that batch without replaying it.
Overlapping/duplicate synonym groups, unoffered words and non-finite/out-of-range
confidence are rejected. SQL checks that merge members/survivors still exist before
deleting any rows. Direct external SQL is outside this coordination contract.

## Usage API

`Tomte.usage(tenant_id, user_id)` returns scoped model/operation-kind totals and a pending
estimate. Omitting scope returns host-level totals; callers must enforce their own
authorization before exposing that administrative view. Observations are in-memory
and reset on shutdown/reinitialization. They are not a billing ledger.

Actual response usage comes from the SDK, including input/output and cache tokens.
Each response is counted once, even with the optional diagnostic probe enabled.
Missing response usage is unknown. Totals expose nullable complete totals, known
partial totals and explicit unknown counters. SDK pricing produces estimated
dollars where available; unknown pricing remains null. No fallback prices, model
catalog, token defaults, spending gate or caller price configuration is added.
Completeness is tracked per token field: an unknown cache count does not hide
known input/output counts. The installed OpenAI text adapter loses cache-write
metadata and defaults that count to zero. Tomte conservatively reports that
ambiguous zero as null, with incomplete usage and unknown cost. Positive SDK
cache-write counts remain usable; embedding observations are unaffected. This
does not recover raw metadata discarded by the SDK or prove a true zero from it.
An unobserved transport failure records an unknown observation, not a fabricated
successful request count. Provider invoices remain authoritative.

Pending tokens/cost use the average per input character of this user's completed,
usable consolidation observations. The first usable observation enables estimates
immediately. Before that, nonempty pending work has a null estimate; an empty queue
is zero. Running/reconciling work is excluded from pending. Estimates are rough:
context size, branching, cached input, failures and future model behavior can vary.

The query also returns `running_batches`, `reconciling_batches` (including ACK
recovery), `in_flight_requests`, and `in_flight_usage_unknown`. These process-local
fields reflect locally observed worker states and active provider methods. They
distinguish pending-empty but unfinished work from idle; they never add guessed
tokens or cost for an unfinished request. They are not a cluster-wide liveness
monitor. After restart, outstanding Redis states become visible when the local
worker observes them. Provider failure without a response remains unknown usage.
Stages such as extraction versus mutation are available in the optional diagnostic
probe; usage totals retain the simpler foreground/consolidation/maintenance kinds.

The optional callback receives metadata only, through a bounded 256-event queue
and a separate daemon dispatcher. Slow listeners cannot block memory processing;
queue overflow and listener exceptions have visible counters. Accumulated totals
are updated before notification, so dropping notifications does not drop totals.
Callbacks should return promptly. Prompt/source text, vectors, credentials and
exception bodies are not included.

## Explicit capacity boundary

`max_model_input_bytes` (environment `MAX_MODEL_INPUT_BYTES`) defaults to 65,536
UTF-8 bytes. This conservative local technical limit is checked on the complete
frozen extraction input before any model stage and on native OpenAI provider inputs
before dispatch, including explicit static prompts and structured schema text.
The built-in Anthropic and Gemini adapters share this same pre-dispatch guard.
Embedding batch text is checked in aggregate. This is not an exact token count,
provider context-window guarantee, or promise of long-context support. Provider
framing, generated output and model-specific limits can still cause rejection.
Custom providers must enforce their own downstream assembled-input limits; the
workflow's frozen-input guard still applies. No model or embedding model changed.

Oversize input raises permanent `CapacityExceeded`, recorded as `capacity_exceeded`
by the worker. The existing exact-snapshot SKIP policy records a tombstone and
advances to later queued work; it does not truncate sources, split a batch, partially
ACK, or retry an impossible request. Earlier stages of a growing pipeline may have
already run before a later assembled input exceeds the limit. Increasing the limit
requires reviewing the chosen provider's capacity; chunking and broader retention
policy remain out of scope. Existing public message-length rejection is unchanged.

## Deferred semantics

Historical assertion behavior (F3), ambiguous relationship links (F4), mixed
contextual corrections (F5), complete erasure (N1), and broader privacy scrubbing
(N5) retain their existing semantics pending owner decisions. The existing deletion
method covers durable memory and persona cache only; it is not complete erasure.
No independent evaluation ground truth is changed by these repairs.
