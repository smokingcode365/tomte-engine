# Tomte V1 local demo

This checkout-only demonstrator hosts a conversation with a real configured LLM.
Tomte supplies memory and queues evidence; it does **not** generate the host reply.
There is no login, remote deployment, tool execution or production-readiness claim.
Use fictional data only. Run one process/browser per profile; never expose it on
a network or run a daemon alongside manual consolidation.

## Run

From the repository root, follow [local setup](../README.md#local-development):
Python 3.10+, the existing uv environment, PostgreSQL/pgvector and Redis. Current
development verification uses Python 3.12. Keep the existing embedding dimension.
Create `.env` from `.env.example` only if it does not exist. Configure local
`POSTGRES_URL`, `REDIS_URL` and a private `OPENAI_API_KEY`; never commit them.
Keep session isolation enabled. The demo refuses nonlocal database/cache URLs or
PII scrubbing that could download extra models. It does not install or bootstrap.

```powershell
docker compose up -d --wait
uv sync --locked
uv run python scripts/bootstrap_local.py --vector-dimension 1536
uv run python -m demo.app --profile playground
```

Open **http://127.0.0.1:8765/** (use this exact host). Alternatively use the existing
environment directly: `.venv\Scripts\python.exe -m demo.app --profile playground`.
Use `--port 8766` if necessary. No new package is required.

The profile selects `tomte-demo-<profile>` / `demo-user`. Letters, digits and
hyphens only, max 32 characters. Reusing that profile retains its durable memory
across restarts. Use a fresh fictional profile for a clean demonstration; this
does not delete earlier data. Sessions get independent random IDs.

## A short demonstration

1. Send: “I own a blue bicycle named Willow.” The host replies. Inspect the
   query/background groups: they may be empty before any consolidation.
2. Click **Consolidate session**. Wait for a confirmed processed marker and ACK.
   The read-only inspector should show the accepted active facts. LLM extraction
   may vary; inspect what actually persisted rather than assuming an expected row.
3. Click **New session**, then ask: “What color is Willow?” The new conversation
   has separate history but can retrieve the same user's durable memory.
4. Send: “I painted Willow red.” Consolidate, then start a third session and ask:
   “What color is Willow now?” Refresh inspector and check the canonical entity's
   active assertion. Do not claim success if the model abstains or stores a wrong fact.

Use the session selector to return to and consolidate pending sessions. Stop only
after consolidating the sessions you want saved: UI session selection is in memory
and is not restored across process restarts, although Redis queues are retained.
No autonomous recovery/scheduler runs in this demo.

## Boundaries and failures

- The host uses current-only structured query and background context, plus returned
  sanitized message/history. Memory and chat text are marked untrusted, without tools.
- Assistant output is logged with the assistant role. Suggestions are not independent
  user evidence. Sending is not a durable-memory commit.
- Empty consolidation, lock busy, confirmed no-memory-change, retry, and committed
  but ACK-pending are distinct statuses. Failures keep frozen data for retry. ACK
  applies only to the supplied snapshot, not newly queued messages.
- The adapter reuses provider lock/freeze/ACK methods and the existing consolidation
  workflow and private `_batch_id` helper. It never polls global idle sessions or
  calls broad maintenance/wipe APIs. The existing 120-second lock lease is retained;
  run one instance per profile. PostgreSQL replay/transaction guarantees remain
  engine-owned. This is not a new distributed-worker design.
- Host failures may leave a user message queued. Logging failure displays a generated
  answer explicitly as **not confirmed logged**. No automatic resend or fabricated reply.
- Inspector is read-only and scoped to the selected demo profile. Only ACTIVE facts
  appear; canonical UUID, aliases and source/evidence metadata can be inspected.
- Nearest-neighbor false positives, occasional LLM mistakes, incomplete timelines/
  relationships and stale related facts remain known limitations. Historical/corrected/
  retracted facts are not exposed as current, and full history answers are unsupported.

Same-origin/token and Host checks reduce accidental browser cross-site access; they
are not authentication. Anyone trusted on this machine can access the local demo.
API/provider errors and request headers are not logged. No secrets are shown in UI.

## Stop and verification

Ctrl+C stops the server and closes provider connections; it does not wipe memory,
queues or volumes. Stop infrastructure separately with `docker compose stop` if
desired. Never use `down --volumes` as routine demo cleanup: it deletes all local data.
There is deliberately no deletion endpoint or global reset button. Synthetic profiles
can be removed later with an explicitly reviewed scoped cleanup, not a truncate/flush.

```powershell
uv run pytest tests/test_demo.py tests/test_extraction_contract.py -q
```

Demo tests use local deterministic doubles and a loopback HTTP smoke check. They
verify orchestration and failure semantics, not real LLM extraction quality. No
benchmark or paid API call runs in this command.
