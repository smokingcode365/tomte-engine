# PostgreSQL transaction ownership

## Policy

`database/postgre/transactions.py` owns connection creation, transaction entry,
commit/rollback, and connection cleanup. Provider connections use autocommit so
vector registration and incidental catalog inspection do not open abandoned
transactions. Public SQL helpers use `@transactional`: each complete operation
opens one transaction, including reads, unless a Tomte scope already exists.
Cursor contexts remain inside those helpers and close on success or failure.

Use `with db.transaction():` to group several provider operations. Only the
outermost scope commits or rolls back. Nested scopes join without savepoints;
an exception in any child marks the entire unit for rollback, even if caught.
Further operations in a failed scope are rejected until it exits. A transaction
opened outside this API is rejected rather than silently committed or rolled
back. Overlapping owners on the same connection are rejected; use separate
provider connections for concurrent units of work. Call `db.close()` after all
scopes finish.

The old manual `commit_transaction()`/`rollback_transaction()` methods have been
replaced by the scoped API. Callers must not use raw connection commits or
rollbacks inside owned scopes.

## Audit and business boundaries

Before this change:

- `PostgresProvider` created a connection with autocommit disabled; vector
  registration could itself leave a transaction open.
- Most memory and dictionary SQL helpers owned cursors but did not finish their
  transactions. Heartbeat and bulk memory updates committed/rolled back internally.
- Selected provider reads committed manually. Maintenance callbacks committed
  batches and sometimes swallowed errors after rollback.
- Dictionary feedback used a manual final commit/rollback. Consolidation's
  memory batch committed before dictionary feedback, permitting partial success.
- Schema's core and dictionary halves committed separately.
- Admin user deletion used provider internals and raw SQL, with no rollback path.
- The bootstrap preflight used a separate autocommit, read-only connection.

Now:

- Chat Hook heartbeat and retrieval calls are independent short operations.
- Consolidation performs model calls before a single scope for its prepared
  memory changes and dictionary feedback. Feedback joins that scope, or owns
  one when called independently. PostgreSQL failure propagates and rolls back
  the unit; model decision logic is unchanged.
- Cross-anchor merges retain one transaction per merge; internal keyword merge
  batches include their cooldown marker. Trim deletion and audit insertion
  remain one atomic CTE. Candidate reads finish before model calls.
- Full schema setup joins its core and dictionary operations into one transaction.
- User memory/registry deletion uses a scoped DB helper; Redis cache deletion
  remains separate and is not covered by PostgreSQL atomicity.
- Bootstrap preflight remains read-only; provider cleanup delegates to the DB layer.

Standalone graph/cross-anchor write callbacks retain their boolean failure
results. Inside an outer scope they propagate failures. Heartbeat and internal
maintenance failures now propagate rather than silently pretending success.

## Verification

Unit tests run without infrastructure. Real transaction tests require the
existing local PostgreSQL schema with 1536-dimensional vectors and local `POSTGRES_URL` in the
environment or `.env`:

```powershell
$env:TOMTE_RUN_POSTGRES_TESTS = '1'
uv run pytest
```

Tests use unique synthetic tenant/user IDs, read committed visibility through a
second connection, and remove only their own user rows through the provider.
Synthetic dictionary writes are rolled back. They never start workers or call
external models. They do not test multi-worker locking, PostgreSQL/Redis atomicity,
or consolidation decision quality.
