"""Transaction mechanics shared by standalone operations and business scopes.

Nested Tomte scopes join the outer scope, without savepoints or early commits.
A failed child poisons the entire scope even if its exception is caught.
An unmanaged transaction is rejected rather than committed or rolled back here.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
from threading import RLock, get_ident
from weakref import WeakKeyDictionary

import psycopg
from psycopg.pq import TransactionStatus
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector


class TransactionScopeError(RuntimeError):
    """A transaction cannot safely complete under Tomte's ownership policy."""


@dataclass
class _Scope:
    owner: int
    failed: bool = False


_scopes = WeakKeyDictionary()
_scope_lock = RLock()


def connect(connection_string: str) -> psycopg.Connection:
    """Create an IDLE provider connection; close it if vector setup fails."""
    conn = psycopg.connect(connection_string, autocommit=True, row_factory=dict_row)
    try:
        register_vector(conn)
    except BaseException:
        conn.close()
        raise
    return conn


def in_transaction(conn: psycopg.Connection) -> bool:
    with _scope_lock:
        return conn in _scopes


def close(conn: psycopg.Connection) -> None:
    """Close a provider connection only after its owned scopes have finished."""
    with _scope_lock:
        if conn in _scopes:
            raise TransactionScopeError("Exit the transaction scope before closing its connection")
        conn.close()


@contextmanager
def transaction(conn: psycopg.Connection):
    """Own a transaction or join this thread's existing Tomte transaction."""
    with _scope_lock:
        scope = _scopes.get(conn)
        joined = scope is not None
        if joined:
            if scope.owner != get_ident():
                raise TransactionScopeError("Connection is in use by another transaction owner")
            if scope.failed:
                raise TransactionScopeError("Transaction has already failed; exit its scope before retrying")
        else:
            if conn.info.transaction_status != TransactionStatus.IDLE:
                raise TransactionScopeError("Unmanaged transaction: finish it before entering a Tomte scope")
            scope = _Scope(get_ident())
            _scopes[conn] = scope
    if joined:
        try:
            yield
        except BaseException:
            scope.failed = True
            raise
        return
    try:
        with conn.transaction():
            yield
            if scope.failed or conn.info.transaction_status == TransactionStatus.INERROR:
                raise TransactionScopeError("A child operation failed; the entire transaction was rolled back")
    finally:
        with _scope_lock:
            del _scopes[conn]


def transactional(operation):
    """One public DB operation owns one scope, unless an outer scope exists."""
    @wraps(operation)
    def execute(conn, *args, **kwargs):
        with transaction(conn):
            return operation(conn, *args, **kwargs)
    return execute
