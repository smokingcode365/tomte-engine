"""Explicit local table setup; run only after Docker's extension initialization."""

import argparse
import logging
import os
from pathlib import Path
import sys

from dotenv import dotenv_values
import psycopg
from psycopg.conninfo import make_conninfo


class BootstrapError(Exception):
    """A safe, actionable error that contains no connection details."""


def vector_dimension(value: str) -> int:
    try:
        dimension = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError("Vector dimension must be an integer.") from None
    # The existing schema uses vector IVFFlat indexes, limited to 2,000 dimensions.
    if not 1 <= dimension <= 2000:
        raise argparse.ArgumentTypeError("Vector dimension must be between 1 and 2000.")
    return dimension


def bootstrap(connection_string: str, dimension: int) -> None:
    # Catalog inspection must precede PostgresProvider's vector registration.
    with psycopg.connect(
        connection_string,
        connect_timeout=5,
        autocommit=True,
        options="-c default_transaction_read_only=on",
    ) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pgcrypto')")
            missing = {"vector", "pgcrypto"} - {row[0] for row in cursor.fetchall()}
            if missing:
                raise BootstrapError(
                    "Missing prerequisite extensions: " + ", ".join(sorted(missing))
                    + ". Provision extensions in the target database before bootstrap."
                )
            for table in ("tomte_memory_engine", "tomte_semantic_anchors"):
                cursor.execute(
                    "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                    "WHERE attrelid = to_regclass(%s) AND attname = 'embedding' "
                    "AND attnum > 0 AND NOT attisdropped",
                    (table,),
                )
                column = cursor.fetchone()
                if column and column[0] != f"vector({dimension})":
                    raise BootstrapError(
                        f"Existing {table}.embedding does not match the selected dimension. "
                        "No schema changes were made; plan a migration or explicit local reset."
                    )

    from tomte.engine.providers.postgres_provider import PostgresProvider

    provider = PostgresProvider(make_conninfo(connection_string, connect_timeout=5))
    try:
        provider.ensure_infrastructure(dimension)
    finally:
        provider.conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vector-dimension", required=True, type=vector_dimension)
    arguments = parser.parse_args()

    # Existing python-dotenv is already supplied by project dependencies.
    # Explicit process environment takes precedence over the repository's .env.
    try:
        config = dotenv_values(Path(__file__).resolve().parents[1] / ".env")
        connection_string = os.environ.get("POSTGRES_URL", config.get("POSTGRES_URL"))
    except Exception:
        print("Bootstrap failed: unable to read local configuration.", file=sys.stderr)
        return 1
    if not connection_string or not connection_string.strip():
        print("Bootstrap failed: set POSTGRES_URL in the environment or .env.", file=sys.stderr)
        return 1
    if "URL_ENCODED_LOCAL_PASSWORD" in connection_string:
        print("Bootstrap failed: replace the example URL password.", file=sys.stderr)
        return 1

    # Existing provider logs may contain connection metadata. Emit only safe
    # diagnostics from this setup script, including on authentication failures.
    previous_logging = logging.root.manager.disable
    logging.disable(logging.CRITICAL)
    try:
        bootstrap(connection_string, arguments.vector_dimension)
    except BootstrapError as error:
        print(f"Bootstrap failed: {error}", file=sys.stderr)
        return 1
    except psycopg.OperationalError:
        print("Bootstrap failed: PostgreSQL unavailable or authentication failed; check service health and credentials.", file=sys.stderr)
        return 1
    except Exception:
        print("Bootstrap failed: check POSTGRES_URL, database permissions, and schema compatibility. No data was deleted; setup may be partially complete.", file=sys.stderr)
        return 1
    finally:
        logging.disable(previous_logging)
    print("Tomte tables and indexes provisioned for the selected vector dimension.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
