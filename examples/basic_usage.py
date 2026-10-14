"""Minimal SDK host example. Registered ingestion uses a paid embedding request.

Run after local infrastructure/bootstrap. No HTTP server or daemon is started.
"""
import argparse
import os
from pathlib import Path
from uuid import uuid4


def send_example(db, cache, llm, tenant_id, user_id, session_id):
    """Inject providers and ingest one synthetic message through the public SDK."""
    from tomte.engine import Tomte

    Tomte.init(db, cache, llm, start_daemon_worker=False)
    client = Tomte.create_tomte_client(tenant_id)
    payload = client.send_user_message(session_id, "I own a blue bicycle.", user_id=user_id)
    if payload.get("error"):
        raise RuntimeError("SDK ingestion failed; inspect private local diagnostics.")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    from dotenv import load_dotenv
    from tomte.engine.providers.postgres_provider import PostgresProvider
    from tomte.engine.providers.redis_provider import RedisCacheProvider
    from tomte.engine.providers.openai_provider import OpenAILLMProvider

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    required = ("POSTGRES_URL", "REDIS_URL", "OPENAI_API_KEY")
    if any(not os.environ.get(name) for name in required):
        parser.error("Set POSTGRES_URL, REDIS_URL and OPENAI_API_KEY privately first.")
    db = PostgresProvider(os.environ["POSTGRES_URL"])
    cache = None
    try:
        cache = RedisCacheProvider(os.environ["REDIS_URL"])
        payload = send_example(db, cache, OpenAILLMProvider(api_key=os.environ["OPENAI_API_KEY"]),
                               "sdk-example-" + uuid4().hex, "synthetic-user", "session-one")
        print(payload["context_text"] or "No existing memories. Synthetic message queued; worker disabled.")
    finally:
        if cache is not None:
            cache.client.close()
        db.close()


if __name__ == "__main__":
    main()
