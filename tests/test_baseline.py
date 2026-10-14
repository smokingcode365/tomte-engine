import subprocess
import sys
from types import SimpleNamespace

import pytest


def test_import_tomte_and_engine_in_fresh_process():
    # Isolated mode excludes PYTHONPATH and checks the installed package.
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", "import tomte; from tomte.engine import Tomte"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_initialize_without_daemon(monkeypatch):
    from tomte.engine import Tomte, TomteWorker

    for name in ("_db", "_cache", "_llm", "_worker"):
        monkeypatch.setattr(Tomte, name, None)

    def unexpected_start(self):
        pytest.fail("Initialization must not start the daemon")

    monkeypatch.setattr(TomteWorker, "start_in_background", unexpected_start)
    db, cache, llm = SimpleNamespace(), SimpleNamespace(), SimpleNamespace()
    Tomte.init(db, cache, llm, start_daemon_worker=False)

    assert Tomte.get_db() is db
    assert Tomte.get_cache() is cache
    assert Tomte.get_llm() is llm
    assert isinstance(Tomte.get_worker(), TomteWorker)
    assert Tomte.create_tomte_client("test-tenant").tenant_id == "test-tenant"


def test_create_client_requires_initialization(monkeypatch):
    from tomte.engine import Tomte

    for name in ("_db", "_cache", "_llm", "_worker"):
        monkeypatch.setattr(Tomte, name, None)
    with pytest.raises(RuntimeError, match="Tomte not initialized"):
        Tomte.create_tomte_client("test-tenant")
