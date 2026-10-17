"""Compatibility entry points use the same managed ownership boundaries."""
from tomte.engine import Tomte, tomtelog
from tomte.engine.workflows.dictionary_maintenance import MaintenanceWorkflow
from tomte.engine.workers.control import WorkControl


def run_consolidation_job():
    """One bounded scheduling pass; completion is owned by the worker handle."""
    Tomte.get_worker().poll()


def run_midnight_job():
    """Explicit maintenance is exclusive in PostgreSQL, even across processes."""
    try:
        return Tomte.get_worker()._maintenance(WorkControl())
    except Exception:
        tomtelog.error("Dictionary maintenance failed")
        return False

