# tomte/engine/clients/admin_client.py

from typing import List, Dict, Any
from tomte.engine import Tomte, tomtelog
from tomte.engine.workflows.dictionary_maintenance import MaintenanceWorkflow
from tomte.engine.exceptions import TomteError

class TomteAdminClient:
    """
    SDK Facade for administrative control, manual trigger overrides, and graph inspection.
    """
    def __init__(self, tenant_id: str):
        self.tenant_id = tenant_id

    def force_midnight_cleanup(self) -> bool:
        """
        Manually triggers Phase 5 Dictionary Maintenance outside of the scheduled worker loop.
        Useful for cleaning up the dictionary immediately after heavy load testing.
        """
        tomtelog.info(f"Admin triggered manual dictionary cleanup for Tenant: {self.tenant_id}")
        try:
            from tomte.engine.workers.dictionary_maintenance_worker import run_midnight_job
            return run_midnight_job()
        except Exception as e:
            tomtelog.error("Manual cleanup failed")
            return False

    def get_user_graph_raw(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Dumps the entire raw JSONB memory payload for a specific user.
        Useful for debugging or rendering a visual Knowledge Graph in an admin dashboard.
        """
        try:
            with Tomte.operation_context() as (db, _):
                return db.get_user_graph_skeleton(tenant_id=self.tenant_id, user_id=user_id)
        except TomteError as e:
            tomtelog.error("Failed to fetch graph")
            return []

    def wipe_user_memory(self, user_id: str) -> bool:
        """
        Deletes durable memory and the legacy persona cache only.
        This is not a complete erasure API; queue/history erasure is deferred.
        """
        try:
            cache = Tomte.get_cache()
            with Tomte.operation_context() as (db, _):
                with db.user_guard(self.tenant_id, user_id) as acquired:
                    if not acquired:
                        return False
                    db.delete_user_memory(self.tenant_id, user_id)

            # Clear Persona Cache
            from tomte.engine.utils.keys import generate_persona_key
            cache.client.delete(generate_persona_key(self.tenant_id, user_id))
            
            tomtelog.info("Deleted durable memory and persona cache; other queued/history data remains")
            return True
        except Exception as e:
            tomtelog.error("Failed to delete durable memory")
            return False
