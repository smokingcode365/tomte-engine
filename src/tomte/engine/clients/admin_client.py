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
            workflow = MaintenanceWorkflow()
            workflow.run_midnight_cleanup()
            return True
        except Exception as e:
            tomtelog.error(f"Manual cleanup failed: {e}")
            return False

    def get_user_graph_raw(self, user_id: str) -> List[Dict[str, Any]]:
        """
        Dumps the entire raw JSONB memory payload for a specific user.
        Useful for debugging or rendering a visual Knowledge Graph in an admin dashboard.
        """
        try:
            db = Tomte.get_db()
            # Fetch the skeleton, which includes the payload
            return db.get_user_graph_skeleton(tenant_id=self.tenant_id, user_id=user_id)
        except TomteError as e:
            tomtelog.error(f"Failed to fetch graph for user {user_id}: {e}")
            return []

    def wipe_user_memory(self, user_id: str) -> bool:
        """
        Hard-deletes a user from the memory engine and cache. 
        Crucial for GDPR/CCPA compliance.
        """
        try:
            db = Tomte.get_db()
            cache = Tomte.get_cache()
            
            db.delete_user_memory(self.tenant_id, user_id)

            # Clear Persona Cache
            from tomte.engine.utils.keys import generate_persona_key
            cache.client.delete(generate_persona_key(self.tenant_id, user_id))
            
            tomtelog.info(f"Successfully wiped all memory and cache for user {user_id}")
            return True
        except Exception as e:
            tomtelog.error(f"Failed to wipe memory for {user_id}: {e}")
            return False