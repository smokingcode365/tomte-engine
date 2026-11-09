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

    def rebuild_retrieval_index(self, user_id: str, *, refresh_summary: bool = False) -> int:
        """Backfill plain-text vectors; optionally refresh the shared summary vector.

        Makes embedding calls. Stop application writers for a rollout. Each entity
        replacement is atomic; errors propagate, and the operation is safe to rerun.
        """
        from tomte.engine.clients.chat_client import TomteChatClient
        from tomte.engine.llm.embedder import get_entity_summary, get_retrieval_entries, embed_texts
        from tomte.engine.exceptions import TomteSecurityViolationError
        TomteChatClient(self.tenant_id)._validate_user(user_id)
        if type(refresh_summary) is not bool:
            raise ValueError("refresh_summary must be a boolean")
        rebuilt = 0
        with Tomte.operation_context() as (db, llm):
            with db.user_guard(self.tenant_id, user_id) as acquired:
                if not acquired:
                    raise TomteError("User memory is busy; retry the index rebuild later")
                for row in db.get_user_graph_skeleton(self.tenant_id, user_id):
                    if (row.get('tenant_id'), row.get('user_id')) != (self.tenant_id, user_id):
                        raise TomteSecurityViolationError("Retrieval rebuild namespace mismatch")
                    entries = get_retrieval_entries(row['payload'])
                    texts = [e['text'] for e in entries]
                    if refresh_summary:
                        texts.insert(0, get_entity_summary(row['payload']))
                    vectors = embed_texts(llm, texts)
                    summary_option = {'summary_embedding': vectors[0]} if refresh_summary else {}
                    if refresh_summary:
                        vectors = vectors[1:]
                    embedded = [{**{k:v for k,v in e.items() if k != 'text'}, 'embedding':v}
                                for e,v in zip(entries,vectors)]
                    db.replace_retrieval_index(self.tenant_id, user_id, row['id'], row['payload'], embedded,
                                               **summary_option)
                    rebuilt += 1
        return rebuilt

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
