from tomte.engine import Tomte, tomtelog
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from tomte.engine.workflows.dictionary_maintenance import MaintenanceWorkflow
from tomte.engine.utils.keys import generate_session_ref

# ==========================================
# JOB 1: THE EVERY-MINUTE CONSOLIDATION LOOP
# ==========================================
def run_consolidation_job():
    tomtelog.debug("Waking up: Polling for idle chat sessions...")

    redis_provider = Tomte.get_cache()
    consolidation_workflow = ConsolidationWorkflow()
    
    while True:
        session_tuple = redis_provider.get_next_idle_session_4_consolidation(idle_seconds=300)
        
        if not session_tuple:
            break
            
        tenant_id, user_id, session_id = session_tuple
        
        lock_key = f"Tomte:lock:consolidation:{tenant_id}:{user_id}"
        lock_value = generate_session_ref(tenant_id, session_id, user_id) 
        
        acquired = redis_provider.acquire_lock(lock_key, lock_value, ttl_seconds=120)
        
        if acquired:
            try:
                chat_items = redis_provider.pop_session_chat_log_4_consolidation(tenant_id, user_id, session_id)
                
                if chat_items:
                    tomtelog.info(f"Consolidating {len(chat_items)} messages for user {user_id}")
                    success = consolidation_workflow.run_consolidation_cycle(tenant_id, user_id, chat_items)
                    if success:
                        redis_provider.clear_session_4_consolidation(tenant_id, user_id, session_id)
                else:
                    redis_provider.clear_session_4_consolidation(tenant_id, user_id, session_id)
                    
            except Exception as e:
                tomtelog.error(f"Consolidation crashed for user {user_id}: {e}")
            finally:
                redis_provider.release_lock(lock_key, lock_value)
        else:
            break


# ==========================================
# JOB 2: THE MIDNIGHT DICTIONARY CLEANUP
# ==========================================
def run_midnight_job():
    tomtelog.info("Waking up: Executing Phase 5 Midnight Maintenance...")
    
    maintenance_workflow = MaintenanceWorkflow()    
    try:
        maintenance_workflow.run_midnight_cleanup()
        tomtelog.info("Midnight Maintenance complete. Going back to sleep.")
    except Exception as e:
        tomtelog.error(f"Midnight Maintenance crashed: {e}")

