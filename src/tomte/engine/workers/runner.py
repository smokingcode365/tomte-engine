# tomte/engine/worker/runner.py

import time
import schedule
import threading 
from tomte.engine import Tomte, tomtelog
from tomte.engine.workflows.consolidation import ConsolidationWorkflow
from tomte.engine.workflows.dictionary_maintenance import MaintenanceWorkflow
from tomte.engine.utils.keys import generate_session_ref
from tomte.engine.workers.dictionary_maintenance_worker import run_consolidation_job, run_midnight_job

class TomteWorker:
    def __init__(self, idle_timeout_seconds: int = 300):
        self.idle_timeout_seconds = idle_timeout_seconds
        try:
            Tomte.get_cache()
            Tomte.get_db()
            Tomte.get_llm()
        except RuntimeError as e:
            raise RuntimeError(f"Cannot start TomteWorker: {e} Call Tomte.init() first.")

    # ... (Keep _run_consolidation_job and _run_midnight_job exactly the same) ...

    def start(self):
        """Starts the infinite scheduler loop (BLOCKING)."""
        tomtelog.info("Starting Tomte Engine Worker Daemon (Blocking Mode)...")
        schedule.every(1).minutes.do(self._run_consolidation_job)
        schedule.every().day.at("00:00").do(self._run_midnight_job)
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            tomtelog.info("Tomte Engine Worker Stopped gracefully.")

    def start_in_background(self) -> threading.Thread:
        """
        Starts the worker in a background daemon thread (NON-BLOCKING).
        This is the "fire and forget" method. The thread will automatically 
        die when the main application (e.g., FastAPI) shuts down.
        """
        tomtelog.info("Starting Tomte Engine Worker in Background Thread...")
        
        # daemon=True means "Do not keep the main program alive just for this thread"
        worker_thread = threading.Thread(target=self.start, daemon=True)
        worker_thread.start()
        
        return worker_thread

    def _run_consolidation_job(self):
        run_consolidation_job()

    def _run_midnight_job(self):
        run_midnight_job()