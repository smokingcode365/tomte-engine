# tomte/engine/workflows/dictionary_maintenance.py

from tomte.engine import Tomte
from tomte.engine.dictionary.service import get_dictionary_service
from tomte.engine.log import tomtelog


class MaintenanceWorkflow:
    """Thin workflow: delegates dictionary maintenance to :class:`DictionaryService`."""

    def __init__(self) -> None:
        self._dictionary = get_dictionary_service()


    def run_midnight_cleanup(self):
        """Executes Phase 5: Midnight Cleanup."""
        tomtelog.info("Starting Phase 5: Midnight Cleanup Workflow")
        
        # Job 1: Pruning
        rows_trimmed =self._dictionary.dictionary_trim()
        tomtelog.info(f"Trim Job complete. Moved {rows_trimmed} stale keywords.")
        
        # Job 2: Cross-Anchor Vector Consolidation
        processed_ids = self._dictionary.cross_anchor_merge()
        tomtelog.info(f"Cross-Anchor Merge Job complete. Processed {processed_ids} anchors.")
        
        # Job 3: Internal Anchor Keyword Consolidation
        self._dictionary.internal_anchor_optimization()
        tomtelog.info("Internal Anchor Optimization Job complete.")

