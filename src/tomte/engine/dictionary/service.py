# tomte/engine/dictionary/service.py

import math
from typing import List, Dict, Any, Set, Tuple

from pydantic import BaseModel
from uuid import UUID

from tomte.engine.core.config import settings
from tomte.engine.core.enums import DictionaryKeywordType
from tomte.engine.dictionary import logic
from tomte.engine import Tomte, tomtelog
from tomte.engine.llm.interfaces import LLMProvider
from tomte.engine.llm.prompts.dictionary_maintenance import (
    CROSS_ANCHOR_MERGE_PROMPT,
    INTERNAL_KEYWORD_MERGE_PROMPT,
)


class SynonymGroup(BaseModel):
    words: List[str]
    confidence_score: float


class InternalMergeResponse(BaseModel):
    groups: List[SynonymGroup]


class DictionaryService:
    def __init__(self):
        self.db = Tomte.get_db()

    def bootstrap(self, chat_vectors: List[List[float]]) -> Dict[DictionaryKeywordType, List[str]]:
        """
        Phase 1: Topic grouping -> Concept Lookup -> Tiered Bucketing -> Final Truncation.
        """
        centroids = logic.calculate_topic_centroids(chat_vectors, eps=0.5)

        tiered_keywords: Dict[str, Dict[tuple, int]] = {
            "high": {},
            "mid": {},
            "low": {},
        }

        for c in centroids:
            res = self.db.get_semantic_keywords(
                c,
                settings.llm_threshold_low,
                id_limit=20,
            )

            for item in res:
                unique_key = (item["keyword"], item["keyword_type"])
                hits = item["hits"]
                score = item["score"]

                if score >= settings.llm_threshold_high:
                    tier = "high"
                elif score >= settings.llm_threshold_mid:
                    tier = "mid"
                else:
                    tier = "low"

                if unique_key in tiered_keywords["high"] and tier != "high":
                    continue
                if unique_key in tiered_keywords["mid"] and tier == "low":
                    continue

                if unique_key not in tiered_keywords[tier] or hits > tiered_keywords[tier][unique_key]:
                    tiered_keywords[tier][unique_key] = hits

        final_entities: List[str] = []
        final_attributes: List[str] = []
        safe_limit = settings.max_keywords_for_llm

        for tier in ["high", "mid", "low"]:
            sorted_tier = sorted(
                tiered_keywords[tier].items(), key=lambda x: x[1], reverse=True
            )

            for k, hits in sorted_tier:
                keyword_str, keyword_type = k

                if keyword_type == DictionaryKeywordType.ENTITY.value:
                    if len(final_entities) < safe_limit:
                        final_entities.append(keyword_str)

                elif keyword_type == DictionaryKeywordType.ATTRIBUTE.value:
                    if len(final_attributes) < safe_limit:
                        final_attributes.append(keyword_str)

        return {
            DictionaryKeywordType.ENTITY: final_entities,
            DictionaryKeywordType.ATTRIBUTE: final_attributes,
        }

    def apply_feedback(self, refined_ops: List[Dict[str, Any]]) -> None:
        """Persist prepared focused evidence; never infer synonymy from similarity.

        Joins the memory/claim transaction. No embedding or model calls occur here.
        Missing/deleted accepted anchors abort atomically rather than redirecting
        feedback to a different concept. Stage fallback supplies no evidence.
        """
        with self.db.transaction():
            for op in refined_ops:
                for evidence in op.get("_dictionary_evidence", []):
                    kind = evidence["keyword_type"]
                    anchor = evidence["dictionary_id"]
                    vector = evidence["vector"]
                    if anchor is None:
                        anchor = self.db.create_dictionary_anchor(vector)
                    else:
                        details = self.db.get_dictionary_anchor_details(anchor)
                        if not details:
                            raise ValueError("Accepted dictionary anchor no longer exists")
                        self.db.update_dictionary_anchor_vector(anchor, logic.nudge_vector(
                            details["embedding"], vector, details["total_hits"]))
                    self.db.upsert_dictionary_keyword(anchor, evidence["keyword"], kind)

    def dictionary_trim(self) -> int:
        return self.db.execute_dictionary_trim()

        
    # ==========================================
    # CROSS-ANCHOR MERGE
    # ==========================================
    def cross_anchor_merge(self) -> set[Any]:
        tomtelog.info("Executing Cross-Anchor Semantic Merge Job...")
        pairs = self.db.get_cross_anchor_merge_candidates()
        processed_ids: set[Any] = set()

        for pair in pairs:
            id_a, id_b = pair["id_a"], pair["id_b"]
            if id_a in processed_ids or id_b in processed_ids:
                continue

            kw_a = self.db.get_anchor_keywords(str(id_a))
            kw_b = self.db.get_anchor_keywords(str(id_b))

            words_a = [k["keyword"] for k in kw_a]
            words_b = [k["keyword"] for k in kw_b]
            prompt = f"Group A: {words_a}\nGroup B: {words_b}\nAre these exact synonyms?"

            is_synonym = Tomte.get_llm().parse_structured(
                system_prompt=CROSS_ANCHOR_MERGE_PROMPT,
                user_text=prompt,
                schema=bool,
            )

            if is_synonym:
                self._calculate_and_commit_cross_merge(
                    str(id_a), pair["emb_a"], kw_a, str(id_b), pair["emb_b"], kw_b
                )
                processed_ids.add(id_b)
                processed_ids.add(id_a)
        return processed_ids

    def _calculate_and_commit_cross_merge(
        self,
        id_a: str,
        emb_a: Any,
        kw_a: List[Dict[str, Any]],
        id_b: str,
        emb_b: Any,
        kw_b: List[Dict[str, Any]],
    ) -> None:
        hits_a = sum(k["hits"] for k in kw_a) or 1
        hits_b = sum(k["hits"] for k in kw_b) or 1

        v_merged: List[float] = []
        for x, y in zip(emb_a, emb_b):
            v_merged.append((float(x) * hits_a + float(y) * hits_b) / (hits_a + hits_b))

        magnitude = math.sqrt(sum(val**2 for val in v_merged))
        v_merged = [val / magnitude for val in v_merged] if magnitude > 0 else list(emb_a)

        merged_kws: Dict[Any, Dict[str, Any]] = {}
        for k in kw_a + kw_b:
            key = (k["keyword"], k["keyword_type"])
            if key not in merged_kws:
                merged_kws[key] = {
                    "hits": k["hits"],
                    "created_at": k["created_at"],
                    "last_modified_at": k["last_modified_at"],
                }
            else:
                merged_kws[key]["hits"] = max(merged_kws[key]["hits"], k["hits"])
                merged_kws[key]["created_at"] = min(
                    merged_kws[key]["created_at"], k["created_at"]
                )
                merged_kws[key]["last_modified_at"] = max(
                    merged_kws[key]["last_modified_at"], k["last_modified_at"]
                )

        if self.db.consolidate_duplicate_anchors(id_a, id_b, v_merged, merged_kws):
            tomtelog.info(f"Successfully merged Duplicate Anchor {id_b} into {id_a}")

    def internal_anchor_optimization(self) -> None:
        tomtelog.info("Executing Internal Anchor Semantic Merge Job...")
        # 1. Fetch candidates (Read transaction opens and closes instantly)
        anchors = self.db.get_anchors_for_internal_merge()
        
        for anchor_id in anchors:
            # 2. Fetch keywords for this specific anchor
            kws = self.db.get_anchor_keywords(str(anchor_id))
            
            if len(kws) > 1:
                # 3. Network Call to LLM (Safe to take 3 seconds, DB is not locked!)
                merges = self._analyze_internal_synonyms(kws)
                
                if merges:
                    # 4. Atomic DB Commit (Opens and closes instantly)
                    self.db.apply_internal_merges(str(anchor_id), merges)
                    continue

            # 5. If no merges were found or kws < 2, just mark the cooldown timer
            self.db.mark_anchor_internal_merge_complete(str(anchor_id))


    def _analyze_internal_synonyms(
        self, raw_kws: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """LLM-driven synonym grouping; results are applied by the DB provider callback."""
        approved_merges: List[Dict[str, Any]] = []

        types = {k["keyword_type"] for k in raw_kws}
        for kw_type in types:
            words_of_type = [k for k in raw_kws if k["keyword_type"] == kw_type]
            if len(words_of_type) < 2:
                continue

            prompt = f"Keywords: {[w['keyword'] for w in words_of_type]}"

            response: InternalMergeResponse = Tomte.get_llm().parse_structured(
                system_prompt=INTERNAL_KEYWORD_MERGE_PROMPT,
                user_text=prompt,
                schema=InternalMergeResponse,
            )

            for group in response.groups:
                if group.confidence_score > 0.9 and len(group.words) > 1:
                    valid_words = [
                        w for w in words_of_type if w["keyword"] in group.words
                    ]
                    if len(valid_words) > 1:
                        survivor_dict = max(valid_words, key=lambda x: x["hits"])
                        survivor_str = survivor_dict["keyword"]

                        victims = [
                            w["keyword"]
                            for w in valid_words
                            if w["keyword"] != survivor_str
                        ]
                        total_hits = sum(w["hits"] for w in valid_words)

                        approved_merges.append(
                            {
                                "kw_type": kw_type,
                                "survivor": survivor_str,
                                "victims": victims,
                                "total_hits": total_hits,
                            }
                        )

        return approved_merges


_dictionary_service_instance: DictionaryService | None = None


def get_dictionary_service() -> DictionaryService:
    """Returns the shared :class:`DictionaryService` instance for the process."""
    global _dictionary_service_instance
    if _dictionary_service_instance is None:
        _dictionary_service_instance = DictionaryService()
    return _dictionary_service_instance
