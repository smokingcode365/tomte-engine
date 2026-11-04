"""Focused dictionary retrieval and fail-open, immutable terminology decisions."""
from copy import deepcopy
from dataclasses import dataclass
import json
import math
from time import perf_counter
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tomte.engine import tomtelog
from tomte.engine.core.config import settings
from tomte.engine.llm.prompts.terminology_alignment import TERMINOLOGY_ALIGNMENT_PROMPT

CANDIDATE_LIMIT = 5


class VocabularyDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity_handle: str
    decision: Literal["KEEP_RAW", "REUSE"]
    candidate_handle: str | None = None

    @model_validator(mode="after")
    def check_candidate(self):
        if (self.decision == "REUSE") != (self.candidate_handle is not None):
            raise ValueError("REUSE requires a candidate; KEEP_RAW forbids one")
        return self


class AttributeDecision(VocabularyDecision):
    fact_handle: str


class FieldDecision(AttributeDecision):
    field_handle: str


class FieldAlignmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    field_decisions: list[FieldDecision] = Field(default_factory=list)


class TerminologyAlignmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    entity_type_decisions: list[VocabularyDecision] = Field(default_factory=list)
    attribute_decisions: list[AttributeDecision] = Field(default_factory=list)


@dataclass
class AlignmentResult:
    entities: list[dict]
    evidence: list[list[dict]]
    metrics: dict


def query_text(domain, entity_type, attribute=None):
    text = f"domain: {domain}\nentity_type: {entity_type}"
    return text if attribute is None else text + f"\nattribute: {attribute}"


def _embed_queries(llm, texts, metrics):
    from tomte.engine.llm.embedder import EMBED_BATCH_SIZE, embed_texts
    vectors = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[start:start + EMBED_BATCH_SIZE]
        metrics["embedding_calls"] += 1
        metrics["embedding_texts"] += len(batch)
        vectors.extend(embed_texts(llm, batch))
    return vectors


def apply_decisions(raw, targets, response):
    """Validate every target first; an invalid response cannot apply partially."""
    response = TerminologyAlignmentResponse.model_validate(
        response.model_dump(mode="json") if isinstance(response, BaseModel) else response)
    selected = {}
    for kind, decisions in (("ENTITY", response.entity_type_decisions),
                            ("ATTRIBUTE", response.attribute_decisions)):
        for decision in decisions:
            key = (decision.entity_handle, getattr(decision, "fact_handle", None))
            if key not in targets or key in selected or targets[key]["keyword_type"] != kind:
                raise ValueError("Unknown, duplicate or wrong-type terminology target")
            target = targets[key]
            candidate = None
            if decision.decision == "REUSE":
                candidate = next((c for c in target["candidates"]
                                  if c["candidate_handle"] == decision.candidate_handle), None)
                if candidate is None or candidate["keyword_type"] != kind:
                    raise ValueError("Terminology candidate was not offered for this target/type")
            selected[key] = candidate
    aligned = deepcopy(raw)
    evidence = [[] for _ in raw]
    for key, target in targets.items():
        candidate = selected.get(key)
        i, j = target["entity_index"], target["fact_index"]
        keyword = candidate["keyword"] if candidate else target["raw_keyword"]
        if j is None:
            aligned[i]["entity_type"] = keyword
        else:
            aligned[i]["facts"][j]["attribute_name"] = keyword
        evidence[i].append({"keyword": keyword, "keyword_type": target["keyword_type"],
                            "vector": target["vector"],
                            "dictionary_id": candidate["dictionary_id"] if candidate else None})
    return aligned, evidence


def align_terms(raw, current_batch, llm, db):
    """All network work precedes persistence. Failure retains the entire extraction.

    Feedback is prepared only when retrieval/decision validation completed. On a
    stage failure memory still proceeds, with no speculative dictionary writes.
    """
    started = perf_counter()
    metrics = {"entities": len(raw), "facts": sum(len(e["facts"]) for e in raw),
               "embedding_texts": 0, "embedding_calls": 0, "candidate_lookups": 0,
               "terminology_calls": 0, "raw_fallback": False}
    try:
        sources = {item["item_id"]: item["content"] for item in current_batch}
        targets = {}
        view = []
        for i, entity in enumerate(raw):
            eh = f"entity_{i}"
            entry = {"entity_handle": eh, "entity_name": entity["entity_name"],
                     "domain": entity["entity_domain"], "raw_entity_type": entity["entity_type"],
                     "candidates": [], "facts": []}
            view.append(entry)
            targets[(eh, None)] = {"entity_index": i, "fact_index": None,
                "keyword_type": "ENTITY", "raw_keyword": entity["entity_type"],
                "text": query_text(entity["entity_domain"], entity["entity_type"]), "view": entry}
            for j, fact in enumerate(entity["facts"]):
                fh = f"fact_{i}_{j}"
                # Provenance has already been resolved; unknown sources fail closed
                # for dictionary alignment, without dropping the user's extraction.
                source = sources[fact["source_chat_id"]]
                fact_view = {"entity_handle": eh, "fact_handle": fh,
                    "raw_attribute_name": fact["attribute_name"], "value_read_only": fact["value"],
                    "source_text": [source], "candidates": []}
                entry["facts"].append(fact_view)
                targets[(eh, fh)] = {"entity_index": i, "fact_index": j,
                    "keyword_type": "ATTRIBUTE", "raw_keyword": fact["attribute_name"],
                    "text": query_text(entity["entity_domain"], entity["entity_type"], fact["attribute_name"]),
                    "view": fact_view}
        if not targets:
            return AlignmentResult(deepcopy(raw), [], metrics)
        texts = [target["text"] for target in targets.values()]
        vectors = _embed_queries(llm, texts, metrics)
        if len(vectors) != len(texts):
            raise ValueError("Dictionary embedding count mismatch")
        for index, (target, vector) in enumerate(zip(targets.values(), vectors)):
            if (len(vector) != llm.get_vector_dimension() or
                    not all(math.isfinite(x) for x in vector) or not any(vector)):
                raise ValueError("Invalid dictionary query vector")
            target["vector"] = list(vector)
            metrics["candidate_lookups"] += 1
            rows = db.get_semantic_keywords(vector, settings.llm_threshold_low, id_limit=20,
                                            keyword_type=target["keyword_type"], keyword_limit=CANDIDATE_LIMIT)
            candidates = []
            for row in rows:
                if row["keyword_type"] != target["keyword_type"]:
                    continue  # Defend the provider boundary as well as SQL filtering.
                if row["score"] <= settings.llm_threshold_low:
                    continue
                candidates.append({"candidate_handle": f"candidate_{index}_{len(candidates)}",
                    "dictionary_id": str(row["dictionary_id"]), "keyword": row["keyword"],
                    "keyword_type": row["keyword_type"], "similarity": row["score"]})
                if len(candidates) == CANDIDATE_LIMIT:
                    break
            target["candidates"] = candidates
            # Similarity and storage UUIDs are not semantic proof and stay internal.
            target["view"]["candidates"] = [
                {k: c[k] for k in ("candidate_handle", "keyword")} for c in candidates]
        metrics["terminology_calls"] = 1
        response = llm.parse_structured(system_prompt=TERMINOLOGY_ALIGNMENT_PROMPT,
            user_text=json.dumps({"entities": view}, ensure_ascii=False), schema=TerminologyAlignmentResponse)
        aligned, evidence = apply_decisions(raw, targets, response)
        aligned, field_evidence = align_value_fields(aligned, current_batch, llm, db, metrics=metrics)
        for existing, fields in zip(evidence, field_evidence):
            existing.extend(fields)
        return AlignmentResult(aligned, evidence, metrics)
    except Exception as exc:
        from tomte.engine.workers.control import WorkStopped
        from tomte.engine.core.capacity import CapacityExceeded
        if isinstance(exc, (WorkStopped, CapacityExceeded)):
            raise
        metrics["raw_fallback"] = True
        # Never log payloads, source text, values, provider exceptions or credentials.
        try:
            tomtelog.warn(f"Terminology alignment RAW fallback: {type(exc).__name__}")
        except Exception:
            pass
        return AlignmentResult(deepcopy(raw), [[] for _ in raw], metrics)
    finally:
        metrics["latency_seconds"] = perf_counter() - started


def field_context(entity, fact):
    """Parent identity is structural; the model decides equivalence of vocabulary."""
    return json.dumps([entity["entity_domain"], entity["entity_type"], fact["attribute_name"]], ensure_ascii=False, separators=(",", ":"))


def align_value_fields(raw, current_batch, llm, db, *, metrics=None):
    """Second pass uses canonical parent terms. All renames validate before application."""
    from tomte.engine.core.memory import field_key
    if metrics is None:
        metrics = {"embedding_calls": 0, "embedding_texts": 0, "candidate_lookups": 0, "terminology_calls": 0}
    targets = {}
    sources = {item["item_id"]: item["content"] for item in current_batch if item["role"] == "user"}
    for i, entity in enumerate(raw):
        for j, fact in enumerate(entity["facts"]):
            context = field_context(entity, fact)
            for n, (name, item) in enumerate(fact["value"].items()):
                key = (f"entity_{i}", f"fact_{i}_{j}", f"field_{n}")
                targets[key] = {"i": i, "j": j, "name": name, "context": context,
                    "view": {"entity_handle": key[0], "fact_handle": key[1], "field_handle": key[2],
                        "parent_context": context, "field_name": name, "value_read_only": item,
                        "source_text": sources[fact["source_chat_id"]], "candidates": []}}
    evidence = [[] for _ in raw]
    if not targets:
        return deepcopy(raw), evidence
    vectors = _embed_queries(llm, [t["context"] + " field: " + t["name"] for t in targets.values()], metrics)
    for index, (target, vector) in enumerate(zip(targets.values(), vectors)):
        target["vector"] = vector
        metrics["candidate_lookups"] += 1
        rows = db.get_semantic_keywords(vector, settings.llm_threshold_low, id_limit=20,
            keyword_type="FIELD", keyword_limit=CANDIDATE_LIMIT, keyword_context=target["context"])
        eligible = [row for row in rows if row["keyword_type"] == "FIELD"
                    and row.get("keyword_context") == target["context"] and row["score"] > settings.llm_threshold_low]
        target["candidates"] = {f"candidate_{index}_{n}": row for n, row in enumerate(eligible[:CANDIDATE_LIMIT])}
        target["view"]["candidates"] = [{"candidate_handle": handle, "keyword": row["keyword"]}
            for handle, row in target["candidates"].items()]
    metrics["terminology_calls"] += 1
    response = llm.parse_structured(system_prompt=TERMINOLOGY_ALIGNMENT_PROMPT,
        user_text=json.dumps({"fields": [t["view"] for t in targets.values()]}, ensure_ascii=False), schema=FieldAlignmentResponse)
    response = FieldAlignmentResponse.model_validate(response.model_dump(mode="json") if isinstance(response, BaseModel) else response)
    selected = {}
    for decision in response.field_decisions:
        key = (decision.entity_handle, decision.fact_handle, decision.field_handle)
        if key not in targets or key in selected:
            raise ValueError("Unknown or duplicate value-field terminology target")
        candidate = targets[key]["candidates"].get(decision.candidate_handle)
        if decision.decision == "REUSE" and candidate is None:
            raise ValueError("Value-field candidate was not offered for this parent")
        selected[key] = candidate
    aligned = deepcopy(raw)
    values = {}
    for key, target in targets.items():
        candidate = selected.get(key)
        name = field_key(candidate["keyword"] if candidate else target["name"])
        location = (target["i"], target["j"])
        value = values.setdefault(location, {})
        if name in value:
            raise ValueError("Value-field alignment would collapse distinct fields")
        value[name] = deepcopy(raw[target["i"]]["facts"][target["j"]]["value"][target["name"]])
        evidence[target["i"]].append({"keyword": name, "keyword_type": "FIELD", "keyword_context": target["context"],
            "attribute_name": raw[target["i"]]["facts"][target["j"]]["attribute_name"],
            "vector": target["vector"], "dictionary_id": str(candidate["dictionary_id"]) if candidate else None})
    for (i, j), value in values.items():
        aligned[i]["facts"][j]["value"] = value
    return aligned, evidence
