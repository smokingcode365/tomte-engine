"""Versioned stage evaluation contracts; no database or live provider creation."""
from collections import Counter
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from itertools import combinations
import ast
import json
import logging
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Literal
from unittest.mock import patch

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tomte.engine.core.extraction import ExtractionMemorySchema
from tomte.engine.core.resolution import ResolutionResponse, Evidence
from tomte.engine.core.contextual import ContextualActionResponse
from tomte.engine.core.assertions import MutationPlanResponse
from tomte.engine.dictionary.alignment import TerminologyAlignmentResponse
from tomte.engine.dictionary.service import InternalMergeResponse
from tomte.engine.llm.prompts.extractor import EXTRACTOR_SYSTEM_PROMPT
from tomte.engine.llm.prompts.junk_filter import JUNK_FILTER_PROMPT
from tomte.engine.llm.prompts.consolidation_golden_record import RECONCILER_SYSTEM_PROMPT
from tomte.engine.llm.prompts.contextual_action import CONTEXTUAL_ACTION_PROMPT
from tomte.engine.llm.prompts.fact_mutation import MUTATION_SYSTEM_PROMPT
from tomte.engine.llm.prompts.terminology_alignment import TERMINOLOGY_ALIGNMENT_PROMPT
from tomte.engine.llm.prompts.dictionary_maintenance import CROSS_ANCHOR_MERGE_PROMPT, INTERNAL_KEYWORD_MERGE_PROMPT

VERSION = "1.0"
ROOT = Path(__file__).resolve().parents[1]
STAGES = {
    "junk": (JUNK_FILTER_PROMPT, bool),
    "extraction": (EXTRACTOR_SYSTEM_PROMPT, ExtractionMemorySchema),
    "terminology": (TERMINOLOGY_ALIGNMENT_PROMPT, TerminologyAlignmentResponse),
    "identity_resolution": (RECONCILER_SYSTEM_PROMPT, ResolutionResponse),
    "contextual_action": (CONTEXTUAL_ACTION_PROMPT, ContextualActionResponse),
    "fact_mutation": (MUTATION_SYSTEM_PROMPT, MutationPlanResponse),
    "cross_anchor_merge": (CROSS_ANCHOR_MERGE_PROMPT, bool),
    "internal_keyword_merge": (INTERNAL_KEYWORD_MERGE_PROMPT, InternalMergeResponse),
}
Stage = Literal["junk", "extraction", "terminology", "identity_resolution", "contextual_action", "fact_mutation", "cross_anchor_merge", "internal_keyword_merge"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class JunkExpected(Strict):
    expected_junk: bool


class CrossExpected(Strict):
    expected_merge: bool


class TermRule(Strict):
    entity_handle: str = Field(min_length=1)
    fact_handle: str | None = None
    expected_decision: Literal["KEEP_RAW", "REUSE"]
    expected_candidate_handle: str | None = None

    @model_validator(mode="after")
    def candidate(self):
        if (self.expected_decision == "REUSE") != (self.expected_candidate_handle is not None):
            raise ValueError("REUSE requires a candidate; KEEP_RAW forbids one")
        return self


class TermExpected(Strict):
    decisions: list[TermRule]


class IdentityRule(Strict):
    extracted_entity_id: str = Field(min_length=1)
    expected_kind: Literal["NEW", "UNRESOLVED", "MATCH_ONE", "MATCH_MANY"]
    required_target_ids: list[str] = Field(default_factory=list)
    forbidden_target_ids: list[str] = Field(default_factory=list)
    allowed_evidence_labels: list[Evidence] | None = None
    allow_additional_targets: bool = False
    explicitly_new: bool | None = None
    evidence_quote_contains: str | None = None
    exact_evidence_quote: str | None = None

    @model_validator(mode="after")
    def shape(self):
        size = len(self.required_target_ids)
        if self.expected_kind == "MATCH_ONE" and size != 1:
            raise ValueError("MATCH_ONE expectation requires one target")
        if self.expected_kind == "MATCH_MANY" and size < 2:
            raise ValueError("MATCH_MANY expectation requires at least two targets")
        if self.expected_kind in {"NEW", "UNRESOLVED"} and size:
            raise ValueError("NEW/UNRESOLVED cannot require targets")
        if len(set(self.required_target_ids)) != size or set(self.required_target_ids) & set(self.forbidden_target_ids):
            raise ValueError("Duplicate or conflicting target expectations")
        return self


class IdentityExpected(Strict):
    resolutions: list[IdentityRule]


class CountRange(Strict):
    minimum: int = Field(ge=0)
    maximum: int = Field(ge=0)

    @model_validator(mode="after")
    def order(self):
        if self.minimum > self.maximum:
            raise ValueError("Reversed count range")
        return self


class ActionRule(Strict):
    action_type: Literal["CORRECTION", "RETRACTION", "CONFIRMATION", "CLARIFICATION"] | None = None
    required_target_db_id: str | None = None
    required_target_fact_id: str | None = None
    require_unresolved: bool = False
    required_source_ids: list[str] = Field(default_factory=list)
    forbidden_source_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def target(self):
        if self.require_unresolved and (self.required_target_db_id or self.required_target_fact_id):
            raise ValueError("Unresolved actions cannot require targets")
        return self


class ActionExpected(Strict):
    expected_action_count: int | CountRange = Field()
    required_actions: list[ActionRule] = Field(default_factory=list)
    require_no_action: bool = False

    @model_validator(mode="after")
    def count(self):
        count = self.expected_action_count
        if isinstance(count, int) and count < 0:
            raise ValueError("Negative action count")
        maximum = count if isinstance(count, int) else count.maximum
        if len(self.required_actions) > maximum or (self.require_no_action and maximum != 0):
            raise ValueError("Conflicting action expectations")
        return self


class MutationRule(Strict):
    target_key: str | None = None
    operation: Literal["ADD", "REINFORCE", "CORRECT", "SUPERSEDE", "RETRACT"] | None = None
    target_fact_id: str | None = None
    attribute_name: str | None = None
    new_value: str | None = None
    required_source_ids: list[str] = Field(default_factory=list)
    forbidden_source_ids: list[str] = Field(default_factory=list)
    allowed_evidence_types: list[Literal["USER_EXPLICIT", "RESOLVED_REFERENCE", "GROUP_EXPLICIT", "USER_IMPLIED", "DERIVED"]] | None = None

    @model_validator(mode="after")
    def constrained(self):
        if not self.model_fields_set:
            raise ValueError("An operation matcher needs constraints")
        return self


class MutationExpected(Strict):
    required_operations: list[MutationRule]
    forbidden_operations: list[MutationRule]
    allow_noop: bool
    allow_additional_operations: bool = False
    exact_operation_count: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def count(self):
        if self.exact_operation_count is not None and self.exact_operation_count < len(self.required_operations):
            raise ValueError("Operation count contradicts requirements")
        if not self.allow_noop and self.exact_operation_count == 0:
            raise ValueError("Zero count contradicts no-op prohibition")
        return self


class EntityRule(Strict):
    entity_name: str | None = None
    entity_domain: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None

    @model_validator(mode="after")
    def constrained(self):
        if not self.model_fields_set:
            raise ValueError("An entity matcher needs constraints")
        return self


class FactRule(Strict):
    entity_name: str | None = None
    entity_selector: EntityRule | None = None
    attribute_name: str | None = None
    value: str | None = None
    required_source_chat_id: str | None = None

    @model_validator(mode="after")
    def constrained(self):
        if not self.model_fields_set:
            raise ValueError("A fact matcher needs constraints")
        return self


class ExtractionExpected(Strict):
    required_entities: list[EntityRule]
    forbidden_entities: list[EntityRule]
    required_facts: list[FactRule]
    forbidden_facts: list[FactRule]
    exact_entity_count: int | None = Field(default=None, ge=0)
    exact_fact_count: int | None = Field(default=None, ge=0)
    require_unique_entity_ids: bool = False


class InternalExpected(Strict):
    required_pairs: list[list[str]]
    forbidden_pairs: list[list[str]]

    @model_validator(mode="after")
    def pairs(self):
        for pair in self.required_pairs + self.forbidden_pairs:
            if len(pair) != 2 or pair[0] == pair[1]:
                raise ValueError("Synonym pairs need two distinct words")
        if {tuple(sorted(p)) for p in self.required_pairs} & {tuple(sorted(p)) for p in self.forbidden_pairs}:
            raise ValueError("Conflicting synonym pairs")
        return self


EXPECTATIONS = dict(zip(STAGES, (JunkExpected, ExtractionExpected, TermExpected, IdentityExpected,
                               ActionExpected, MutationExpected, CrossExpected, InternalExpected)))
INPUT_KEYS = {
    "junk": {"CURRENT_BATCH": list, "RECENT_CONTEXT_REFERENCE_ONLY": list},
    "extraction": {"CURRENT_BATCH": list, "RECENT_CONTEXT_REFERENCE_ONLY": list},
    "terminology": {"entities": list},
    "identity_resolution": {"CURRENT_BATCH": list, "REFERENCE_ONLY_CONTEXT": list,
                            "EXTRACTED_SEMANTIC_INFORMATION": list, "CANDIDATES_BY_OBJECT": dict},
    "contextual_action": {"CURRENT_BATCH": list, "REFERENCE_ONLY_CONTEXT": list, "CANDIDATES": list},
    "fact_mutation": {"CURRENT_BATCH": list, "REFERENCE_ONLY_CONTEXT": list,
                      "CANONICAL_ATTRIBUTE_GUIDANCE": dict, "TARGETS": list},
}


def offered_keywords(text):
    if not text.startswith("Keywords: "):
        raise ValueError("Internal merge input must be the exact Keywords: list text")
    words = ast.literal_eval(text[len("Keywords: "):])
    if not isinstance(words, list) or any(not isinstance(w, str) for w in words):
        raise ValueError("Invalid offered keyword list")
    return set(words)


def terminology_targets(data):
    offered = {}
    try:
        for entity in data["entities"]:
            targets = [(entity["entity_handle"], None, entity["candidates"])]
            targets.extend((entity["entity_handle"], f["fact_handle"], f["candidates"]) for f in entity["facts"])
            for entity_handle, fact_handle, candidates in targets:
                key = (entity_handle, fact_handle)
                if not isinstance(entity_handle, str) or (fact_handle is not None and not isinstance(fact_handle, str)) or key in offered:
                    raise ValueError("Invalid/duplicate terminology handle")
                handles = [c["candidate_handle"] for c in candidates]
                if any(not isinstance(h, str) for h in handles) or len(set(handles)) != len(handles):
                    raise ValueError("Invalid/duplicate candidate handle")
                offered[key] = set(handles)
    except (KeyError, TypeError) as exc:
        raise ValueError("Invalid terminology input") from exc
    return offered


class Case(Strict):
    case_id: str = Field(min_length=1)
    schema_version: Literal[1]
    stage: Stage
    partition: Literal["dev", "validation"]
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    tags: list[str]
    input: dict | str
    expected: dict

    @model_validator(mode="before")
    @classmethod
    def version(cls, data):
        if isinstance(data, dict) and type(data.get("schema_version")) is not int:
            raise ValueError("Case schema_version must be integer 1")
        return data

    @model_validator(mode="after")
    def contract(self):
        expectation = EXPECTATIONS[self.stage].model_validate(self.expected)
        canonical(self.input)
        if self.stage in INPUT_KEYS:
            if not isinstance(self.input, dict) or any(
                    key not in self.input or type(self.input[key]) is not kind
                    for key, kind in INPUT_KEYS[self.stage].items()):
                raise ValueError("Invalid stage input envelope")
        elif not isinstance(self.input, str):
            raise ValueError("Maintenance input must be exact user-prompt text")
        if self.stage == "internal_keyword_merge":
            offered_keywords(self.input)
        if self.stage == "terminology":
            offered = terminology_targets(self.input)
            keys = [(r.entity_handle, r.fact_handle) for r in expectation.decisions]
            if len(set(keys)) != len(keys):
                raise ValueError("Duplicate expected terminology target")
            for rule, key in zip(expectation.decisions, keys):
                if key not in offered or (rule.expected_decision == "REUSE" and rule.expected_candidate_handle not in offered[key]):
                    raise ValueError("Expectation must reference offered target/candidate")
        if self.stage == "identity_resolution":
            keys = [r.extracted_entity_id for r in expectation.resolutions]
            if len(set(keys)) != len(keys) or any(k not in self.input["CANDIDATES_BY_OBJECT"] for k in keys):
                raise ValueError("Duplicate or unknown expected resolution object")
            if any(not isinstance(v, list) or any(not isinstance(c, dict) or not isinstance(c.get("id"), str) for c in v)
                   for v in self.input["CANDIDATES_BY_OBJECT"].values()):
                raise ValueError("Invalid identity candidates")
        if self.stage == "fact_mutation":
            targets = self.input["TARGETS"]
            if any(not isinstance(t, dict) or not isinstance(t.get("target_key"), str)
                   or not isinstance(t.get("assertions"), list) or any(not isinstance(f, dict) or not isinstance(f.get("fact_id"), str)
                   for f in t["assertions"]) for t in targets):
                raise ValueError("Invalid mutation targets")
            if len({t["target_key"] for t in targets}) != len(targets):
                raise ValueError("Duplicate input target")
        if self.stage == "contextual_action":
            candidates = self.input["CANDIDATES"]
            if any(not isinstance(c, dict) or not isinstance(c.get("db_id"), str)
                   or not isinstance(c.get("assertions"), list) or any(not isinstance(f, dict) or not isinstance(f.get("fact_id"), str)
                   for f in c["assertions"]) for c in candidates):
                raise ValueError("Invalid contextual candidates")
        return self

    def expectation(self):
        return EXPECTATIONS[self.stage].model_validate(self.expected)


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return sha256(value.encode("utf-8")).hexdigest()


def local_path(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Evaluation files must stay inside tomte-engine")
    return path


def load_cases(paths):
    cases, hashes, ids = [], {}, set()
    for path in sorted({local_path(p) for p in paths}):
        raw = path.read_bytes()
        hashes[path.relative_to(ROOT).as_posix()] = sha256(raw).hexdigest()
        document = json.loads(raw)
        if not isinstance(document, dict) or type(document.get("schema_version")) is not int or document.get("schema_version") != 1 or not isinstance(document.get("cases"), list):
            raise ValueError("Case files require schema_version=1 and cases=[]")
        for item in document["cases"]:
            case = Case.model_validate(item)
            if case.case_id in ids:
                raise ValueError("Duplicate case ID")
            ids.add(case.case_id)
            cases.append(case)
    return cases, hashes


def select_cases(cases, stages=None, case_ids=None, partition=None, limit=None):
    chosen = [c for c in cases if (not stages or c.stage in stages)
              and (not case_ids or c.case_id in case_ids) and (not partition or c.partition == partition)]
    if case_ids and set(case_ids) - {c.case_id for c in cases}:
        raise ValueError("Unknown requested case ID")
    return chosen if limit is None else chosen[:limit]


def prompt_hashes(stages, prompts=None):
    return {s: digest((prompts or {}).get(s, STAGES[s][0])) for s in sorted(stages)}


def schema_hashes(stages):
    return {s: digest(canonical({"type": "boolean"} if STAGES[s][1] is bool
                               else STAGES[s][1].model_json_schema())) for s in sorted(stages)}


def matches(rule, actual, sources="source_chat_ids", evidence="evidence_type"):
    fields = rule.model_dump(exclude_unset=True)
    required = set(fields.pop("required_source_ids", []))
    forbidden = set(fields.pop("forbidden_source_ids", []))
    labels = fields.pop("allowed_evidence_types", None)
    selected = set(actual.get(sources, []))
    return (all(actual.get(k) == v for k, v in fields.items()) and required <= selected
            and not forbidden & selected and (labels is None or actual.get(evidence) in labels))


def assignment(rules, outputs, matcher):
    """Maximum one-to-one matching, so one result cannot satisfy two requirements."""
    owners = {}
    def visit(index, seen):
        for slot, output in enumerate(outputs):
            if slot in seen or not matcher(rules[index], output):
                continue
            seen.add(slot)
            if slot not in owners or visit(owners[slot], seen):
                owners[slot] = index
                return True
        return False
    for index in range(len(rules)):
        visit(index, set())
    return set(owners.values()), set(owners)


def score(case, output):
    """Compare raw model output to frozen invariants; never apply persistence rules."""
    checks = []
    def check(name, passed, code):
        checks.append({"name": name, "passed": bool(passed), "reason_code": None if passed else code})
    schema = STAGES[case.stage][1]
    try:
        if schema is bool:
            if type(output) is not bool:
                raise ValueError("Expected bool")
        else:
            output = schema.model_validate(output).model_dump(mode="json")
        canonical(output)
    except (ValueError, TypeError):
        check("structured_output", False, "INVALID_STRUCTURED_OUTPUT")
        return finish(checks)
    exp = case.expectation()
    stage = case.stage
    if stage == "junk":
        check("junk", output == exp.expected_junk, "WRONG_JUNK_CLASSIFICATION")
    elif stage == "cross_anchor_merge":
        check("merge", output == exp.expected_merge, "FALSE_ANCHOR_MERGE" if output else "MISSED_ANCHOR_MERGE")
    elif stage == "terminology":
        offered = terminology_targets(case.input)
        decisions = {}
        for item in output["entity_type_decisions"] + output["attribute_decisions"]:
            key = (item["entity_handle"], item.get("fact_handle"))
            check("offered_target", key in offered, "UNKNOWN_TARGET")
            check("unique_decision", key not in decisions, "DUPLICATE_TARGET")
            if item["decision"] == "REUSE":
                check("offered_candidate", item["candidate_handle"] in offered.get(key, set()), "WRONG_CANDIDATE")
            decisions[key] = item
        for rule in exp.decisions:
            key = (rule.entity_handle, rule.fact_handle)
            item = decisions.get(key, {"decision": "KEEP_RAW"})
            check("decision", item["decision"] == rule.expected_decision,
                  "FALSE_REUSE" if item["decision"] == "REUSE" else "MISSED_REUSE")
            if rule.expected_decision == "REUSE":
                check("candidate", item.get("candidate_handle") == rule.expected_candidate_handle, "WRONG_CANDIDATE")
    elif stage == "identity_resolution":
        offered = case.input["CANDIDATES_BY_OBJECT"]
        results = {}
        for item in output["resolutions"]:
            key = item["extracted_entity_id"]
            check("offered_object", key in offered, "UNKNOWN_TARGET")
            check("unique_resolution", key not in results, "DUPLICATE_TARGET")
            allowed = {c["id"] for c in offered.get(key, [])}
            for target in item["targets"]:
                check("offered_db_id", target["db_id"] in allowed, "WRONG_RESOLUTION_TARGET")
            if item["kind"] == "UNRESOLVED":
                check("unresolved_targets", not item["targets"], "WRONG_RESOLUTION_TARGET")
            if item["kind"] == "MATCH_MANY":
                check("group_distribution", item["applies_equally_to_all"], "INVALID_GROUP_DISTRIBUTION")
            results[key] = item
        for rule in exp.resolutions:
            item = results.get(rule.extracted_entity_id)
            check("resolution_present", item is not None, "MISSING_RESOLUTION")
            if item is None:
                continue
            check("kind", item["kind"] == rule.expected_kind, "FALSE_NEW" if item["kind"] == "NEW"
                  else "MISSED_NEW" if rule.expected_kind == "NEW" else "WRONG_RESOLUTION_KIND")
            ids = {t["db_id"] for t in item["targets"]}
            for target in rule.required_target_ids:
                check("required_target", target in ids, "MISSING_TARGET")
            for target in ids:
                check("forbidden_target", target not in rule.forbidden_target_ids, "FORBIDDEN_TARGET")
                check("extra_target", rule.allow_additional_targets or target in rule.required_target_ids, "WRONG_RESOLUTION_TARGET")
            if rule.allowed_evidence_labels is not None:
                check("evidence_labels", all(t["evidence"] in rule.allowed_evidence_labels for t in item["targets"]), "WRONG_EVIDENCE_LABEL")
            if rule.explicitly_new is not None:
                check("explicitly_new", item["explicitly_new"] == rule.explicitly_new, "WRONG_NEW_DECLARATION")
            if rule.evidence_quote_contains is not None:
                check("new_quote_contains", rule.evidence_quote_contains in (item["new_entity_evidence"] or ""), "WRONG_NEW_EVIDENCE")
            if rule.exact_evidence_quote is not None:
                check("new_quote", item["new_entity_evidence"] == rule.exact_evidence_quote, "WRONG_NEW_EVIDENCE")
    elif stage == "fact_mutation":
        operations = []
        offered = {t["target_key"]: {f["fact_id"] for f in t["assertions"]} for t in case.input["TARGETS"]}
        seen = set()
        for plan in output["plans"]:
            key = plan["target_key"]
            check("offered_target", key in offered, "WRONG_MUTATION_TARGET")
            check("unique_plan", key not in seen, "DUPLICATE_TARGET")
            seen.add(key)
            for op in plan["operations"]:
                if op["target_fact_id"] is not None:
                    check("offered_fact", op["target_fact_id"] in offered.get(key, set()), "WRONG_MUTATION_TARGET")
                operations.append({"target_key": key, **op})
        matched, used = assignment(exp.required_operations, operations, matches)
        for index, rule in enumerate(exp.required_operations):
            check("required_mutation", index in matched, "MISSING_REQUIRED_MUTATION")
            if index not in matched:
                # Diagnostic reasons compare structured fields only, never language.
                candidates = [o for o in operations if rule.target_key is None or o["target_key"] == rule.target_key]
                if candidates:
                    for field, code in (("operation", "WRONG_MUTATION_TYPE"), ("new_value", "WRONG_MUTATION_VALUE")):
                        if field in rule.model_fields_set:
                            check(field, any(o.get(field) == getattr(rule, field) for o in candidates), code)
                    check("source", any(set(rule.required_source_ids) <= set(o["source_chat_ids"])
                                        and not set(rule.forbidden_source_ids) & set(o["source_chat_ids"]) for o in candidates), "WRONG_MUTATION_SOURCE")
        for op in operations:
            check("forbidden_mutation", not any(matches(r, op) for r in exp.forbidden_operations), "FORBIDDEN_MUTATION")
        for index in range(len(operations)):
            check("extra_mutation", exp.allow_additional_operations or index in used, "EXTRA_MUTATION")
        check("noop", bool(operations) or exp.allow_noop, "UNEXPECTED_NOOP")
        if exp.exact_operation_count is not None:
            check("operation_count", len(operations) == exp.exact_operation_count, "WRONG_OPERATION_COUNT")
    elif stage == "extraction":
        entities = [e for group in output.values() for e in group]
        facts = [(e, f) for e in entities for f in e["facts"]]
        def entity_match(rule, entity):
            return matches(rule, entity)
        def fact_match(rule, pair, check_source=True):
            entity, fact = pair
            return ((rule.entity_name is None or entity["entity_name"] == rule.entity_name)
                    and (rule.entity_selector is None or entity_match(rule.entity_selector, entity))
                    and (rule.attribute_name is None or fact["attribute_name"] == rule.attribute_name)
                    and (rule.value is None or fact["value"] == rule.value)
                    and (not check_source or "required_source_chat_id" not in rule.model_fields_set or fact["source_chat_id"] == rule.required_source_chat_id))
        for rules, values, matcher, code in ((exp.required_entities, entities, entity_match, "MISSING_ENTITY"),
                                            (exp.required_facts, facts, fact_match, "MISSING_FACT")):
            matched, _ = assignment(rules, values, matcher)
            for index in range(len(rules)):
                check(code.lower(), index in matched, code)
                if code == "MISSING_FACT" and index not in matched and "required_source_chat_id" in rules[index].model_fields_set:
                    if any(fact_match(rules[index], pair, check_source=False) for pair in facts):
                        check("fact_source", False, "WRONG_FACT_SOURCE")
        for rules, values, matcher, code in ((exp.forbidden_entities, entities, entity_match, "FORBIDDEN_ENTITY"),
                                            (exp.forbidden_facts, facts, fact_match, "FORBIDDEN_FACT")):
            for value in values:
                check(code.lower(), not any(matcher(r, value) for r in rules), code)
        for expected, actual, code in ((exp.exact_entity_count, len(entities), "EXTRA_ENTITY"),
                                       (exp.exact_fact_count, len(facts), "EXTRA_FACT")):
            if expected is not None:
                check("exact_count", actual == expected, code if actual > expected else "WRONG_ENTITY_COUNT" if code == "EXTRA_ENTITY" else "WRONG_FACT_COUNT")
        if exp.require_unique_entity_ids:
            check("unique_ids", len({e["entity_id"] for e in entities}) == len(entities), "DUPLICATE_ENTITY_ID")
    elif stage == "contextual_action":
        actions = output["actions"]
        offered = {c["db_id"]: {f["fact_id"] for f in c["assertions"]} for c in case.input["CANDIDATES"]}
        for action in actions:
            if action["target"] is not None:
                key = action["target"]["db_id"]
                check("offered_target", key in offered, "WRONG_CONTEXTUAL_TARGET")
                check("offered_assertion", action["target_fact_id"] in offered.get(key, set()), "WRONG_CONTEXTUAL_ASSERTION")
        bounds = exp.expected_action_count
        minimum, maximum = (bounds, bounds) if isinstance(bounds, int) else (bounds.minimum, bounds.maximum)
        check("action_count", minimum <= len(actions) <= maximum, "FALSE_CONTEXTUAL_ACTION" if len(actions) > maximum else "MISSED_CONTEXTUAL_ACTION")
        if exp.require_no_action:
            check("no_action", not actions, "FALSE_CONTEXTUAL_ACTION")
        def action_match(rule, action):
            return ((rule.action_type is None or rule.action_type == action["action_type"])
                    and (not rule.require_unresolved or action["target"] is None and action["target_fact_id"] is None)
                    and (rule.required_target_db_id is None or action["target"] is not None and action["target"]["db_id"] == rule.required_target_db_id)
                    and (rule.required_target_fact_id is None or action["target_fact_id"] == rule.required_target_fact_id)
                    and set(rule.required_source_ids) <= set(action["source_chat_ids"])
                    and not set(rule.forbidden_source_ids) & set(action["source_chat_ids"]))
        matched, _ = assignment(exp.required_actions, actions, action_match)
        for index, rule in enumerate(exp.required_actions):
            check("required_action", index in matched, "MISSED_CONTEXTUAL_ACTION")
            if index not in matched:
                for passed, code in ((any(a["action_type"] == rule.action_type for a in actions), "WRONG_CONTEXTUAL_TYPE"),
                    (any(a["target"] is not None and a["target"]["db_id"] == rule.required_target_db_id for a in actions), "WRONG_CONTEXTUAL_TARGET"),
                    (any(a["target_fact_id"] == rule.required_target_fact_id for a in actions), "WRONG_CONTEXTUAL_ASSERTION"),
                    (any(set(rule.required_source_ids) <= set(a["source_chat_ids"]) and not set(rule.forbidden_source_ids) & set(a["source_chat_ids"]) for a in actions), "WRONG_CONTEXTUAL_SOURCE")):
                    # Only diagnose fields actually constrained by this case.
                    relevant = {"WRONG_CONTEXTUAL_TYPE": rule.action_type is not None,
                                "WRONG_CONTEXTUAL_TARGET": rule.required_target_db_id is not None or rule.require_unresolved,
                                "WRONG_CONTEXTUAL_ASSERTION": rule.required_target_fact_id is not None,
                                "WRONG_CONTEXTUAL_SOURCE": bool(rule.required_source_ids or rule.forbidden_source_ids)}
                    if relevant[code]:
                        check(code.lower(), passed, code)
    elif stage == "internal_keyword_merge":
        pairs = {tuple(sorted(pair)) for group in output["groups"] for pair in combinations(set(group["words"]), 2)}
        offered = offered_keywords(case.input)
        for word in {w for group in output["groups"] for w in group["words"]}:
            check("offered_keyword", word in offered, "INVENTED_KEYWORD")
        for pair in exp.required_pairs:
            check("required_pair", tuple(sorted(pair)) in pairs, "MISSING_SYNONYM_GROUP")
        for pair in exp.forbidden_pairs:
            check("forbidden_pair", tuple(sorted(pair)) not in pairs, "FALSE_SYNONYM_GROUP")
    return finish(checks)


def finish(checks):
    failures = [c["reason_code"] for c in checks if not c["passed"]]
    return {"passed": not failures, "checks": checks, "failures": failures}


def summarize(records):
    def counts(items):
        return {"total_cases": len(items), "passed_cases": sum(r["passed"] for r in items),
                "failed_cases": sum(not r["passed"] for r in items)}
    codes = Counter(code for r in records for code in r["failures"])
    summary = {**counts(records), "unique_cases": len({r["case_id"] for r in records}),
               "failure_reason_counts": dict(sorted(codes.items()))}
    for field in ("stage", "severity", "partition"):
        summary["by_" + field] = {value: counts([r for r in records if r[field] == value])
                                  for value in sorted({r[field] for r in records})}
    for severity in ("CRITICAL", "HIGH"):
        summary[severity.lower() + "_failures"] = sum(not r["passed"] and r["severity"] == severity for r in records)
    for name, reasons in {
        "forbidden_mutation_count": ["FORBIDDEN_MUTATION"], "missed_required_mutation_count": ["MISSING_REQUIRED_MUTATION"],
        "false_new_count": ["FALSE_NEW"],
        "false_merge_count": ["FALSE_ANCHOR_MERGE", "FALSE_SYNONYM_GROUP"],
        "forbidden_fact_count": ["FORBIDDEN_FACT"], "missed_required_fact_count": ["MISSING_FACT"],
    }.items():
        summary[name] = sum(codes[c] for c in reasons)
    summary["wrong_target_count"] = sum(r["stage"] == "identity_resolution" and bool(
        {"WRONG_RESOLUTION_TARGET", "MISSING_TARGET", "FORBIDDEN_TARGET"} & set(r["failures"])) for r in records)
    return summary


@contextmanager
def quiet_provider_logs():
    loggers = [logging.getLogger(n) for n in ("tomte.engine", "openai", "httpx", "httpcore", "pydantic_ai")]
    levels = [logger.level for logger in loggers]
    try:
        for logger in loggers:
            logger.setLevel(logging.CRITICAL)
        yield
    finally:
        for logger, level in zip(loggers, levels):
            logger.setLevel(level)


def execute(provider, case, prompts=None):
    """Call production routes; overrides affect evaluation agents only and restore them."""
    from pydantic_ai import Agent
    from tomte.engine.providers import openai_provider
    prompt, schema = STAGES[case.stage]
    prompt = (prompts or {}).get(case.stage, prompt)
    text = case.input if isinstance(case.input, str) else json.dumps(case.input, ensure_ascii=False)
    with ExitStack() as stack:
        if prompts and case.stage in prompts and case.stage in {"junk", "extraction"}:
            name = "junk_filter_agent" if case.stage == "junk" else "tomte_extractor_agent"
            stack.enter_context(patch.object(openai_provider, name, Agent(output_type=schema, system_prompt=prompt)))
        if case.stage == "junk":
            return provider.is_junk(text)
        return provider.parse_structured(system_prompt=None if case.stage == "extraction" else prompt,
                                         user_text=text, schema=schema)


def evaluate(provider, case, repetition=1, prompts=None):
    from pydantic_ai import capture_run_messages
    from pydantic_ai.messages import ModelRequest, ModelResponse, RetryPromptPart
    started = perf_counter()
    output = None
    with quiet_provider_logs(), capture_run_messages() as messages:
        start_index = len(messages)  # Native nested captures may reuse an enclosing history.
        try:
            output = execute(provider, case, prompts)
        except Exception:
            # Never serialize exception bodies, validation inputs or transport details.
            outcome = finish([{"name": "model_execution", "passed": False, "reason_code": "MODEL_CALL_FAILED"}])
        else:
            outcome = score(case, output)
        messages = messages[start_index:]
    responses = [m for m in messages if isinstance(m, ModelResponse)]
    usage = {"models": sorted({m.model_name for m in responses if m.model_name}),
             "request_count": sum(isinstance(m, ModelRequest) for m in messages),
             "input_tokens": sum(m.usage.input_tokens for m in responses) if responses else None,
             "output_tokens": sum(m.usage.output_tokens for m in responses) if responses else None}
    return {"case_id": case.case_id, "partition": case.partition, "stage": case.stage,
            "severity": case.severity, "tags": case.tags, "repetition": repetition, **outcome,
            "structured_output": output.model_dump(mode="json") if isinstance(output, BaseModel) else output,
            "latency_seconds": perf_counter() - started,
            "retry_count": sum(isinstance(p, RetryPromptPart) for m in messages for p in m.parts), "usage": usage}


def manifest(cases, hashes, repeat, partition, prompts=None):
    from tomte.engine.core.config import settings
    git = ["git", "-c", "safe.directory=" + ROOT.as_posix()]
    commit = subprocess.check_output(git + ["rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    stages = sorted({c.stage for c in cases})
    return {"git_commit": commit, "git_dirty": bool(subprocess.check_output(git + ["status", "--porcelain"], cwd=ROOT)),
            "timestamp": datetime.now(timezone.utc).isoformat(), "partition": partition,
            "stages": stages, "repeat": repeat, "selected_case_ids": [c.case_id for c in cases],
            "strong_model": settings.openai.strong_model, "cheap_model": settings.openai.cheap_model,
            "embedding_model": settings.openai.embed_model, "prompt_hashes": prompt_hashes(stages, prompts),
            "schema_hashes": schema_hashes(stages), "case_file_hashes": hashes,
            "evaluator_version": VERSION, "case_schema_version": 1,
            "evaluator_hashes": {p: sha256((ROOT / p).read_bytes()).hexdigest()
                                 for p in ("scripts/evaluate_prompts.py", "scripts/prompt_eval_support.py")}}
