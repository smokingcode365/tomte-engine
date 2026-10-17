"""Artificial harness mechanics only; no semantic benchmark or real provider calls."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts import prompt_eval_support as support
from scripts import evaluate_prompts as cli

FIXTURE = support.ROOT / "evals/prompts/fixtures/harness_smoke.json"
DOCUMENT = json.loads(FIXTURE.read_text(encoding="utf-8"))
SMOKE = {c["stage"]: c for c in DOCUMENT["cases"]}
UUID = "00000000-0000-0000-0000-000000000001"


def case(stage, expected=None, payload=None):
    data = deepcopy(SMOKE[stage])
    if expected is not None:
        data["expected"] = expected
    if payload is not None:
        data["input"] = payload
    return support.Case.model_validate(data)


@pytest.fixture
def files(tmp_path, monkeypatch):
    # Keep fixture writes isolated in pytest's temp directory; no repository changes.
    monkeypatch.setattr(support, "ROOT", tmp_path)
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    def write(cases, name="cases.json"):
        path = tmp_path / name
        path.write_text(json.dumps({"schema_version": 1, "cases": cases}), encoding="utf-8")
        return path
    return write


def test_duplicate_ids_rejected_across_files_before_filtering(files):
    first = files([SMOKE["junk"]], "a.json")
    second = files([SMOKE["junk"]], "b.json")
    with pytest.raises(ValueError, match="Duplicate"):
        support.load_cases([first, second])


@pytest.mark.parametrize("field,value", [("partition", "heldout"), ("severity", "URGENT"),
    ("stage", "invented"), ("schema_version", 2), ("schema_version", True), ("tags", "text")])
def test_bad_case_metadata_rejected(field, value):
    data = deepcopy(SMOKE["junk"])
    data[field] = value
    with pytest.raises(ValueError):
        support.Case.model_validate(data)


@pytest.mark.parametrize("field", list(SMOKE["junk"]))
def test_all_required_fields_are_required(field):
    data = deepcopy(SMOKE["junk"])
    del data[field]
    with pytest.raises(ValueError):
        support.Case.model_validate(data)


@pytest.mark.parametrize("stage", list(support.STAGES))
def test_malformed_stage_expectation_rejected(stage):
    with pytest.raises(ValueError):
        case(stage, {"unexpected": True})


@pytest.mark.parametrize("stage", list(support.STAGES))
def test_every_stage_pass_example(stage):
    c = case(stage)
    result = support.score(c, DOCUMENT["offline_outputs"][c.case_id])
    assert result["passed"] and not result["failures"]


@pytest.mark.parametrize("stage,output", [
    ("junk", False), ("cross_anchor_merge", True),
    ("extraction", {"assets_and_possessions": [{"entity_id": "tmp", "entity_name": "Token",
        "entity_domain": "assets_and_possessions", "entity_type": "token", "facts": []}]}),
    ("terminology", {"entity_type_decisions": [{"entity_handle": "unknown", "decision": "KEEP_RAW"}]}),
    ("identity_resolution", {"resolutions": [{"extracted_entity_id": "unknown", "kind": "NEW", "reasoning": "fixture"}]}),
    ("contextual_action", {"actions": [{"action_type": "CLARIFICATION", "reference_hint": "token",
        "source_chat_ids": ["s"], "interpretation_confidence": .1}]}),
    ("fact_mutation", {"plans": [{"target_key": "unknown", "operations": [{"operation": "ADD",
        "attribute_name": "token", "new_value": "a", "source_chat_ids": ["s"],
        "interpretation_confidence": .1, "evidence_type": "USER_EXPLICIT"}]}]}),
    ("internal_keyword_merge", {"groups": [{"words": ["invented"], "confidence_score": .1}]}),
])
def test_every_stage_fail_example(stage, output):
    result = support.score(case(stage), output)
    assert not result["passed"] and result["failures"]


def term(expected="KEEP_RAW"):
    return case("terminology", {"decisions": [{"entity_handle": "e", "expected_decision": expected,
        **({"expected_candidate_handle": "c"} if expected == "REUSE" else {})}]},
        {"entities": [{"entity_handle": "e", "candidates": [{"candidate_handle": "c", "keyword": "token"}], "facts": []}]})


def test_omitted_keep_raw_is_success_but_omitted_reuse_fails():
    assert support.score(term(), {})["passed"]
    assert "MISSED_REUSE" in support.score(term("REUSE"), {})["failures"]
    result = support.score(term(), {"entity_type_decisions": [{"entity_handle": "e", "decision": "REUSE", "candidate_handle": "c"}]})
    assert "FALSE_REUSE" in result["failures"]


def test_identity_target_contract_ignores_reasoning_and_confidence():
    c = case("identity_resolution", {"resolutions": [{"extracted_entity_id": "e", "expected_kind": "MATCH_ONE",
        "required_target_ids": [UUID], "allowed_evidence_labels": ["EXPLICIT"]}]},
        {**SMOKE["identity_resolution"]["input"], "CANDIDATES_BY_OBJECT": {"e": [{"id": UUID}]}, "EXTRACTED_SEMANTIC_INFORMATION": [{"id": "e"}]})
    for confidence in (.01, 1.):
        result = support.score(c, {"resolutions": [{"extracted_entity_id": "e", "kind": "MATCH_ONE",
            "reasoning": "Arbitrary wording", "targets": [{"db_id": UUID, "resolution_confidence": confidence, "evidence": "EXPLICIT"}]}]})
        assert result["passed"]


def mutation_case(required=None, forbidden=None, **extra):
    return case("fact_mutation", {"required_operations": required or [], "forbidden_operations": forbidden or [],
        "allow_noop": True, **extra}, {**SMOKE["fact_mutation"]["input"], "TARGETS": [{"target_key": "t", "assertions": []}],
        "CURRENT_BATCH": [{"item_id": "s", "role": "USER"}, {"item_id": "bad", "role": "ASSISTANT"}]})


def mutation_output(value="a", sources=None):
    return {"plans": [{"target_key": "t", "operations": [{"operation": "ADD", "attribute_name": "token",
        "new_value": value, "source_chat_ids": sources or ["s"], "interpretation_confidence": .01, "evidence_type": "USER_EXPLICIT"}]}]}


def test_mutation_forbidden_extra_and_missing_required_reasons():
    output = mutation_output()
    assert "FORBIDDEN_MUTATION" in support.score(mutation_case(forbidden=[{"operation": "ADD"}]), output)["failures"]
    assert "EXTRA_MUTATION" in support.score(mutation_case(), output)["failures"]
    assert "MISSING_REQUIRED_MUTATION" in support.score(mutation_case(required=[{"operation": "ADD"}]), {"plans": []})["failures"]
    assert support.score(mutation_case(allow_additional_operations=True), output)["passed"]


def test_mutation_value_source_constraints_and_one_to_one_requirements():
    required = [{"target_key": "t", "operation": "ADD", "new_value": "a", "required_source_ids": ["s"],
                 "forbidden_source_ids": ["bad"], "allowed_evidence_types": ["USER_EXPLICIT"]}]
    assert support.score(mutation_case(required=required), mutation_output())["passed"]
    assert "WRONG_MUTATION_VALUE" in support.score(mutation_case(required=required), mutation_output("b"))["failures"]
    assert "WRONG_MUTATION_SOURCE" in support.score(mutation_case(required=required), mutation_output(sources=["bad"]))["failures"]
    assert "MISSING_REQUIRED_MUTATION" in support.score(mutation_case(required=required * 2), mutation_output())["failures"]


def extracted_output(source="s", entity_id="tmp_arbitrary"):
    return {"assets_and_possessions": [{"entity_id": entity_id, "entity_name": "Token", "entity_domain": "assets_and_possessions",
        "entity_type": "token", "facts": [{"attribute_name": "token", "value": "a", "source_chat_id": source, "confidence": .01}]}]}


def test_extraction_required_forbidden_and_source_constraints():
    rule = {"entity_name": "Token", "attribute_name": "token", "value": "a", "required_source_chat_id": "s"}
    expected = {**SMOKE["extraction"]["expected"], "required_facts": [rule], "exact_entity_count": 1, "exact_fact_count": 1}
    c = case("extraction", expected, {**SMOKE["extraction"]["input"], "CURRENT_BATCH": [{"item_id": "s"}]})
    assert support.score(c, extracted_output())["passed"]
    assert support.score(c, extracted_output(entity_id="tmp_different"))["passed"]
    assert "MISSING_FACT" in support.score(c, {})["failures"]
    assert "WRONG_FACT_SOURCE" in support.score(c, extracted_output("other"))["failures"]
    expected["forbidden_facts"] = [rule]
    assert "FORBIDDEN_FACT" in support.score(case("extraction", expected, c.input), extracted_output())["failures"]


def test_internal_merge_pairs_order_text_and_invented_words():
    c = case("internal_keyword_merge", {"required_pairs": [["a", "b"]], "forbidden_pairs": [["b", "c"]]}, "Keywords: ['a', 'b', 'c']")
    assert support.score(c, {"groups": [{"words": ["b", "a"], "confidence_score": .01}]})["passed"]
    bad = support.score(c, {"groups": [{"words": ["a", "b", "c", "A"], "confidence_score": 1}]})
    assert {"FALSE_SYNONYM_GROUP", "INVENTED_KEYWORD"} <= set(bad["failures"])


def test_hashing_and_case_file_changes_are_deterministic(files):
    assert support.prompt_hashes(support.STAGES) == support.prompt_hashes(reversed(support.STAGES))
    assert support.schema_hashes(support.STAGES) == support.schema_hashes(reversed(support.STAGES))
    assert support.schema_hashes(["junk"])["junk"] == support.digest('{"type":"boolean"}')
    path = files([SMOKE["junk"]])
    _, first = support.load_cases([path])
    assert support.load_cases([path])[1] == first
    path.write_text(path.read_text() + "\n")
    assert support.load_cases([path])[1] != first


def test_stage_dispatch_uses_production_prompts_schemas_and_no_vocabulary():
    provider = Mock()
    for stage, (prompt, schema) in support.STAGES.items():
        c = case(stage)
        support.execute(provider, c)
        if stage == "junk":
            provider.is_junk.assert_called_once()
        else:
            kwargs = provider.parse_structured.call_args.kwargs
            assert kwargs["system_prompt"] == (None if stage == "extraction" else prompt)
            assert kwargs["schema"] is schema
            assert "taxonomy" not in kwargs and "lexicon" not in kwargs


def test_default_cli_makes_no_provider_or_network_calls(files, monkeypatch, capsys):
    import socket
    forbidden = Mock(side_effect=AssertionError("No network/provider in default mode"))
    monkeypatch.setattr(cli, "create_provider", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)
    path = files(DOCUMENT["cases"])
    assert cli.main(["--cases", str(path)]) == 0
    forbidden.assert_not_called()
    assert json.loads(capsys.readouterr().out)["selected_cases"] == 8
    assert not (path.parent / "log").exists()


def test_cli_all_filters_and_invalid_slice(files, capsys):
    path = files(DOCUMENT["cases"])
    assert cli.main(["--cases", str(path), "--stage", "junk", "--case-id", SMOKE["junk"]["case_id"],
                     "--partition", "dev", "--limit", "1", "--repeat", "2"]) == 0
    assert json.loads(capsys.readouterr().out)["selected_cases"] == 1
    assert cli.main(["--cases", str(path), "--partition", "validation"]) == 0
    assert json.loads(capsys.readouterr().out)["selected_cases"] == 0
    with pytest.raises(SystemExit):
        cli.main(["--cases", str(path), "--case-id", "unknown"])


def test_summary_severity_partition_and_failure_metrics():
    records = [dict(case_id="a", stage="fact_mutation", partition="dev", severity="CRITICAL", passed=False,
        failures=["FORBIDDEN_MUTATION", "MISSING_REQUIRED_MUTATION"]),
        dict(case_id="b", stage="extraction", partition="validation", severity="HIGH", passed=False,
        failures=["FORBIDDEN_FACT"]), dict(case_id="c", stage="junk", partition="dev", severity="LOW", passed=True, failures=[])]
    summary = support.summarize(records)
    assert summary["critical_failures"] == summary["high_failures"] == 1
    assert summary["passed_cases"] == 1 and summary["failed_cases"] == 2
    assert summary["forbidden_mutation_count"] == summary["missed_required_mutation_count"] == 1
    assert summary["forbidden_fact_count"] == 1
    assert summary["by_partition"]["validation"]["failed_cases"] == 1


def test_live_reporting_repeat_and_sanitized_failure_with_fake_provider(files, monkeypatch, capsys):
    path = files([SMOKE["junk"]])
    provider = Mock()
    provider.is_junk.return_value = True
    monkeypatch.setattr(cli, "create_provider", lambda: provider)
    monkeypatch.setattr(cli, "manifest", lambda *a: {"timestamp": "synthetic", "git_commit": "fake"})
    assert cli.main(["--cases", str(path), "--live", "--repeat", "2"]) == 0
    output = next((path.parent / "log/prompt_evals").iterdir())
    assert {p.name for p in output.iterdir()} == {"manifest.json", "results.jsonl", "summary.json", "usage.json"}
    records = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()]
    assert [r["repetition"] for r in records] == [1, 2]
    provider.is_junk.side_effect = RuntimeError("synthetic-secret-NEVER-serialize")
    record = support.evaluate(provider, case("junk"))
    assert record["failures"] == ["MODEL_CALL_FAILED"]
    assert "synthetic-secret" not in json.dumps(record) + capsys.readouterr().out


def test_override_agent_and_logging_restored_even_on_exception():
    import logging
    from tomte.engine.providers import openai_provider
    original = openai_provider.junk_filter_agent
    logger = logging.getLogger("httpx")
    original_level = logger.level
    provider = Mock()
    provider.is_junk.side_effect = RuntimeError("synthetic failure")
    with pytest.raises(RuntimeError):
        support.execute(provider, case("junk"), {"junk": "Synthetic alternate instructions"})
    assert openai_provider.junk_filter_agent is original
    support.evaluate(provider, case("junk"))
    assert logger.level == original_level


def test_contextual_count_target_source_and_unresolved_matching():
    payload = {**SMOKE["contextual_action"]["input"], "CANDIDATES": [{"db_id": UUID, "assertions": [{"fact_id": "f"}]}],
        "CURRENT_BATCH": [{"item_id": "s", "role": "USER"}, {"item_id": "bad", "role": "ASSISTANT"}]}
    output = {"actions": [{"action_type": "CONFIRMATION", "reference_hint": "Arbitrary wording",
        "source_chat_ids": ["s"], "interpretation_confidence": .01,
        "target": {"db_id": UUID, "resolution_confidence": .01, "evidence": "EXPLICIT"}, "target_fact_id": "f"}]}
    expected = {"expected_action_count": {"minimum": 1, "maximum": 2}, "required_actions": [{
        "action_type": "CONFIRMATION", "required_target_db_id": UUID, "required_target_fact_id": "f",
        "required_source_ids": ["s"], "forbidden_source_ids": ["bad"]}]}
    assert support.score(case("contextual_action", expected, payload), output)["passed"]
    wrong = deepcopy(output)
    wrong["actions"][0]["source_chat_ids"] = ["bad"]
    assert "WRONG_CONTEXTUAL_SOURCE" in support.score(case("contextual_action", expected, payload), wrong)["failures"]
    unresolved = {"expected_action_count": 1, "required_actions": [{"require_unresolved": True}]}
    assert not support.score(case("contextual_action", unresolved, payload), output)["passed"]
    output["actions"][0].update(target=None, target_fact_id=None)
    assert support.score(case("contextual_action", unresolved, payload), output)["passed"]


def test_native_usage_and_retry_capture_and_restoration(monkeypatch):
    from pydantic_ai import Agent, capture_run_messages
    from pydantic_ai.messages import ModelResponse, ToolCallPart, TextPart
    from pydantic_ai.models.function import FunctionModel
    from pydantic_ai.usage import RequestUsage
    calls = []
    def model(messages, info):
        calls.append(1)
        value = {"groups": [{"words": ["token_a"], "confidence_score": "invalid"}]} if len(calls) == 1 else {"groups": []}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, value)], model_name="synthetic-model",
                             usage=RequestUsage(input_tokens=7, output_tokens=3))
    fake_model = FunctionModel(model)
    agent = Agent(fake_model, output_type=support.STAGES["internal_keyword_merge"][1])
    provider = SimpleNamespace(parse_structured=lambda **kwargs: agent.run_sync(kwargs["user_text"]).output)
    with capture_run_messages() as history:
        history.append(ModelResponse(parts=[TextPart("synthetic prior result")], model_name="prior",
                                     usage=RequestUsage(input_tokens=100, output_tokens=100)))
        record = support.evaluate(provider, case("internal_keyword_merge"))
    assert record["passed"] and record["retry_count"] == 1
    assert record["usage"] == {"models": [fake_model.model_name], "request_count": 2, "input_tokens": 14, "output_tokens": 6}


def test_strict_boolean_and_nonfinite_output_rejected():
    assert support.score(case("junk"), 1)["failures"] == ["INVALID_STRUCTURED_OUTPUT"]
    output = {"groups": [{"words": ["token_a"], "confidence_score": float("nan")}]}
    assert support.score(case("internal_keyword_merge"), output)["failures"] == ["INVALID_STRUCTURED_OUTPUT"]


def test_maximum_matching_does_not_greedily_steal_specific_result():
    rules = [support.MutationRule(operation="ADD"), support.MutationRule(operation="ADD", new_value="a")]
    outputs = [{"operation": "ADD", "new_value": "a"}, {"operation": "ADD", "new_value": "b"}]
    assert support.assignment(rules, outputs, support.matches) == ({0, 1}, {0, 1})


def identity_data():
    data = deepcopy(SMOKE["identity_resolution"])
    data["input"]["EXTRACTED_SEMANTIC_INFORMATION"] = [{"id": "e1"}, {"id": "e2"}]
    data["input"]["CANDIDATES_BY_OBJECT"] = {"e1": [{"id": UUID}], "e2": []}
    data["expected"]["resolutions"] = [
        {"extracted_entity_id": "e1", "expected_kind": "MATCH_ONE", "required_target_ids": [UUID]},
        {"extracted_entity_id": "e2", "expected_kind": "UNRESOLVED"}]
    return data


@pytest.mark.parametrize("fault", ["missing_expectation", "missing_candidate_object", "extra_candidate_object",
                                  "required_target", "forbidden_target", "missing_id", "duplicate_id"])
def test_identity_incomplete_or_unoffered_contract_rejected(fault):
    data = identity_data()
    if fault == "missing_expectation":
        data["expected"]["resolutions"].pop()
    elif fault == "missing_candidate_object":
        del data["input"]["CANDIDATES_BY_OBJECT"]["e2"]
    elif fault == "extra_candidate_object":
        data["input"]["CANDIDATES_BY_OBJECT"]["extra"] = []
    elif fault in {"required_target", "forbidden_target"}:
        data["expected"]["resolutions"][0][fault + "_ids"] = ["not-offered"]
    elif fault == "missing_id":
        data["input"]["EXTRACTED_SEMANTIC_INFORMATION"][1] = {"entity_id": "e2"}
    else:
        data["input"]["EXTRACTED_SEMANTIC_INFORMATION"][1]["id"] = "e1"
    with pytest.raises(ValueError):
        support.Case.model_validate(data)


def test_identity_full_multi_object_contract_scores_every_object():
    c = support.Case.model_validate(identity_data())
    output = {"resolutions": [{"extracted_entity_id": "e1", "kind": "MATCH_ONE", "reasoning": "fixture",
        "targets": [{"db_id": UUID, "resolution_confidence": .5, "evidence": "EXPLICIT"}]},
        {"extracted_entity_id": "e2", "kind": "UNRESOLVED", "reasoning": "fixture"}]}
    assert support.score(c, output)["passed"]
    output["resolutions"].pop()
    assert "MISSING_RESOLUTION" in support.score(c, output)["failures"]
    data = identity_data()
    data["expected"]["resolutions"][1]["forbidden_target_ids"] = [UUID]
    with pytest.raises(ValueError):  # Offered for e1 is not offered for e2.
        support.Case.model_validate(data)


def terminology_data():
    data = deepcopy(SMOKE["terminology"])
    data["input"]["entities"] = [{"entity_handle": "e", "candidates": [{"candidate_handle": "ec"}],
                                  "facts": [{"fact_handle": "f", "candidates": [{"candidate_handle": "fc"}]}]}]
    data["expected"]["decisions"] = [{"entity_handle": "e", "expected_decision": "KEEP_RAW"},
        {"entity_handle": "e", "fact_handle": "f", "expected_decision": "KEEP_RAW"}]
    return data


@pytest.mark.parametrize("fault", ["missing", "extra", "wrong_candidate", "duplicate"])
def test_terminology_exact_expectation_coverage_rejected(fault):
    data = terminology_data()
    rules = data["expected"]["decisions"]
    if fault == "missing":
        rules.pop()
    elif fault == "extra":
        rules.append({"entity_handle": "extra", "expected_decision": "KEEP_RAW"})
    elif fault == "duplicate":
        rules.append(deepcopy(rules[0]))
    else:
        rules[1].update(expected_decision="REUSE", expected_candidate_handle="ec")
    with pytest.raises(ValueError):
        support.Case.model_validate(data)


def test_terminology_full_coverage_scores_unexpected_fact_reuse():
    c = support.Case.model_validate(terminology_data())
    assert support.score(c, {})["passed"]  # Model omission still means KEEP_RAW.
    result = support.score(c, {"attribute_decisions": [{"entity_handle": "e", "fact_handle": "f",
        "decision": "REUSE", "candidate_handle": "fc"}]})
    assert "FALSE_REUSE" in result["failures"]


def reference_data(stage):
    data = deepcopy(SMOKE[stage])
    data["input"]["CURRENT_BATCH"] = [{"item_id": "s", "role": "ASSISTANT"}, {"item_id": "other-source", "role": "USER"}]
    if stage == "fact_mutation":
        data["input"]["TARGETS"] = [{"target_key": "t", "assertions": [{"fact_id": "f"}]},
                                   {"target_key": "other", "assertions": [{"fact_id": "other-f"}]}]
    elif stage == "contextual_action":
        data["input"]["CANDIDATES"] = [{"db_id": UUID, "assertions": [{"fact_id": "f"}]},
                                       {"db_id": "other", "assertions": [{"fact_id": "other-f"}]}]
        data["expected"] = {"expected_action_count": 1, "required_actions": []}
    return data


@pytest.mark.parametrize("collection", ["required_operations", "forbidden_operations"])
@pytest.mark.parametrize("rule", [{"target_key": "unknown"}, {"target_fact_id": "unknown"},
    {"target_key": "t", "target_fact_id": "other-f"}, {"required_source_ids": ["unknown"]},
    {"forbidden_source_ids": ["unknown"]}])
def test_mutation_expectation_unoffered_reference_rejected(collection, rule):
    data = reference_data("fact_mutation")
    data["expected"][collection] = [rule]
    with pytest.raises(ValueError):
        support.Case.model_validate(data)


@pytest.mark.parametrize("collection", ["required_operations", "forbidden_operations"])
@pytest.mark.parametrize("target", [None, "t"])
def test_mutation_valid_references_accept_assistant_sources(collection, target):
    data = reference_data("fact_mutation")
    data["expected"][collection] = [{"target_key": target, "target_fact_id": "f",
                                     "required_source_ids": ["s"], "forbidden_source_ids": ["other-source"]}]
    support.Case.model_validate(data)  # Structural validation does not apply persistence policy.


@pytest.mark.parametrize("rule", [{"required_target_db_id": "unknown"}, {"required_target_fact_id": "unknown"},
    {"required_target_db_id": UUID, "required_target_fact_id": "other-f"},
    {"required_source_ids": ["unknown"]}, {"forbidden_source_ids": ["unknown"]}])
def test_contextual_expectation_unoffered_reference_rejected(rule):
    data = reference_data("contextual_action")
    data["expected"]["required_actions"] = [rule]
    with pytest.raises(ValueError):
        support.Case.model_validate(data)


@pytest.mark.parametrize("target", [None, UUID])
def test_contextual_valid_references_accept_assistant_sources(target):
    data = reference_data("contextual_action")
    data["expected"]["required_actions"] = [{"required_target_db_id": target, "required_target_fact_id": "f",
        "required_source_ids": ["s"], "forbidden_source_ids": ["other-source"]}]
    support.Case.model_validate(data)


@pytest.mark.parametrize("collection", ["required_facts", "forbidden_facts"])
@pytest.mark.parametrize("source", ["unknown", None])
def test_extraction_expectation_unoffered_source_rejected(collection, source):
    data = reference_data("extraction")
    data["expected"][collection] = [{"required_source_chat_id": source}]
    with pytest.raises(ValueError):
        support.Case.model_validate(data)


@pytest.mark.parametrize("collection", ["required_facts", "forbidden_facts"])
def test_extraction_valid_assistant_source_reference_accepted(collection):
    data = reference_data("extraction")
    data["expected"][collection] = [{"required_source_chat_id": "s"}]
    support.Case.model_validate(data)


def test_internal_extra_pairs_fail_closed_with_explicit_opt_out():
    expected = {"required_pairs": [["a", "b"]], "forbidden_pairs": []}
    c = case("internal_keyword_merge", expected, "Keywords: ['a', 'b', 'c']")
    required_only = {"groups": [{"words": ["a", "b"], "confidence_score": .5}]}
    assert support.score(c, required_only)["passed"]
    extra = {"groups": [{"words": ["a", "b", "c"], "confidence_score": .5}]}
    result = support.score(c, extra)
    assert result["failures"] == ["EXTRA_SYNONYM_GROUP", "EXTRA_SYNONYM_GROUP"]
    record = dict(case_id=c.case_id, stage=c.stage, partition=c.partition, severity=c.severity, **result)
    assert support.summarize([record])["false_merge_count"] == 2
    expected["allow_additional_pairs"] = True
    assert support.score(case(c.stage, expected, c.input), extra)["passed"]
    expected["forbidden_pairs"] = [["b", "c"]]
    assert "FALSE_SYNONYM_GROUP" in support.score(case(c.stage, expected, c.input), extra)["failures"]
    extra["groups"][0]["words"].append("invented")
    assert "INVENTED_KEYWORD" in support.score(case(c.stage, expected, c.input), extra)["failures"]


def test_generic_override_is_passed_to_execution():
    provider = Mock()
    support.execute(provider, case("fact_mutation"), {"fact_mutation": "ALT_PROMPT"})
    assert provider.parse_structured.call_args.kwargs["system_prompt"] == "ALT_PROMPT"


@pytest.mark.parametrize("stage,name", [("junk", "junk_filter_agent"), ("extraction", "tomte_extractor_agent")])
@pytest.mark.parametrize("fail", [False, True])
def test_special_override_is_bound_during_execution_and_restored(monkeypatch, stage, name, fail):
    import pydantic_ai
    from tomte.engine.providers import openai_provider
    original = getattr(openai_provider, name)
    replacement = object()
    constructor = Mock(return_value=replacement)
    monkeypatch.setattr(pydantic_ai, "Agent", constructor)
    def observe(*args, **kwargs):
        assert getattr(openai_provider, name) is replacement
        if fail:
            raise RuntimeError("synthetic failure")
        return "synthetic result"
    provider = SimpleNamespace(is_junk=observe, parse_structured=observe)
    if fail:
        with pytest.raises(RuntimeError, match="synthetic failure"):
            support.execute(provider, case(stage), {stage: "ALT_PROMPT"})
    else:
        assert support.execute(provider, case(stage), {stage: "ALT_PROMPT"}) == "synthetic result"
    constructor.assert_called_once_with(output_type=support.STAGES[stage][1], system_prompt="ALT_PROMPT")
    assert getattr(openai_provider, name) is original


def test_checked_in_case_schema_matches_runtime_export():
    schema = json.loads((support.ROOT / "evals/prompts/case_schema.json").read_text(encoding="utf-8"))
    assert schema == support.case_json_schema()
