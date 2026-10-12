"""Target persistence policy, with strict XFAILs for current implementation gaps.

Uses the existing local-only PostgreSQL/Redis mechanics fixture. Enable
TOMTE_RUN_POSTGRES_TESTS=1 and TOMTE_RUN_REDIS_TESTS=1. All model decisions and
embeddings are local deterministic stubs; the fixture forbids worker startup.
"""
from copy import deepcopy
import json
import pytest

from resolution_helpers import mutation_response
from test_consolidation_mechanics import mechanics, processed_count
from tomte.engine.core.assertions import adapt_assertions, project_facts
from tomte.engine.core.extraction import ExtractionMemorySchema
from tomte.engine.core.schema import Entity, Fact
from tomte.engine.workflows.consolidation import ConsolidationWorkflow


known_gap = pytest.mark.xfail(
    strict=True,
    reason="known persistence-contract gap",
    raises=(AssertionError, ValueError),
)


@pytest.fixture
def contract(mechanics, monkeypatch):
    """Prescribe model outputs, retaining real workflow, SQL, and queue behavior."""
    case = mechanics
    case.extracted = []
    case.matches = {}
    case.accepted_aliases = {}
    case.new_evidence = None
    case.contextual = None
    case.plan = lambda target, data: []
    case.junk = False
    case.calls = []

    def junk(text):
        case.calls.append("junk")
        return case.junk

    def embeddings(texts):
        case.calls.append("embedding")
        return [case.llm.vector() for _ in texts]

    def model(system_prompt, user_text, schema, **kwargs):
        case.calls.append(schema.__name__)
        data = json.loads(user_text)
        if schema is ExtractionMemorySchema:
            return schema.model_validate({"assets_and_possessions": case.extracted})
        if schema.__name__ == "TerminologyAlignmentResponse":
            return schema()  # KEEP_RAW still exercises real dictionary preparation.
        if schema.__name__ == "ContextualActionResponse":
            return schema.model_validate({"actions": case.contextual(data) if case.contextual else []})
        if schema.__name__ == "ResolutionResponse":
            resolutions = []
            for entity in data["EXTRACTED_SEMANTIC_INFORMATION"]:
                name = case.matches.get(entity["id"])
                candidate = next((c for c in data["CANDIDATES_BY_OBJECT"][entity["id"]]
                                  if c["name"] == name), None)
                if name is not None:
                    assert candidate is not None, "Fixture target must be offered by scoped retrieval"
                resolutions.append({
                    "extracted_entity_id": entity["id"],
                    "kind": "MATCH_ONE" if candidate else "NEW",
                    "targets": [{"db_id": candidate["id"], "resolution_confidence": 1,
                                 "evidence": "EXPLICIT"}] if candidate else [],
                    "explicitly_new": candidate is None,
                    "new_entity_evidence": (case.new_evidence.get(entity["id"])
                        if isinstance(case.new_evidence, dict) else
                        case.new_evidence or data["CURRENT_BATCH"][0]["content"]),
                    "reasoning": "Deterministic contract fixture",
                    "accepted_aliases": case.accepted_aliases.get(entity["id"], []),
                })
            return schema.model_validate({"resolutions": resolutions})
        assert schema.__name__ == "MutationPlanResponse", "Unexpected model stage"
        if case.plan is None:
            return mutation_response(schema, user_text)
        return schema.model_validate({"plans": [
            {"target_key": target["target_key"], "operations": case.plan(target, data)}
            for target in data["TARGETS"]
        ]})

    monkeypatch.setattr(case.llm, "is_junk", junk)
    monkeypatch.setattr(case.llm, "embed_batch", embeddings)
    monkeypatch.setattr(case.llm, "parse_structured", model)
    return case


def proposal(name="Tesla Model Y", entity_id="temporary", aliases=(),
             attribute="purchase_year", value="2024", source=None):
    return {"entity_id": entity_id, "entity_name": name,
            "entity_type": "vehicle", "entity_domain": "assets_and_possessions",
            "aliases": list(aliases), "facts": [{"attribute_name": attribute,
            "value": value, "confidence": 1, "source_chat_id": source}]}


def seed(case, name="Tesla Model Y", entity_id="tesla", year="2024", aliases=("Tesla",)):
    payload = Entity(entity_id=entity_id, entity_name=name, entity_type="vehicle",
                     entity_domain="assets_and_possessions", aliases=list(aliases),
                     facts=[Fact(attribute_name="purchase_year", value=year, confidence=1,
                                 source_chat_id="synthetic-seed",
                                 last_mentioned_at="2026-10-09T00:00:00Z")]).model_dump(mode="json")
    payload["facts"] = adapt_assertions(payload["facts"], case.tenant + case.user + entity_id)
    assert case.db.commit_graph_updates([{
        "mode": "INSERT", "tenant_id": case.tenant, "user_id": case.user,
        **{k: payload[k] for k in ("entity_id", "entity_name", "entity_type", "entity_domain")},
        "payload": payload, "embedding": case.llm.vector(),
    }])


def freeze(case, *messages):
    """Messages are (role, content); assistant queue entries have no embedding."""
    for role, content in messages:
        case.cache.enqueue_for_consolidation(
            case.tenant, case.user, case.session, content, role,
            vector=case.llm.vector() if role == "user" else None,
        )
    return case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session)


def rows(case):
    """Separate read-only connection observes every persisted memory column."""
    return case.observer.execute(
        "SELECT * FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s ORDER BY id",
        (case.tenant, case.user),
    ).fetchall()


def dictionary(case):
    """Check keywords, hits, vectors, timestamps and audit state, not call mocks."""
    return tuple(case.observer.execute(
        f"SELECT to_jsonb(t)::text FROM {table} AS t ORDER BY 1"
    ).fetchall() for table in (
        "tomte_semantic_anchors", "tomte_keyword_registry", "tomte_keyword_audit_log",
    ))


def payloads(case):
    return dict(case.observer.execute(
        "SELECT entity_name,payload FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s",
        (case.tenant, case.user),
    ).fetchall())


def complete_noop(case, batch):
    before, vocabulary = rows(case), dictionary(case)
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch) is True
    assert rows(case) == before
    assert dictionary(case) == vocabulary
    assert processed_count(case) == 1
    calls = list(case.calls)
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch) is True
    assert case.calls == calls
    assert rows(case) == before and dictionary(case) == vocabulary
    assert processed_count(case) == 1


def contextual_action(case, unresolved=False, action_type="CORRECTION"):
    def action(data):
        candidate = next(c for c in data["CANDIDATES"] if c["entity_name"] == "Honda Civic")
        source = next(m["item_id"] for m in data["CURRENT_BATCH"] if m["role"] == "user")
        return [{"action_type": action_type, "reference_hint": "Honda Civic",
                 "source_chat_ids": [source], "interpretation_confidence": 1,
                 "target": None if unresolved else {"db_id": candidate["db_id"],
                     "resolution_confidence": 1, "evidence": "RESOLVED_REFERENCE"},
                 "target_fact_id": None if unresolved else candidate["assertions"][0]["fact_id"]}]
    case.contextual = action


def test_junk_batch_is_successful_processed_noop_contract(contract):
    """Junk completes once without graph/dictionary writes or replay model work."""
    case = contract
    case.junk = True
    complete_noop(case, freeze(case, ("user", "Thanks!")))
    assert case.calls == ["junk"]


def test_assistant_only_batch_is_successful_processed_noop_contract(contract):
    """Assistant-only batches need no user embedding and must not retry forever."""
    complete_noop(contract, freeze(contract, ("assistant", "Your Honda is blue.")))
    assert contract.calls == []


@pytest.mark.parametrize("has_other_user_vector", [False, True])
def test_user_batch_missing_required_embedding_remains_retryable_control(contract, has_other_user_vector):
    """Missing user vectors are not silently processed as assistant-only no-ops."""
    case = contract
    if has_other_user_vector:
        case.cache.enqueue_for_consolidation(case.tenant, case.user, case.session,
                                            "My Tesla is blue.", "user", vector=case.llm.vector())
    case.cache.enqueue_for_consolidation(case.tenant, case.user, case.session,
                                        "I bought a bicycle.", "user", vector=None)
    batch = case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session)
    before, vocabulary = rows(case), dictionary(case)
    try:
        result = ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    except ValueError:
        result = False  # An explicit data-integrity error is also retryable.
    assert result is False
    assert rows(case) == before and dictionary(case) == vocabulary
    assert processed_count(case) == 0
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == batch


def test_unresolved_contextual_action_is_successful_noop_contract(contract):
    """An ambiguous referent is a completed no-op, not an infrastructure failure."""
    seed(contract, "Honda Civic", "honda", "2021", ("Honda",))
    contextual_action(contract, unresolved=True)
    complete_noop(contract, freeze(contract, ("user", "I meant the other car.")))


def test_contextual_resolution_followed_by_mutation_abstention_is_valid_contract(contract):
    """Selecting an assertion does not oblige the planner to change it."""
    seed(contract, "Honda Civic", "honda", "2021", ("Honda",))
    contextual_action(contract)
    complete_noop(contract, freeze(contract, ("user", "The Honda paperwork seems wrong.")))


def test_empty_mutation_plan_does_not_write_existing_entity_contract(contract):
    """Abstention preserves every row column, aliases and dictionary state."""
    seed(contract)
    contract.extracted = [proposal(aliases=("an unapproved reference",))]
    contract.matches = {"temporary": "Tesla Model Y"}
    complete_noop(contract, freeze(contract, ("user", "Can we discuss my Tesla?")))


def test_contextual_reference_is_not_automatically_persisted_as_alias_contract(contract):
    """A guessed referent must not create an alias; legitimate aliases survive."""
    case = contract
    seed(case, "Honda Civic", "honda", "2021", ("Honda",))
    seed(case)
    before = deepcopy(payloads(case))
    phrase = "the car I bought in 2024"
    case.extracted = [proposal("Honda Civic", aliases=(phrase,), value="2021")]
    case.matches = {"temporary": "Honda Civic"}
    case.plan = None  # Valid existing fact reinforcement must still persist.
    batch = freeze(case, ("user", "My Honda Civic was bought in 2021. Can we talk about " + phrase + "?"))
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    after = payloads(case)
    reinforced = project_facts(after["Honda Civic"]["facts"])[0]
    assert reinforced["value"] == "2021" and reinforced["mention_count"] == 2
    assert phrase not in after["Honda Civic"]["aliases"]
    assert after["Honda Civic"]["aliases"] == before["Honda Civic"]["aliases"]
    assert after["Tesla Model Y"] == before["Tesla Model Y"]


def test_different_extracted_name_is_not_automatic_alias_contract(contract):
    """Even an accepted fact reinforcement cannot authorize a different name."""
    case = contract
    seed(case)
    case.extracted = [proposal("The electric car")]
    case.matches = {"temporary": "Tesla Model Y"}
    case.plan = None  # Valid reinforcement provides an independent write control.
    batch = freeze(case, ("user", "My Tesla Model Y was bought in 2024."))
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    stored = payloads(case)["Tesla Model Y"]
    assert stored["aliases"] == ["Tesla"]
    assert stored["entity_name"] == "Tesla Model Y"
    assert project_facts(stored["facts"])[0]["mention_count"] == 2


def test_assistant_message_cannot_authorize_fact_persistence_contract(contract):
    """A real assistant source is omitted, permitting a completed empty batch."""
    case = contract
    batch = freeze(case, ("assistant", "Your bicycle is blue."), ("user", "I bought a new bicycle."))
    case.extracted = [proposal("Bicycle", attribute="paint_color", value="blue", source=batch[0]["item_id"])]
    case.new_evidence = batch[1]["content"]
    case.plan = None
    complete_noop(case, batch)
    assert case.cache.clear_session_4_consolidation(case.tenant, case.user, case.session, batch)


def test_assistant_text_cannot_authorize_new_entity_contract(contract):
    """Only user evidence may satisfy an explicitly NEW entity introduction."""
    case = contract
    batch = freeze(case, ("assistant", "You bought a new bicycle."), ("user", "That sounds interesting."))
    case.extracted = [proposal("Bicycle", attribute="ownership_status", value="owned", source=batch[1]["item_id"])]
    case.new_evidence = batch[0]["content"]
    case.plan = None
    complete_noop(case, batch)


def test_user_text_can_authorize_explicit_new_entity_control(contract):
    """User introductions must remain eligible for actual durable creation."""
    case = contract
    batch = freeze(case, ("user", "I bought a new bicycle."))
    case.extracted = [proposal("Bicycle", attribute="ownership_status", value="owned")]
    case.plan = None
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    stored = payloads(case)["Bicycle"]
    assert project_facts(stored["facts"])[0]["value"] == "owned"
    assert stored["facts"][0]["evidence"][0]["source_chat_id"] == batch[0]["item_id"]
    assert processed_count(case) == 1


def test_explicitly_accepted_nickname_persists_without_identity_split_and_replays(contract):
    case = contract
    seed(case)
    seed(case, "Honda Civic", "honda", "2021", ("Honda",))
    before_ids = case.observer.execute("SELECT id FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s ORDER BY id", (case.tenant, case.user)).fetchall()
    other = payloads(case)["Honda Civic"]
    case.extracted = [proposal(aliases=("Little Dog",), attribute="nickname", value="Little Dog")]
    case.matches = {"temporary": "Tesla Model Y"}
    case.accepted_aliases = {"temporary": ["Little Dog"]}
    case.plan = None
    batch = freeze(case, ("user", "I named my Tesla Little Dog."))
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    stored = payloads(case)["Tesla Model Y"]
    assert stored["aliases"] == ["Tesla", "Little Dog"]
    assert stored["entity_name"] == "Tesla Model Y"
    assert any(f["attribute_name"] == "nickname" and f["value"] == "Little Dog" for f in project_facts(stored["facts"]))
    assert payloads(case)["Honda Civic"] == other
    after = rows(case)
    assert case.observer.execute("SELECT id FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s ORDER BY id", (case.tenant, case.user)).fetchall() == before_ids
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    assert rows(case) == after


@pytest.mark.parametrize("source", ["unoffered", "assistant", "reference", "unapproved"])
def test_bad_alias_does_not_cancel_supported_fact(contract, source):
    case = contract
    seed(case)
    alias = "Little Dog"
    case.extracted = [proposal(aliases=() if source == "unoffered" else (alias,))]
    case.matches = {"temporary": "Tesla Model Y"}
    case.accepted_aliases = {"temporary": [] if source == "unapproved" else [alias]}
    case.plan = None
    messages = [("user", "My Tesla Model Y was bought in 2024.")]
    if source == "assistant":
        messages.insert(0, ("assistant", "Little Dog"))
    if source == "unoffered":
        messages[0] = ("user", "Little Dog was bought in 2024.")
    if source == "unapproved":
        messages[0] = ("user", "Little Dog was bought in 2024.")
    if source == "reference":
        case.cache.append_chat_history(case.tenant, case.session, "Little Dog", "user", case.user)
    batch = freeze(case, *messages)
    case.extracted[0]["facts"][0]["source_chat_id"] = next(m["item_id"] for m in batch if m["role"] == "user")
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    stored = payloads(case)["Tesla Model Y"]
    assert stored["aliases"] == ["Tesla"]
    assert project_facts(stored["facts"])[0]["mention_count"] == 2


def test_partial_alias_word_does_not_persist_or_cancel_reinforcement(contract):
    case = contract
    seed(case)
    seed(case, "Honda Civic", "honda", "2021", ("Honda",))
    before = payloads(case)
    before_ids = case.observer.execute(
        "SELECT id FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s ORDER BY id",
        (case.tenant, case.user)).fetchall()
    case.extracted = [proposal(aliases=("Ann",))]
    case.matches = {"temporary": "Tesla Model Y"}
    case.accepted_aliases = {"temporary": ["Ann"]}
    case.plan = None
    batch = freeze(case, ("user", "Annette asked about my Tesla Model Y. It was bought in 2024."))
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    stored = payloads(case)
    assert stored["Tesla Model Y"]["aliases"] == before["Tesla Model Y"]["aliases"] == ["Tesla"]
    assert project_facts(stored["Tesla Model Y"]["facts"])[0]["mention_count"] == 2
    assert stored["Honda Civic"] == before["Honda Civic"]
    assert case.observer.execute(
        "SELECT id FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s ORDER BY id",
        (case.tenant, case.user)).fetchall() == before_ids
    keywords = {r[0] for r in case.observer.execute("SELECT keyword FROM tomte_keyword_registry").fetchall()}
    assert not {"Ann", "ann", "Annette", "annette"} & keywords


def test_new_keeps_only_approved_current_user_aliases(contract):
    case = contract
    case.extracted = [proposal("Bicycle", aliases=("Speedy", "unused alias"),
                               attribute="ownership_status", value="owned")]
    case.accepted_aliases = {"temporary": ["Speedy", "invented"]}
    case.plan = None
    batch = freeze(case, ("user", "I bought a new bicycle named Speedy."))
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    assert payloads(case)["Bicycle"]["aliases"] == ["Speedy"]


def test_approved_alias_with_no_fact_plan_still_has_no_write(contract):
    seed(contract)
    contract.extracted = [proposal(aliases=("Little Dog",))]
    contract.matches = {"temporary": "Tesla Model Y"}
    contract.accepted_aliases = {"temporary": ["Little Dog"]}
    complete_noop(contract, freeze(contract, ("user", "I named my Tesla Little Dog.")))


@known_gap
def test_unsafe_correction_does_not_cancel_independent_supported_mutation_contract(contract):
    """Reject one unsafe target mutation while retaining an independent valid ADD."""
    case = contract
    seed(case)
    seed(case, "Honda Civic", "honda", "2021", ("Honda",))
    before = case.observer.execute(
        "SELECT * FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s AND entity_id=%s",
        (case.tenant, case.user, "honda"),
    ).fetchone()
    batch = freeze(case, ("user", "My Tesla Model Y is blue."),
                   ("user", "Can we talk about the Honda paperwork?"))
    case.extracted = [proposal(attribute="paint_color", value="blue", source=batch[0]["item_id"]),
                      proposal("Honda Civic", "other", value="2024", source=batch[1]["item_id"])]
    case.matches = {"temporary": "Tesla Model Y", "other": "Honda Civic"}

    def plan(target, data):
        safe = target["identity"]["entity_name"] == "Tesla Model Y"
        return [{"operation": "ADD" if safe else "CORRECT",
                 "target_fact_id": None if safe else target["assertions"][0]["fact_id"],
                 "attribute_name": "paint_color" if safe else "purchase_year",
                 "new_value": "blue" if safe else "2024",
                 "source_chat_ids": [batch[0 if safe else 1]["item_id"]],
                 "interpretation_confidence": 1, "evidence_type": "USER_EXPLICIT"}]

    case.plan = plan
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    after = payloads(case)
    assert any(f["attribute_name"] == "paint_color" and f["value"] == "blue"
               for f in project_facts(after["Tesla Model Y"]["facts"]))
    assert case.observer.execute(
        "SELECT * FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s AND entity_id=%s",
        (case.tenant, case.user, "honda"),
    ).fetchone() == before
    assert processed_count(case) == 1


def test_semantic_noop_does_not_write_entity_dictionary_feedback_contract(contract):
    """Unapproved type evidence must not change global keywords, hits or vectors."""
    case = contract
    seed(case)
    case.extracted = [proposal()]
    case.extracted[0]["entity_type"] = "electric_vehicle"
    case.matches = {"temporary": "Tesla Model Y"}
    batch = freeze(case, ("user", "Can we discuss my Tesla?"))
    before, vocabulary = rows(case), dictionary(case)
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    assert dictionary(case) == vocabulary  # Isolate dictionary authorization first.
    assert rows(case) == before
    assert processed_count(case) == 1


def test_user_confirmation_can_use_assistant_context_control(contract):
    """Assistant wording supplies context; the user's Yes owns the evidence."""
    case = contract
    seed(case, "Honda Civic", "honda", "2021", ("Honda",))
    contextual_action(case, action_type="CONFIRMATION")
    batch = freeze(case, ("assistant", "Was your Honda bought in 2021?"), ("user", "Yes."))

    def plan(target, data):
        assert data["CURRENT_BATCH"][0]["role"] == "assistant"
        return [{"operation": "REINFORCE", "target_fact_id": target["assertions"][0]["fact_id"],
                 "source_chat_ids": [batch[1]["item_id"]], "interpretation_confidence": 1,
                 "evidence_type": "RESOLVED_REFERENCE"}]

    case.plan = plan
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    fact = project_facts(payloads(case)["Honda Civic"]["facts"])[0]
    assert fact["value"] == "2021" and fact["mention_count"] == 2
    assert fact["evidence"][-1]["source_chat_id"] == batch[1]["item_id"]
    assert batch[0]["item_id"] not in {e["source_chat_id"] for e in fact["evidence"]}
    assert payloads(case)["Honda Civic"]["aliases"] == ["Honda"]
    assert processed_count(case) == 1


def test_semantic_noop_ack_preserves_later_queued_messages_contract(contract):
    """ACK of a completed empty-action snapshot preserves later queued messages."""
    case = contract
    batch_a = freeze(case, ("user", "Something non-actionable."))
    case.cache.enqueue_for_consolidation(case.tenant, case.user, case.session,
                                        "I bought a bicycle.", "user", vector=case.llm.vector())
    complete_noop(case, batch_a)  # Empty extraction + actions=[] currently completes.
    assert case.cache.clear_session_4_consolidation(case.tenant, case.user, case.session, batch_a)
    batch_b = case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session)
    assert len(batch_b) == 1 and batch_b[0]["content"] == "I bought a bicycle."
    assert batch_b[0]["item_id"] not in {m["item_id"] for m in batch_a}
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == batch_b


def test_empty_plan_drops_only_that_target_not_whole_batch(contract):
    """Accepted facts persist independently; abstained targets retain all columns."""
    case = contract
    seed(case)
    seed(case, "Honda Civic", "honda", "2021", ("Honda",))
    original = dict(zip(("tesla", "honda"), (
        case.observer.execute(
            "SELECT * FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s AND entity_id=%s",
            (case.tenant, case.user, eid),
        ).fetchone() for eid in ("tesla", "honda")
    )))
    batch = freeze(case, ("user", "My Tesla is blue. Let's discuss the Honda."))
    case.extracted = [proposal(attribute="paint_color", value="blue"),
                      proposal("Honda Civic", "other", aliases=("unapproved Honda alias",), value="2021")]
    case.matches = {"temporary": "Tesla Model Y", "other": "Honda Civic"}
    case.extracted[1]["entity_type"] = "unapproved_honda_type"

    def plan(target, data):
        if target["identity"]["entity_name"] == "Honda Civic":
            return []
        return [{"operation": "ADD", "attribute_name": "paint_color", "new_value": "blue",
                 "source_chat_ids": [batch[0]["item_id"]], "interpretation_confidence": 1,
                 "evidence_type": "USER_EXPLICIT"}]

    case.plan = plan
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    current = {eid: case.observer.execute(
        "SELECT * FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s AND entity_id=%s",
        (case.tenant, case.user, eid),
    ).fetchone() for eid in original}
    assert current["honda"] == original["honda"]
    assert current["tesla"] != original["tesla"]
    assert any(f["attribute_name"] == "paint_color" and f["value"] == "blue"
               for f in project_facts(payloads(case)["Tesla Model Y"]["facts"]))
    assert not case.observer.execute(
        "SELECT 1 FROM tomte_keyword_registry WHERE keyword=%s", ("unapproved_honda_type",)
    ).fetchall()
    assert processed_count(case) == 1


def test_processed_semantic_noop_replay_is_idempotent(contract, monkeypatch):
    """Replaying a processed no-op performs no model, graph or dictionary work."""
    case = contract
    case.junk = True
    batch = freeze(case, ("user", "Thanks!"))
    workflow = ConsolidationWorkflow()
    assert workflow.run_consolidation_cycle(case.tenant, case.user, batch)
    before, vocabulary, calls = rows(case), dictionary(case), list(case.calls)

    def forbidden(*args, **kwargs):
        raise AssertionError("Processed replay must not perform persistence or model work")

    for name in ("claim_consolidation_batch", "commit_graph_updates"):
        monkeypatch.setattr(case.db, name, forbidden)
    monkeypatch.setattr(workflow.dict_service, "apply_feedback", forbidden)
    monkeypatch.setattr(case.llm, "is_junk", forbidden)
    monkeypatch.setattr(case.llm, "parse_structured", forbidden)
    monkeypatch.setattr(case.llm, "embed_batch", forbidden)
    assert workflow.run_consolidation_cycle(case.tenant, case.user, batch)
    assert rows(case) == before and dictionary(case) == vocabulary
    assert case.calls == calls and processed_count(case) == 1


def test_mixed_batch_requires_only_user_embeddings(contract):
    """User vectors suffice; assistant context without vectors remains usable."""
    case = contract
    batch = freeze(case, ("assistant", "Are you buying a bicycle?"),
                   ("user", "I bought a new bicycle."))
    assert batch[0]["vector"] is None and batch[1]["vector"]
    case.extracted = [proposal("Bicycle", attribute="ownership_status", value="owned",
                               source=batch[1]["item_id"])]
    case.new_evidence = batch[1]["content"]
    case.plan = None
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    fact = project_facts(payloads(case)["Bicycle"]["facts"])[0]
    assert fact["value"] == "owned" and fact["source_chat_id"] == batch[1]["item_id"]
    assert processed_count(case) == 1


@pytest.mark.parametrize("same_target", [False, True])
@pytest.mark.parametrize("mixed_sources", [False, True])
def test_assistant_sourced_mutation_is_dropped_without_canceling_user_mutation(contract, same_target, mixed_sources):
    """Evidence filtering drops individual mutations, preserving independent USER work."""
    case = contract
    seed(case)
    seed(case, "Honda Civic", "honda", "2021", ("Honda",))
    original_honda = case.observer.execute(
        "SELECT * FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s AND entity_id=%s",
        (case.tenant, case.user, "honda"),
    ).fetchone()
    batch = freeze(case, ("assistant", "Your Honda was bought in 2021."),
                   ("user", "My Tesla Model Y is blue. Let's discuss the Honda."))
    case.extracted = [proposal(attribute="paint_color", value="blue", source=batch[1]["item_id"])]
    case.matches = {"temporary": "Tesla Model Y"}
    if not same_target:
        case.extracted.append(proposal("Honda Civic", "other", value="2021", source=batch[1]["item_id"]))
        case.matches["other"] = "Honda Civic"

    def plan(target, data):
        supported = target["identity"]["entity_name"] == "Tesla Model Y"
        operations = []
        if supported:
            operations.append({"operation": "ADD", "attribute_name": "paint_color", "new_value": "blue",
                "source_chat_ids": [batch[1]["item_id"]], "interpretation_confidence": 1,
                "evidence_type": "USER_EXPLICIT"})
        if same_target or not supported:
            operations.append({"operation": "REINFORCE", "target_fact_id": target["assertions"][0]["fact_id"],
                "source_chat_ids": [batch[0]["item_id"]] + ([batch[1]["item_id"]] if mixed_sources else []),
                "interpretation_confidence": 1, "evidence_type": "RESOLVED_REFERENCE"})
        return operations

    case.plan = plan
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    tesla = payloads(case)["Tesla Model Y"]
    assert any(f["attribute_name"] == "paint_color" and f["value"] == "blue" for f in project_facts(tesla["facts"]))
    assert next(f for f in tesla["facts"] if f["attribute_name"] == "purchase_year")["mention_count"] == 1
    assert case.observer.execute(
        "SELECT * FROM tomte_memory_engine WHERE tenant_id=%s AND user_id=%s AND entity_id=%s",
        (case.tenant, case.user, "honda"),
    ).fetchone() == original_honda
    assert all(e["source_chat_id"] != batch[0]["item_id"] for f in tesla["facts"] for e in f["evidence"])
    assert processed_count(case) == 1


@pytest.mark.parametrize("stage", ["extraction", "timeline", "mutation", "contextual"])
def test_foreign_source_id_remains_structural_failure_control(contract, stage):
    """Foreign provenance fails even alongside unsupported real assistant evidence."""
    case = contract
    seed(case, "Honda Civic", "honda", "2021", ("Honda",))
    batch = freeze(case, ("assistant", "Your Honda is blue."), ("user", "Let's discuss Honda Civic."))
    if stage == "contextual":
        contextual_action(case)
        original_action = case.contextual
        def foreign_action(data):
            actions = original_action(data)
            actions[0]["source_chat_ids"] = [batch[0]["item_id"], "invented-not-in-batch"]
            return actions
        case.contextual = foreign_action
    else:
        source = "invented-not-in-batch" if stage == "extraction" else batch[1]["item_id"]
        case.extracted = [proposal("Honda Civic", value="2021", source=source)]
        case.matches = {"temporary": "Honda Civic"}
        if stage == "timeline":
            # Validate foreign event IDs even when the whole fact will be omitted.
            case.extracted[0]["facts"][0]["source_chat_id"] = batch[0]["item_id"]
            case.extracted[0]["facts"][0]["timeline"] = [{"event_description": "Synthetic event",
                "source_chat_id": "invented-not-in-batch"}]
        if stage == "mutation":
            case.plan = lambda target, data: [{"operation": "REINFORCE",
                "target_fact_id": target["assertions"][0]["fact_id"],
                "source_chat_ids": [batch[0]["item_id"], "invented-not-in-batch"],
                "interpretation_confidence": 1, "evidence_type": "USER_EXPLICIT"}]
    before, vocabulary = rows(case), dictionary(case)
    with pytest.raises(ValueError):
        ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    assert rows(case) == before and dictionary(case) == vocabulary
    assert processed_count(case) == 0
    assert case.cache.pop_session_chat_log_4_consolidation(case.tenant, case.user, case.session) == batch


def test_assistant_new_evidence_does_not_cancel_user_new_entity(contract):
    """NEW quote ownership is checked per entity without judging its semantics."""
    case = contract
    batch = freeze(case, ("assistant", "You bought a new boat."), ("user", "I bought a new bicycle."))
    case.extracted = [proposal("Bicycle", "bicycle", attribute="ownership_status", value="owned", source=batch[1]["item_id"]),
                      proposal("Boat", "boat", attribute="ownership_status", value="owned", source=batch[1]["item_id"])]
    case.new_evidence = {"bicycle": batch[1]["content"], "boat": batch[0]["content"]}
    case.plan = None
    assert ConsolidationWorkflow().run_consolidation_cycle(case.tenant, case.user, batch)
    assert set(payloads(case)) == {"Bicycle"}
    assert project_facts(payloads(case)["Bicycle"]["facts"])[0]["value"] == "owned"
    assert processed_count(case) == 1


@pytest.mark.parametrize("unsupported", ["assistant_source", "mixed_sources", "assistant_replacement"])
def test_contextual_assistant_evidence_is_successful_noop(contract, unsupported):
    """Assistant-owned action sources or replacement quotes cannot authorize writes."""
    case = contract
    seed(case, "Honda Civic", "honda", "2021", ("Honda",))
    batch = freeze(case, ("assistant", "The Honda year is 2024."), ("user", "Let's discuss Honda Civic."))
    contextual_action(case)
    original_action = case.contextual
    def unsupported_action(data):
        actions = original_action(data)
        if unsupported == "assistant_replacement":
            actions[0]["replacement_text"] = "2024"
        else:
            actions[0]["source_chat_ids"] = [batch[0]["item_id"]] + ([batch[1]["item_id"]] if unsupported == "mixed_sources" else [])
        return actions
    case.contextual = unsupported_action
    complete_noop(case, batch)
