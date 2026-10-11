"""Conservative source support for model-proposed destructive corrections.

This is an English evidence gate, not a semantic truth classifier. Uncertain
corrections fail before persistence; assertion mechanics remain in assertions.py.
"""
import re


class UnsupportedCorrection(ValueError):
    """The current batch cannot safely authorize invalidating an assertion."""


def _contains(text, phrase):
    return bool(phrase and re.search(r"(?<!\w)" + re.escape(phrase.strip()) + r"(?!\w)", text, re.I))


def _names(identity):
    payload = identity.get("_expected_payload", identity.get("payload", {}))
    return [identity.get("entity_name", ""), *payload.get("aliases", [])]


def _mentions(text, identities):
    return {index for index, identity in enumerate(identities)
            if any(_contains(text, name) for name in _names(identity) if name.strip())}


def _value_present(text, value):
    if _contains(text, value):
        return True
    # A single inflected word can differ from the planner's scalar label (own/owned).
    # Multiword paraphrases are deliberately not inferred by this lexical gate.
    if re.fullmatch(r"[a-zA-Z]+", value) and value.lower().endswith("ed"):
        return _contains(text, value[:-2]) or _contains(text, value[:-1])
    return False


def validate_corrections(mutations, target, identities, current_batch, reference=()):
    """Require corrective user text and an unambiguous target/value binding.

    Explicit target names/aliases bind locally. A pronoun-only correction may use
    the most recent preceding named turn, but competing names veto that binding.
    Older reference turns are referent context only, never replacement evidence.
    No confidence score authorizes a correction without source support.
    """
    def identity_key(identity):
        return str(identity.get("db_id", identity.get("id", identity.get("entity_id"))))
    unique = {identity_key(target): target}
    for identity in identities:
        unique.setdefault(identity_key(identity), identity)
    identities = list(unique.values())
    target_index = 0
    by_source = {item["item_id"]: index for index, item in enumerate(current_batch)}
    cue = r"\b(?:actually|correction|correct(?:ed)?|wrong|mistaken|misunderstood|wasn['’]t|was not|not .+? (?:but|it was))\b|^\s*no\b"
    hypothetical = r"\b(?:if|suppose|hypothetically|imagine|maybe|perhaps|might|could)\b"
    for mutation in mutations:
        if mutation.operation != "CORRECT":
            continue
        supported = False
        for source in mutation.source_chat_ids:
            index = by_source.get(source)
            if index is None:
                continue  # Existing assertion validation separately rejects foreign IDs.
            item = current_batch[index]
            text = item["content"]
            if (item["role"].lower() != "user" or "?" in text or
                    re.search(hypothetical, text, re.I) or not re.search(cue, text, re.I)):
                continue
            clauses = re.split(r"[.!?;\n]|\b(?:and|but)\b", text, flags=re.I)
            for clause in clauses:
                if mutation.new_value is not None:
                    if not _value_present(clause, mutation.new_value):
                        continue
                    if re.search(r"\b(?:not|wasn['’]t)\s+" + re.escape(mutation.new_value) + r"(?!\w)", clause, re.I):
                        continue
                else:
                    if not re.search(cue, clause, re.I):
                        continue
                    old = next((f for f in target.get("payload", {}).get("facts", [])
                                if f.get("fact_id") == mutation.target_fact_id), None)
                    # Invalidating without replacement still needs evidence about
                    # this assertion, not merely 'the paperwork is wrong'.
                    if old is None or not (
                            _value_present(clause, str(old["value"])) or any(
                                _mentions(prior["content"], identities) == {target_index}
                                and _value_present(prior["content"], str(old["value"]))
                                for prior in [*reference, *current_batch[:index]])):
                        continue
                mentioned = _mentions(clause, identities)
                if not mentioned:
                    # Context may identify 'it', but may not supply a new value.
                    # Names elsewhere in this source outrank older reference turns.
                    mentioned = _mentions(text, identities)
                    if not mentioned:
                        for prior in reversed([*reference, *current_batch[:index]]):
                            mentioned = _mentions(prior["content"], identities)
                            if mentioned:
                                break
                if mentioned == {target_index}:
                    supported = True
                    break
            if supported:
                break
        if not supported:
            raise UnsupportedCorrection("CORRECT lacks unambiguous current user target/value evidence")
