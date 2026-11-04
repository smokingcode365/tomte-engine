"""Transient proposal correlation and explicit group atomicity; no memory edges."""
from copy import deepcopy
import json


def prepare_proposals(operations):
    mapping = {handle: op.get("db_id") or op["reserved_id"]
               for op in operations for handle in op.get("_single_handles", [])}
    for op in operations:
        accepted, seen = [], set()
        for original in op["extracted_facts"]:
            fact = deepcopy(original)
            participants = fact.get("group_participant_handles", []) or fact.get("_group_handles", [])
            targets = set(fact.get("_group_targets", []))
            if participants and (any(p not in mapping for p in participants) or
                                 (targets and {mapping[p] for p in participants} != targets)):
                continue
            if targets:
                fact["_group_key"] = json.dumps([sorted(targets), fact["attribute_name"], fact["value"], fact["source_chat_id"]], sort_keys=True)
            signature = json.dumps({k: v for k, v in fact.items() if k not in
                ("evidence_context", "group_participant_handles", "_group_handles")}, sort_keys=True)
            if signature not in seen:
                fact["incoming_fact_key"] = f"fact_{len(accepted)}"
                accepted.append(fact)
                seen.add(signature)
        op["extracted_facts"] = accepted
    return operations, mapping


def incomplete_groups(operations):
    """Identify failed shared claims without dropping unrelated entity changes."""
    groups = {}
    for op in operations:
        for key in op.get("_accepted_group_keys", []):
            groups.setdefault(key, set()).add(op.get("db_id") or op.get("reserved_id"))
    return {key for op in operations for key, targets in op.get("_required_groups", {}).items()
            if not set(targets) <= groups.get(key, set())}
