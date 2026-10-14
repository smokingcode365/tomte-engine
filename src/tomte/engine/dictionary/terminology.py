"""Conservative attribute aliases until the semantic dictionary has a durable alias map.

Only reviewed equivalent keys belong here. Topic proximity is not synonymy.
"""
from copy import deepcopy

ATTRIBUTE_ALIASES = {
    "ownership_status": frozenset({"ownership_status", "ownership", "vehicle_ownership"}),
}


def canonical_attribute(name):
    for canonical, aliases in ATTRIBUTE_ALIASES.items():
        if name in aliases:
            return canonical
    return name


def normalize_facts(facts):
    result = {}
    for original in facts:
        fact = deepcopy(original)
        key = canonical_attribute(fact["attribute_name"])
        fact["attribute_name"] = key
        # A completed sale is past as an event, but Sold describes current ownership.
        if key == "ownership_status" and fact.get("status") == "past" and str(fact["value"]).lower() in {"sold", "no longer owned"}:
            fact["status"] = "active"
        if key in result:
            if result[key]["value"] != fact["value"]:
                raise ValueError("Conflicting synonymous attributes require explicit resolution")
            fact = merge_facts([result[key]], [fact])[0]
        result[key] = fact
    return list(result.values())


def merge_facts(old_facts, incoming):
    old = {f["attribute_name"]: deepcopy(f) for f in normalize_facts(old_facts)}
    for fact in normalize_facts(incoming):
        previous = old.get(fact["attribute_name"])
        if previous and fact["attribute_name"] in ATTRIBUTE_ALIASES:
            history = deepcopy(previous.get("timeline", []))
            if previous["value"] != fact["value"]:
                history.append({"event_description": f"Previous {previous['attribute_name']}: {previous['value']}",
                                "source_chat_id": previous["source_chat_id"],
                                "timestamp": previous["last_mentioned_at"]})
            for event in fact.get("timeline", []):
                if event not in history:
                    history.append(event)
            fact["timeline"] = history
        old[fact["attribute_name"]] = fact
    return list(old.values())
