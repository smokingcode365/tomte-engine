"""Translate old prescribed operation fixtures into the new identity-only contract.

The fixtures still prescribe semantics, not real model quality; production never
accepts legacy mapping output. Mutation modes are determined by production code.
"""
import json


def resolution_response(schema, prescribed, user_text):
    prompt = json.loads(user_text)
    current = prompt["CURRENT_BATCH"]
    evidence = current[0]["content"] if current else None
    results=[]
    for mapping in prescribed["mappings"]:
        new = mapping["mode"] == "INSERT"
        handle = mapping["extracted_entity_id"]
        offered = {e["id"] for e in prompt["EXTRACTED_SEMANTIC_INFORMATION"]}
        if handle not in offered and "tmp_" + handle in offered: handle = "tmp_" + handle
        results.append({"extracted_entity_id":handle,
            "kind":"NEW" if new else "MATCH_ONE", "explicitly_new":new,
            "new_entity_evidence":evidence if new else None,
            "targets":[] if new else [{"db_id":mapping["matched_db_id"],
                                      "resolution_confidence":1,"evidence":"EXPLICIT"}],
            "reasoning":mapping["reasoning"], "accepted_aliases":mapping.get("accepted_aliases", []), "alias_evidence":mapping.get("alias_evidence", [])})
    return schema.model_validate({"resolutions":results})


# Prescribed fixture semantics only: production never uses this slot-based policy.
def mutation_response(schema, user_text):
    """A prescribed single-slot fixture policy, never production semantic logic."""
    data = json.loads(user_text)
    plans = []
    for target in data["TARGETS"]:
        operations = []
        for incoming in target["incoming_semantics"]:
            attribute = incoming["attribute_name"]
            existing = [f for f in target["facts"] if f["attribute_name"] == attribute]
            same = next((f for f in existing if f["value"] == incoming["value"]), None)
            chosen = same or (existing[0] if existing else None)
            kind = "MENTION" if same else "UPDATE" if chosen else "ADD"
            source = incoming.get("source_chat_id") or data["CURRENT_BATCH"][0]["item_id"]
            operation = {"operation": kind, "target_fact_id": chosen["fact_id"] if chosen else None,
                "attribute_name": attribute, "incoming_fact_key": incoming.get("incoming_fact_key"),
                "fields": [] if kind == "MENTION" else [{"field_name": k, **v} for k, v in incoming["value"].items()],
                "source_chat_ids": [source]}
            if kind != "MENTION":
                operation["importance"] = incoming.get("importance", 3)
            if kind == "UPDATE":
                operation.update(change_type="real_change", update_mode="replace_value")
            operations.append(operation)
        plans.append({"target_key": target["target_key"], "operations": operations})
    return schema.model_validate({"plans": plans})


def mutation_aware(model):
    def wrapped(system_prompt,user_text,schema,**kwargs):
        if schema.__name__ in ("TerminologyAlignmentResponse", "FieldAlignmentResponse", "ContextualActionResponse"):
            return schema()  # Prescribed KEEP_RAW; decision mechanics tested separately.
        if schema.__name__=="MutationPlanResponse":
            return mutation_response(schema,user_text)
        return model(system_prompt,user_text,schema,**kwargs)
    return wrapped
