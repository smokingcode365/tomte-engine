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
        results.append({"extracted_entity_id":mapping["extracted_entity_id"],
            "kind":"NEW" if new else "MATCH_ONE", "explicitly_new":new,
            "new_entity_evidence":evidence if new else None,
            "targets":[] if new else [{"db_id":mapping["matched_db_id"],
                                      "resolution_confidence":1,"evidence":"EXPLICIT"}],
            "reasoning":mapping["reasoning"]})
    return schema.model_validate({"resolutions":results})


# Prescribed fixture semantics only: production never uses this slot-based policy.
def mutation_response(schema, user_text):
    from tomte.engine.dictionary.terminology import canonical_attribute
    data=json.loads(user_text)
    sources={m["item_id"] for m in data["CURRENT_BATCH"]}
    plans=[]
    for target in data["TARGETS"]:
        operations=[]
        for incoming in target["incoming_semantics"]:
            attribute=canonical_attribute(incoming["attribute_name"])
            active=[f for f in target["assertions"] if f["assertion_state"]=="ACTIVE" and f["attribute_name"]==attribute]
            same=next((f for f in active if f["value"]==incoming["value"]),None)
            kind="REINFORCE" if same else "SUPERSEDE" if active else "ADD"
            chosen=same or (active[0] if active else None)
            source=incoming.get("source_chat_id")
            if source not in sources:source=data["CURRENT_BATCH"][0]["item_id"]
            operations.append({"operation":kind,"target_fact_id":chosen["fact_id"] if chosen else None,
                "attribute_name":attribute,"new_value":None if kind=="REINFORCE" else incoming["value"],
                "source_chat_ids":[source],"interpretation_confidence":incoming.get("confidence",1),
                "evidence_type":"USER_EXPLICIT"})
        plans.append({"target_key":target["target_key"],"operations":operations})
    return schema.model_validate({"plans":plans})


def mutation_aware(model):
    def wrapped(system_prompt,user_text,schema,**kwargs):
        if schema.__name__ == "TerminologyAlignmentResponse":
            return schema()  # Prescribed KEEP_RAW; decision mechanics tested separately.
        if schema.__name__=="MutationPlanResponse":
            return mutation_response(schema,user_text)
        return model(system_prompt,user_text,schema,**kwargs)
    return wrapped
