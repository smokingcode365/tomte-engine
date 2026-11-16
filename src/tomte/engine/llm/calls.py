"""Shared structured-call mechanics; prompts and decisions belong to callers."""
import json

from tomte.engine.core.capacity import check_provider_capacity


def call_model(llm, control, prompt, payload, schema):
    control.check()
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    kwargs = dict(system_prompt=prompt, user_text=text, schema=schema)
    check_provider_capacity('parse_structured', kwargs)
    result = llm.parse_structured(**kwargs)
    control.check()
    return schema.model_validate(result)
