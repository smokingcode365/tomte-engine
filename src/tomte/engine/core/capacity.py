"""Cheap, conservative byte limits; not a tokenizer or model context guarantee."""
import json
from tomte.engine.core.config import settings


class CapacityExceeded(ValueError):
    """Permanent refusal of a complete input; never truncate source evidence."""


def check_capacity(*texts):
    size = sum(len(text.encode('utf-8')) for text in texts if text)
    if size > settings.max_model_input_bytes:
        raise CapacityExceeded('model_input_capacity_exceeded')


def check_provider_capacity(operation, arguments):
    """Account for explicit prompt/schema inputs before native provider dispatch."""
    if operation in {'embed_text', 'embed_batch'}:
        texts = [arguments['text']] if operation == 'embed_text' else arguments['texts']
        check_capacity(*texts)
        return
    from tomte.engine.llm.trust import TRUST_BOUNDARY
    if operation == 'is_junk':
        from tomte.engine.llm.prompts.junk_filter import JUNK_FILTER_PROMPT
        check_capacity(TRUST_BOUNDARY, JUNK_FILTER_PROMPT, arguments['chat_log'])
        return
    if operation != 'parse_structured':
        return
    schema = arguments['schema']
    from tomte.engine.core.schema import TomteMemorySchema
    prompt = arguments.get('system_prompt') or ''
    if isinstance(schema, type) and issubclass(schema, TomteMemorySchema):
        from tomte.engine.llm.prompts.extractor import EXTRACTOR_SYSTEM_PROMPT, DICTIONARY_EXTRACTION_GUIDANCE
        prompt = EXTRACTOR_SYSTEM_PROMPT
        if arguments.get('taxonomy') or arguments.get('lexicon'):
            prompt += DICTIONARY_EXTRACTION_GUIDANCE + repr(arguments.get('taxonomy')) + repr(arguments.get('lexicon'))
    definition = json.dumps(schema.model_json_schema()) if hasattr(schema, 'model_json_schema') else ''
    check_capacity(TRUST_BOUNDARY, prompt, arguments['user_text'], definition)
