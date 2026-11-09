"""Lossless reader representations of already-projected public memory views."""
import json
from typing import Literal

MemoryTextFormat = Literal['labelled_rows', 'compact_json']

ROWS_GUIDE = '''Memory rows use JSON-escaped arrays with these columns:
entity: entity_id, entity_name, entity_domain, entity_type, aliases, additional entity fields.
fact: fact_id, attribute_name, history_is_partial, additional fact fields.
current: current values. history: next historical value, with any additional history fields.
field: field name, text, fact_confidence. Each entity/fact heading starts a new record.
'''

UNRESOLVED_HEADING = (
    'UNRESOLVED MEMORIES — earlier user reports with missing attribution; '
    'do not assume they describe any particular known entity.'
)


def resolve_memory_text_format(output_format: MemoryTextFormat | None = None) -> MemoryTextFormat:
    if output_format is None:
        from tomte.engine.core.config import settings
        output_format = settings.memory_text_format
    if output_format not in ('labelled_rows', 'compact_json'):
        raise ValueError('Memory text format must be labelled_rows or compact_json')
    return output_format


def _dump(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def _labelled_rows(entities: list[dict]) -> str:
    lines = []
    entity_keys = ('entity_id', 'entity_name', 'entity_domain', 'entity_type', 'aliases')
    fact_keys = ('fact_id', 'attribute_name', 'history_is_partial')
    for entity in entities:
        extra = {k: v for k, v in entity.items() if k not in (*entity_keys, 'facts')}
        lines.append('entity ' + _dump([*[entity[k] for k in entity_keys], extra]))
        for fact in entity['facts']:
            extra = {k: v for k, v in fact.items() if k not in (*fact_keys, 'value', 'history_values')}
            lines.append('fact ' + _dump([*[fact[k] for k in fact_keys], extra]))
            lines.append('current')
            for key, value in fact['value'].items():
                lines.append('field ' + _dump([key, value['text'], value['fact_confidence']]))
            for history in fact['history_values']:
                lines.append('history ' + _dump({k: v for k, v in history.items() if k != 'value'}))
                for key, value in history['value'].items():
                    lines.append('field ' + _dump([key, value['text'], value['fact_confidence']]))
    return '\n'.join(lines)


def render_memory_context(
    sections: list[tuple[str, list[dict]]], output_format: MemoryTextFormat | None = None,
) -> str:
    """Render public views only; callers must apply public_entity before this boundary.

    Section titles are application constants, never memory-derived headings.
    Keep the row legend once for the entire packet. Empty packets stay empty.
    """
    output_format = resolve_memory_text_format(output_format)
    if not any(entities for _, entities in sections):
        return ''
    blocks = []
    for title, entities in sections:
        body = (_dump(entities) if output_format == 'compact_json' else _labelled_rows(entities)) if entities else ''
        blocks.append((title + '\n' if title else '') + body)
    return (ROWS_GUIDE if output_format == 'labelled_rows' else '') + '\n\n'.join(blocks)
