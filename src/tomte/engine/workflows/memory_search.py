"""Shared scoped read/projection used by retrieval and iterative answering."""
from tomte.engine.core.config import settings
from tomte.engine.core.enums import MemoryDomain
from tomte.engine.core.memory import public_entity
from tomte.engine.core.memory_text import UNRESOLVED_HEADING, render_memory_context
from tomte.engine.core.pending_windows import render_pending_context
from tomte.engine.core.retrieval import prepare_memory_views
from tomte.engine.core.unresolved import pending_records
from tomte.engine.exceptions import TomteSecurityViolationError


def read_memory_views(tenant, user, query, vector, limit, db, *, include_unresolved=True):
    mode = settings.stored_retrieval_mode
    if mode == 'vector':
        entities = db.get_raw_semantic_context(tenant, user, vector, limit=limit)
    else:
        entities = db.get_ranked_memory_context(tenant, user, query, vector, limit, mode=mode)
    for entity in entities:
        if entity.get('tenant_id') != tenant or entity.get('user_id') != user:
            raise TomteSecurityViolationError('Retrieved entity namespace mismatch')
    entities = [e for e in entities if e['entity_domain'] != MemoryDomain.UNRESOLVED]
    # Retrieval never invokes the optional generative history selector.
    views = prepare_memory_views(entities, query, vector, tenant, user, db, llm=None, retrieval_mode=mode)
    unresolved = []
    if include_unresolved:
        unresolved = [public_entity(row['payload']) for row in pending_records(
            db, tenant, user, vector, settings.unresolved_return_limit)]
    return views, unresolved


def render_retrieved_memory(result, output_format):
    sections = [('QUERY-RELEVANT MEMORIES', result['entities'])]
    if result['unresolved_memories']:
        sections.append((UNRESOLVED_HEADING, result['unresolved_memories']))
    text = render_memory_context(sections, output_format)
    if 'pending_message_windows' in result:
        pending = render_pending_context({key: result[key] for key in
            ('pending_message_windows', 'pending_search')}, output_format)
        text = '\n\n'.join(filter(None, [text, pending]))
    return text
