"""Mechanical retention and lifecycle of user-supported, identity-unresolved claims."""
from copy import deepcopy
from datetime import datetime, timezone
from uuid import UUID
from pydantic import Field

from tomte.engine.core.assertions import stable_id
from tomte.engine.core.enums import MemoryDomain
from tomte.engine.core.extraction import source_timestamps
from tomte.engine.core.memory import Entity, MemoryModel
from tomte.engine.core.proposals import ProposalResponse
from tomte.engine.core.resolution import ResolvedTarget


class UnresolvedBinding(MemoryModel):
    unresolved_id: UUID
    target: ResolvedTarget
    source_chat_id: str = Field(min_length=1)
    evidence_quote: str = Field(min_length=1, description="Verbatim current USER clarification binding this earlier claim to this target.")


class UnresolvedResolutionResponse(ProposalResponse):
    proposal_collections = {"bindings": None}
    bindings: list[UnresolvedBinding]


def pending_records(db, tenant_id, user_id, vector, limit, recent_limit=0):
    """Only unsupported legacy providers may omit this additive capability."""
    from tomte.engine.exceptions import TomteSecurityViolationError
    read = getattr(db, 'get_unresolved_memories', None)
    if read is None:
        return []
    try:
        rows = read(tenant_id, user_id, vector, limit=limit, recent_limit=recent_limit)
    except NotImplementedError:
        return []
    accepted = []
    seen = set()
    for row in rows:
        if row.get('tenant_id') != tenant_id or row.get('user_id') != user_id:
            raise TomteSecurityViolationError('Unresolved memory namespace mismatch')
        payload = Entity.model_validate(row['payload']).model_dump(mode='json')
        if str(row['id']) != payload['entity_id']:
            raise ValueError('Unresolved storage identity mismatch')
        if payload['entity_domain'] != MemoryDomain.UNRESOLVED or payload['_unresolved']['status'] != 'pending':
            continue
        if payload['entity_id'] not in seen:
            accepted.append({**row, 'payload': payload})
            seen.add(payload['entity_id'])
    return accepted[:limit + recent_limit]


def resolution_operations(operations, pending):
    """Retire only claims acknowledged by accepted mutation units in this write set."""
    destinations = {}
    for operation in operations:
        for identity in operation.get('_accepted_unresolved_ids', []):
            destinations.setdefault(identity, []).append(operation['entity_id'])
    result = []
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    for identity, targets in destinations.items():
        row = pending[identity]
        payload = deepcopy(row['payload'])
        payload['_unresolved'].update(status='resolved', resolved_at=stamp, resolved_entity_ids=sorted(set(targets)))
        result.append(dict(mode='REINFORCE', tenant_id=row['tenant_id'], user_id=row['user_id'], db_id=identity,
            payload=payload, _expected_payload=deepcopy(row['payload']), _semantic_changed=False,
            extracted_facts=[], _dictionary_evidence=[], importance=payload['facts'][0]['_importance'],
            **{key: payload[key] for key in ('entity_id', 'entity_name', 'entity_type', 'entity_domain')}))
    return result


def validate_transitions(operations):
    """Structural, scoped write-set checks, independent of LLM language decisions."""
    accepted = {}
    for op in operations:
        if op['entity_domain'] != MemoryDomain.UNRESOLVED:
            for identity in op.get('_accepted_unresolved_ids', []):
                accepted.setdefault((op['tenant_id'], op['user_id'], identity), set()).add(op['entity_id'])
    completed = set()
    for op in operations:
        payload, old = op['payload'], op.get('_expected_payload')
        if payload['entity_domain'] != MemoryDomain.UNRESOLVED:
            if old and old['entity_domain'] == MemoryDomain.UNRESOLVED:
                raise ValueError('Unresolved records cannot become entity identities')
            continue
        if op['mode'] == 'INSERT':
            if payload['_unresolved']['status'] != 'pending':
                raise ValueError('Unresolved records must start pending')
            continue
        if not old or old['entity_domain'] != MemoryDomain.UNRESOLVED:
            raise ValueError('Existing entity cannot become an unresolved record')
        expected = deepcopy(old)
        expected['_unresolved'].update(status='resolved', resolved_at=payload['_unresolved']['resolved_at'],
            resolved_entity_ids=payload['_unresolved']['resolved_entity_ids'])
        key = (op['tenant_id'], op['user_id'], op['entity_id'])
        if (old['_unresolved']['status'] != 'pending' or payload != expected
                or set(payload['_unresolved']['resolved_entity_ids']) != accepted.get(key, set())
                or not accepted.get(key)):
            raise ValueError('Resolution requires immutable pending evidence and accepted scoped target writes')
        completed.add(key)
    if completed != set(accepted):
        raise ValueError('Accepted unresolved claims and completion markers must commit together')


def retained_operations(proposals, current_batch, tenant_id, user_id, batch_id):
    """Copy validated extraction meanings; do not guess referents or interpret words."""
    timestamps = source_timestamps(current_batch)
    users = {item['item_id'] for item in current_batch if item['role'] == 'user'}
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    operations = []
    for proposal in proposals:
        entity, reason = proposal['entity'], proposal['reason']
        if entity['entity_domain'] == MemoryDomain.UNRESOLVED:
            continue
        # The explicit UNRESOLVED decision is sufficient; an omitted explanation
        # must not discard supported content. This is a protocol label, not an
        # inferred interpretation of the user's language.
        reason = reason.strip() or 'unresolved_subject_identity'
        for index, fact in enumerate(entity['facts']):
            source = fact.get('source_chat_id')
            if source not in users or not fact.get('value'):
                continue
            identity = stable_id('tomte-unresolved-v1', tenant_id, user_id, batch_id, entity['entity_id'], index)
            stored_fact = dict(fact_id=stable_id(identity, 'claim'), attribute_name=fact['attribute_name'],
                value={key: {**deepcopy(value), '_recorded_at': stamp, '_updated_at': stamp}
                       for key, value in fact['value'].items()}, history_values=[],
                _importance=fact.get('importance', 3), _mention_count=1, _last_mentioned_at=timestamps[source])
            payload = Entity.model_validate(dict(entity_id=identity, entity_name=entity['entity_name'],
                entity_type=entity['entity_type'], entity_domain=MemoryDomain.UNRESOLVED,
                aliases=[], facts=[stored_fact], unresolved_detail=reason,
                _unresolved=dict(original_domain=entity['entity_domain'], reported_at=timestamps[source])))
            payload = payload.model_dump(mode='json')
            operations.append(dict(mode='INSERT', tenant_id=tenant_id, user_id=user_id,
                reserved_id=identity, payload=payload, extracted_facts=[], _dictionary_evidence=[],
                importance=stored_fact['_importance'],
                **{key: payload[key] for key in ('entity_id', 'entity_name', 'entity_type', 'entity_domain')}))
    return operations
