"""Bounded interpretation of earlier unresolved evidence, before normal mutation."""
from collections import Counter
from copy import deepcopy
import json

from tomte.engine.core.config import settings
from tomte.engine.core.enums import MemoryDomain
from tomte.engine.core.memory import public_entity, public_fact, public_value
from tomte.engine.core.relationships import prepare_proposals
from tomte.engine.core.resolution import CANDIDATE_LIMIT, compatible_identity_domains
from tomte.engine.core.unresolved import pending_records, UnresolvedResolutionResponse
from tomte.engine.exceptions import TomteSecurityViolationError
from tomte.engine.llm.prompts.unresolved_resolution import UNRESOLVED_RESOLUTION_PROMPT


def prepare_unresolved_bindings(workflow, tenant_id, user_id, input_data, vector, operations):
    rows = pending_records(workflow.db, tenant_id, user_id, vector,
                           settings.unresolved_candidate_limit, settings.unresolved_recent_limit)
    pending = {row['payload']['entity_id']: row for row in rows}
    if not pending:
        return operations, pending
    current = {m['item_id']: m['content'] for m in input_data['CURRENT_BATCH'] if m['role'] == 'user'}
    context = '\n'.join(current.values())
    by_id = {op['entity_id']: op for op in operations}
    offered, views = {}, []
    for identity, row in pending.items():
        payload = row['payload']
        domain = payload['_unresolved']['original_domain']
        candidates = workflow.db.get_resolution_candidates(tenant_id, user_id,
            [payload['entity_name'].strip().lower()], domain, vector, context, CANDIDATE_LIMIT)
        targets = {}
        # Bound both sources; exceeding this window leaves the claim pending.
        for target_id, operation in by_id.items():
            if compatible_identity_domains(domain, operation['entity_domain']):
                targets[target_id] = dict(id=target_id, payload=operation['payload'])
                if len(targets) == CANDIDATE_LIMIT:
                    break
        for candidate in candidates[:CANDIDATE_LIMIT]:
            if candidate.get('tenant_id') != tenant_id or candidate.get('user_id') != user_id:
                raise TomteSecurityViolationError('Unresolved target namespace mismatch')
            value = candidate['payload']
            if str(candidate['id']) != value['entity_id']:
                raise ValueError('Unresolved target identity mismatch')
            if compatible_identity_domains(domain, value['entity_domain']) and value['entity_id'] not in targets:
                targets[value['entity_id']] = candidate
        if not targets:
            continue
        offered[identity] = targets
        target_views = []
        for candidate in targets.values():
            value = candidate['payload']
            target_views.append({key: deepcopy(value[key]) for key in
                ('entity_id', 'entity_name', 'entity_type', 'entity_domain', 'aliases')})
            target_views[-1]['facts'] = [public_fact(f, []) for f in value['facts'][:50]]
            target_views[-1]['facts_truncated'] = len(value['facts']) > 50
        views.append({'earlier_memory': public_entity(payload), 'TARGETS': target_views})
    from tomte.engine.core.capacity import CapacityExceeded, check_provider_capacity
    envelope = {'CURRENT_BATCH': input_data['CURRENT_BATCH'],
                'REFERENCE_ONLY_CONTEXT': input_data['RECENT_CONTEXT_REFERENCE_ONLY'], 'PENDING': []}
    arguments = dict(system_prompt=UNRESOLVED_RESOLUTION_PROMPT, schema=UnresolvedResolutionResponse)
    for view in views:
        trial = {**envelope, 'PENDING': envelope['PENDING'] + [view]}
        arguments['user_text'] = json.dumps(trial, ensure_ascii=False)
        try:
            check_provider_capacity('parse_structured', arguments)
        except CapacityExceeded:
            continue  # Defer a whole record, never truncate its evidence or stop current extraction.
        envelope = trial
    if not envelope['PENDING']:
        return operations, pending
    offered = {v['earlier_memory']['entity_id']: offered[v['earlier_memory']['entity_id']] for v in envelope['PENDING']}
    arguments['user_text'] = json.dumps(envelope, ensure_ascii=False)
    response = workflow._parse_proposals(**arguments)
    counts = Counter(str(binding.unresolved_id) for binding in response.bindings)
    rejected_ids = {str(raw.get('unresolved_id')) for _, raw in response.rejected_items() if isinstance(raw, dict)}
    for binding in response.bindings:
        identity, target_id = str(binding.unresolved_id), str(binding.target.db_id)
        if (counts[identity] != 1 or identity in rejected_ids or identity not in offered or target_id not in offered[identity]
                or binding.source_chat_id not in current or not binding.evidence_quote.strip()
                or binding.evidence_quote not in current[binding.source_chat_id]):
            continue
        row = pending[identity]
        payload = row['payload']
        fact = payload['facts'][0]
        proposal = dict(attribute_name=fact['attribute_name'], value=public_value(fact['value']),
            importance=fact['_importance'], source_chat_id=binding.source_chat_id,
            stored_unresolved_id=identity, reported_at=payload['_unresolved']['reported_at'],
            evidence_context='Earlier stored user statement. source_chat_id authorizes identity clarification only; it is not the original report.',
            clarification_quote=binding.evidence_quote)
        if target_id not in by_id:
            operation = workflow._prepare_operation(tenant_id, user_id, {'facts': []}, offered[identity][target_id])
            operations.append(operation)
            by_id[target_id] = operation
        by_id[target_id]['extracted_facts'].append(proposal)
    prepare_proposals(operations)
    return operations, pending
