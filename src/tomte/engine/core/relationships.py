"""Batch-local identity mapping; no inference from names or legacy labels."""
from copy import deepcopy
import json


def normalize_relationships(operations):
    mapping = {}
    reserved = [op['reserved_id'] for op in operations if op.get('reserved_id')]
    for op in operations:
        op['_batch_reserved_ids'] = reserved
        for handle in op.get('_single_handles', []):
            mapping[handle] = op.get('db_id') or op['reserved_id']
    for op in operations:
        accepted, rejected, seen = [], [], set()
        for original in op['extracted_facts']:
            fact = deepcopy(original)
            refs = fact.get('related_entity_ids', [])
            participants = fact.get('group_participant_handles', []) or fact.get('_group_handles', [])
            targets = set(fact.get('_group_targets', []))
            unsafe = any(ref not in mapping for ref in refs + participants)
            if participants and targets and {mapping.get(p) for p in participants} != targets:
                unsafe = True
            if unsafe:
                rejected.append(fact)
                continue
            fact['related_entity_uuids'] = sorted({mapping[ref] for ref in refs})
            fact['related_entity_ids'] = []
            if targets:
                fact['_group_key'] = json.dumps([sorted(targets), fact['attribute_name'], fact['value'], fact['source_chat_id']])
            # Repeated facets mapping to one UUID produce one semantic proposal.
            signature = json.dumps({k:v for k,v in fact.items() if k not in
                ('evidence_context', 'group_participant_handles', '_group_handles')}, sort_keys=True)
            if signature not in seen:
                fact['incoming_fact_key'] = f'fact_{len(accepted)}'
                accepted.append(fact)
                seen.add(signature)
        op['extracted_facts'] = accepted
        op['_rejected_facts'] = rejected
    return [op for op in operations if op['extracted_facts'] or op.get('reserved_id') or not op['_rejected_facts']], mapping


def prune_missing_new_targets(operations, reserved_ids):
    """Fail closed on dependent operations if their NEW endpoint was abstained.

    Dropping the entire prepared operation also avoids persisting a supersession
    without its replacement. Independent operations survive. Iterate for chains.
    """
    remaining = operations
    while True:
        present = {op.get('reserved_id') for op in remaining}
        accepted_ids = {op.get('db_id') or op.get('reserved_id') for op in remaining}
        groups = {}
        for op in remaining:
            for key in op.get('_accepted_group_keys', []):
                groups.setdefault(key, set()).add(op.get('db_id') or op.get('reserved_id'))
        missing = reserved_ids - present
        safe = [op for op in remaining if not any(
            set(f.get('related_entity_uuids', [])) & missing for f in op['payload']['facts'])]
        safe = [op for op in safe if all(set(group) <= accepted_ids for group in op.get('_group_sets', []))]
        safe = [op for op in safe if all(set(targets) <= groups.get(key, set())
                for key, targets in op.get('_required_groups', {}).items())]
        referenced = {uid for op in safe for f in op['payload']['facts'] for uid in f.get('related_entity_uuids', [])}
        safe = [op for op in safe if op.get('db_id') or op['payload']['facts'] or op.get('reserved_id') in referenced]
        if len(safe) == len(remaining):
            return safe
        remaining = safe
