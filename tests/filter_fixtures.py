"""Decode the reader's labelled input for boundary assertions in offline tests."""
import json


def atoms(line):
    result = []
    while line.strip():
        line = line.lstrip()
        if line[0] in ('"', '['):
            value, end = json.JSONDecoder().raw_decode(line)
            result.append(value)
            line = line[end:]
        else:
            token, _, line = line.partition(' ')
            result.append(token)
    return result


def model_input(text):
    if not text.startswith('Need '):
        return json.loads(text)
    out = {'previously_kept': [], 'candidates': []}
    entities, section, fact = {}, None, None
    for line in text.splitlines():
        tokens = atoms(line)
        if line.startswith('Need '):
            out['information_needed'] = tokens[1]
        elif line.startswith('Owner '):
            out['memory_owner_name'] = tokens[1]
        elif line == 'Entities':
            section = 'entities'
        elif line.startswith('Previous'):
            section = 'previously_kept'
            if '(partial)' in line:
                out['previously_kept_is_partial'] = True
        elif line == 'Candidates':
            section = 'candidates'
        elif line.startswith('  '):
            value = {'text': tokens[1], 'fact_confidence': 'high'}
            if len(tokens) > 2 and tokens[2] in ('medium', 'low'):
                value['fact_confidence'] = tokens.pop(2)
            if len(tokens) > 2:
                assert tokens[2] == 'context'
                value['context'] = tokens[3]
            fact['value'][tokens[0]] = value
        elif section == 'entities':
            entity = dict(entity_name=tokens[1], entity_type=tokens[2], entity_domain=tokens[3], aliases=[])
            for i in range(4, len(tokens), 2):
                entity[tokens[i]] = tokens[i + 1]
            entities[tokens[0]] = entity
        else:
            fact = dict(handle=f'F{int(tokens[0]):03d}', entity=entities[tokens[1]], attribute=tokens[2],
                        value_kind='history' if 'history' in tokens[3:] else 'current', value={})
            if section == 'previously_kept':
                fact['category'] = tokens[-1]
            out[section].append(fact)
    return out
