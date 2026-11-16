"""Select entities across domains, then facts; independent of answer generation."""
from copy import deepcopy
from uuid import UUID

from tomte.engine.core.catalogue import CataloguePaths, CatalogueSelection
from tomte.engine.core.config import settings
from tomte.engine.exceptions import TomteSecurityViolationError
from tomte.engine.llm.calls import call_model
from tomte.engine.llm.prompts.catalogue import CATALOGUE_SELECTION_PROMPT


def _label(value):
    """Escape row delimiters without interpreting or shortening human language."""
    if not isinstance(value, str) or not value.strip() or '\x00' in value:
        raise ValueError('Invalid catalogue label')
    return (value.replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r')
            .replace('\t', '\\t').replace('|', '\\|'))


def _uuid(value):
    if not isinstance(value, str) or str(UUID(value)) != value:
        raise ValueError('Invalid catalogue UUID')
    return value


def render_catalogue(level, labelled, entities=None):
    """Compact rows with parent headings printed once; no values or UUIDs."""
    lines, previous = [], None
    for handle, row in labelled.items():
        if level == 'entities':
            domain = row['entity_domain']
            if domain != previous:
                lines.append(_label(domain)); previous = domain
            aliases = row['aliases']
            extra = ' | aliases: ' + ', '.join(_label(a) for a in aliases) if aliases else ''
            lines.append(f"  {handle} {_label(row['entity_name'])} | {_label(row['entity_type'])}{extra}")
        else:
            eid = row['entity_id']
            if eid != previous:
                entity = entities[eid]
                lines.append(f"{_label(entity['entity_name'])} | {_label(entity['entity_type'])}")
                previous = eid
            fields = ', '.join(_label(name) for name in row['field_names'])
            lines.append(f"  {handle} {_label(row['attribute_name'])}: {fields}")
    return '\n'.join(lines)


class CatalogueNavigator:
    """One operation's cached index reads; select_paths has no evidence-search side effects."""
    def __init__(self, tenant, user, db, llm, control):
        self.tenant, self.user = tenant, user
        self.db, self.llm, self.control = db, llm, control
        self.config = settings.answer_retrieval.model_copy(deep=True)
        self.cache = {}

    def _load(self, level, parents=()):
        key = (level, tuple(sorted(parents)))
        if key not in self.cache:
            self.control.check()
            method = getattr(self.db, 'get_catalogue_' + level)
            # None explicitly means all owned domains; [] remains an empty filter.
            args = (self.tenant, self.user, None if level == 'entities' else list(key[1]))
            result = method(*args, self.config.catalog_limit)
            self.control.check()
            for row in result['rows']:
                if (row.get('tenant_id'), row.get('user_id')) != (self.tenant, self.user):
                    raise TomteSecurityViolationError('Catalogue namespace mismatch')
                if level == 'entities':
                    _uuid(row['entity_id'])
                    _label(row['entity_domain'])
                    if row['entity_domain'] == 'unresolved_memories' or not isinstance(row['aliases'], list):
                        raise ValueError('Invalid catalogue entity domain or aliases')
                    _label(row['entity_name']); _label(row['entity_type'])
                    for alias in row['aliases']: _label(alias)
                else:
                    _uuid(row['entity_id']); _uuid(row['fact_id']); _label(row['attribute_name'])
                    if row['entity_id'] not in parents or not isinstance(row['field_names'], list):
                        raise ValueError('Invalid catalogue fact parent or fields')
                    for name in row['field_names']: _label(name)
            self.cache[key] = deepcopy(result)
        return self.cache[key]

    def _select(self, level, result, request, coverage, entities=None):
        prefix = {'entities': 'E', 'facts': 'F'}[level]
        offered = {}
        for i, row in enumerate(result['rows']):
            trial = {**offered, f'{prefix}{i + 1}': row}
            if len(render_catalogue(level, trial, entities).encode('utf-8')) <= self.config.catalog_bytes:
                offered = trial
        partial = result['is_partial'] or len(offered) != len(result['rows'])
        entry = {'catalogue_level': level, 'offered': len(offered), 'is_partial': partial}
        coverage.append(entry)
        if not offered:
            return []
        read_all = self.config.catalog_read_all_facts if level == 'facts' else 1
        if len(offered) <= read_all:
            entry['selection'] = 'read_all'
            return list(offered.values())
        choice = call_model(self.llm, self.control, CATALOGUE_SELECTION_PROMPT, {
            'information_needed': request, 'level': level, 'is_partial': partial,
            'catalogue': render_catalogue(level, offered, entities),
        }, CatalogueSelection)
        if any(handle not in offered for handle in choice.handles):
            entry['invalid_handles'] = True
        return [offered[h] for h in dict.fromkeys(choice.handles) if h in offered]

    def select_paths(self, information_needed):
        """Return scoped entity/fact paths; at most one LLM selection per level."""
        self.control.check()
        result = CataloguePaths()
        if getattr(self.db, 'supports_catalogue_navigation', False) is not True:
            result.coverage.append({'catalogue_unavailable': True})
            return result
        entities = self._select('entities', self._load('entities'), information_needed, result.coverage)
        if not entities:
            return result
        by_id = {row['entity_id']: row for row in entities}
        facts = self._select('facts', self._load('facts', list(by_id)), information_needed,
                             result.coverage, by_id)
        result.paths = list({(row['entity_id'], row['fact_id']): {
            'entity_id': row['entity_id'], 'fact_id': row['fact_id'],
        } for row in facts}.values())
        return result
