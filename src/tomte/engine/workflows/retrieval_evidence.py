"""Evidence transport and merging; no model calls, search, or answer decisions."""
from copy import deepcopy
from dataclasses import dataclass, field
import json

from tomte.engine.core.answer_facts import fit_packets, packet_key, render_fact_evidence
from tomte.engine.core.answer_text import render_answer_evidence
from tomte.engine.core.config import settings
from tomte.engine.core.memory import public_entity
from tomte.engine.core.memory_text import resolve_memory_text_format
from tomte.engine.workflows.memory_search import render_retrieved_memory


@dataclass
class RetrievalResult:
    views: list[dict] = field(default_factory=list)
    unresolved: list[dict] = field(default_factory=list)
    pending: dict = field(default_factory=dict)
    packets: list[dict] = field(default_factory=list)
    possible_packets: list[dict] = field(default_factory=list)
    observed_packets: list[dict] = field(default_factory=list)
    coverage: list[dict] = field(default_factory=list)
    snapshot_at: str | None = None


class EvidenceBuffer:
    """An answer owns its buffer; standalone retrieval renders a fresh one."""
    def __init__(self):
        self.views, self.unresolved, self.packets = {}, {}, {}
        self.possible_packets = {}
        self.windows, self.seen_messages, self.coverage = [], set(), []
        self.fact_coverage = []
        self.snapshot_at = None
        self.received = False
        self.output_format = resolve_memory_text_format()

    def merge(self, result):
        self.received = True
        self.snapshot_at = result.snapshot_at
        for packet in result.observed_packets:
            key = packet_key(packet)
            if key in self.packets:
                self.packets[key] = deepcopy(packet)
            if key in self.possible_packets:
                self.possible_packets[key] = deepcopy(packet)
        self.packets.update((packet_key(p), deepcopy(p)) for p in result.packets)
        for packet in result.possible_packets:
            key = packet_key(packet)
            if key not in self.packets:
                self.possible_packets[key] = deepcopy(packet)
        for key in self.packets:
            self.possible_packets.pop(key, None)
        self.fact_coverage.extend(deepcopy(result.coverage))
        for view in result.views:
            incoming = deepcopy(view)
            previous = self.views.get(view['entity_id'])
            if previous is not None:
                old_facts = {f['fact_id']: f for f in previous['facts']}
                for fact in incoming['facts']:
                    fid = fact['fact_id']
                    old = old_facts.get(fid)
                    # Preserve distinct historical indices when the snapshot still agrees.
                    if old and fact['history_values'][:len(old['history_values'])] == old['history_values']:
                        selected = incoming['_history_selection'][fid]
                        incoming['_history_selection'][fid] = sorted(set(selected)
                            | set(previous['_history_selection'].get(fid, [])))
            self.views[view['entity_id']] = incoming
        for entity in result.unresolved:
            self.unresolved[entity['entity_id']] = deepcopy(entity)
        for window in result.pending.get('pending_message_windows', []):
            entry = deepcopy(window)
            entry['messages'] = []
            for message in window['messages']:
                key = (window['session_id'], message['message_id'], message.get('char_start', 0),
                       message.get('char_end', len(message['content'])))
                if key not in self.seen_messages:
                    self.seen_messages.add(key)
                    entry['messages'].append(deepcopy(message))
            if entry['messages']:
                self.windows.append(entry)
        if 'pending_search' in result.pending:
            self.coverage.append(deepcopy(result.pending['pending_search']))

    def render_for_answer(self):
        """A smaller presentation of the same bounded evidence; no public API change."""
        if not self.received:
            return ''
        selected, possible, omitted = self._fit_facts()
        return render_answer_evidence(
            packets=selected, possible_packets=possible,
            entities=[public_entity(v, v['_history_selection']) for v in self.views.values()],
            unresolved=list(self.unresolved.values()), windows=self.windows,
            fact_coverage=self.fact_coverage, pending_coverage=self.coverage, omitted=omitted)

    def _fit_facts(self):
        selected, omitted = fit_packets(
            [*self.packets.values(), *self.possible_packets.values()], settings.answer_retrieval.memory_bytes)
        return ([p for p in selected if packet_key(p) in self.packets],
                [p for p in selected if packet_key(p) in self.possible_packets], omitted)

    def render(self):
        if not self.received:
            return ''
        evidence = {
            'entities': [public_entity(v, v['_history_selection']) for v in self.views.values()],
            'unresolved_memories': list(self.unresolved.values()),
            'pending_message_windows': self.windows,
            'pending_search': {
                'complete': bool(self.coverage) and all(c['complete'] for c in self.coverage),
                'reasons': sorted({r for c in self.coverage for r in c['reasons']}),
                'snapshot_at': self.snapshot_at,
            },
        }
        memory = render_retrieved_memory(evidence, self.output_format)
        if self.fact_coverage:
            selected, possible, omitted = self._fit_facts()
            facts = render_fact_evidence(selected, self.output_format)
            if possible:
                facts += '\n\nPOSSIBLE ANSWER CANDIDATES (not confirmed answers)\n' + render_fact_evidence(possible, self.output_format)
            memory = facts + '\n\n' + memory
            memory += '\nFACT SEARCH COVERAGE ' + json.dumps(self.fact_coverage +
                [{'accumulated_budget_omissions': omitted}], ensure_ascii=False)
        return memory
