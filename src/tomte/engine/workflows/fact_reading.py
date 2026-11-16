"""Read a fixed ranking in batches; meaning is judged only by the LLM."""
from collections import Counter
from dataclasses import dataclass, field

from pydantic import ValidationError

from tomte.engine.core.answer_facts import fit_packets, packet_key
from tomte.engine.core.answering import AnswerReaderUnavailable, FactBatchJudgment, validate_memory_owner_name
from tomte.engine.core.config import settings
from tomte.engine.llm.calls import call_model
from tomte.engine.llm.fact_filter_format import render_filter_input
from tomte.engine.llm.prompts.fact_reading import FACT_BATCH_READER_PROMPT

MAX_FACT_CANDIDATES = 100
FACT_BATCH_SIZE = 20


def reader_packet(handle, packet):
    return {
        'handle': handle,
        '_entity_id': packet['entity']['entity_id'],
        'entity': {k: v for k, v in packet['entity'].items() if k != 'entity_id'},
        'attribute': packet['fact']['attribute_name'],
        'value_kind': packet['value_kind'],
        'value': packet['fact']['value'],
    }


@dataclass
class BatchResult:
    """Accumulate model decisions; never infer relevance from fact text."""
    found: dict = field(default_factory=dict)
    possible: dict = field(default_factory=dict)
    last_unrelated_number: int = 0
    contains_found_or_possible: bool = False

    def append(self, batch, categories):
        self.contains_found_or_possible = False
        self.last_unrelated_number = 0
        for handle, packet in batch.items():
            category = categories.get(handle)
            if category == 'unrelated':
                self.last_unrelated_number += 1
            else:
                # Related-only and unknown entries both break an unrelated run.
                self.last_unrelated_number = 0
            if category == 'found':
                self.found[handle] = packet
                self.contains_found_or_possible = True
            elif category == 'possible':
                self.possible[handle] = packet
                self.contains_found_or_possible = True

    def too_far_from_answer(self, tail_size):
        return (not self.contains_found_or_possible
                and self.last_unrelated_number >= tail_size)


class BatchedFactReader:
    def __init__(self, llm, control, *, memory_owner_name=None):
        self.llm, self.control = llm, control
        self.memory_owner_name = validate_memory_owner_name(memory_owner_name)

    def read(self, information_needed, packets):
        """Return found, possible and diagnostics without rewriting stored facts.

        Invalid entries remain unknown; valid siblings survive. A failed call
        selects nothing from that batch and cannot establish satisfaction or low
        relevance. Cancellation, capacity and budget failures still propagate.
        """
        packets = packets[:MAX_FACT_CANDIDATES]
        labelled = {f'F{i + 1:03d}': p for i, p in enumerate(packets)}
        handles = list(labelled)
        result, coverage = BatchResult(), []
        bounded_only = True
        examined = 0
        stop = 'candidate_list_exhausted'
        config = settings.answer_retrieval
        for offset in range(0, len(handles), FACT_BATCH_SIZE):
            self.control.check()
            batch = {h: labelled[h] for h in handles[offset:offset + FACT_BATCH_SIZE]}
            kept = result.found | result.possible
            previous, omitted = fit_packets(list(kept.values()), config.memory_bytes)
            previous_keys = {packet_key(p) for p in previous}
            previous_handles = {h for h, p in kept.items() if packet_key(p) in previous_keys}
            payload = {
                'information_needed': information_needed,
                'memory_owner': 'the user whose memories are being searched',
                'previously_kept': [dict(reader_packet(h, kept[h]),
                    category='found' if h in result.found else 'possible')
                    for h in kept if h in previous_handles],
                'candidates': [reader_packet(h, p) for h, p in batch.items()],
            }
            if self.memory_owner_name is not None:
                payload['memory_owner_name'] = self.memory_owner_name
            if omitted:
                payload['previously_kept_is_partial'] = True
            diagnostic = {'batch_start': offset + 1, 'batch_end': offset + len(batch),
                          'previous_context_omissions': omitted}
            try:
                choice = call_model(self.llm, self.control, FACT_BATCH_READER_PROMPT,
                                    render_filter_input(payload), FactBatchJudgment)
            except (AnswerReaderUnavailable, ValidationError):
                result.append(batch, {})
                bounded_only = False
                diagnostic.update(reader_unavailable=True, is_partial=True)
            else:
                counts = Counter(f.handle for f in choice.facts)
                categories = {f.handle: f.category for f in choice.facts
                              if f.handle in batch and counts[f.handle] == 1}
                invalid = [h for h in batch if h not in categories]
                unknown = sorted(set(counts) - set(batch))
                result.append(batch, categories)
                bounded_only = bounded_only and choice.request_kind == 'bounded'
                supported = (choice.satisfied and bool(choice.satisfied_by)
                    and set(choice.satisfied_by) <= (previous_handles | set(batch))
                    and set(choice.satisfied_by) <= set(result.found))
                diagnostic.update(request_kind=choice.request_kind,
                    categories=[{'handle': h, 'category': categories.get(h)} for h in batch])
                if invalid or unknown:
                    diagnostic.update(invalid_handles=invalid, unknown_handles=unknown, is_partial=True)
                if choice.satisfied and not (supported and bounded_only and not omitted and not invalid):
                    diagnostic['invalid_satisfaction'] = True
                if supported and bounded_only and not omitted and not invalid:
                    stop = 'bounded_need_satisfied'
            examined += len(batch)
            coverage.append(diagnostic)
            if stop == 'bounded_need_satisfied':
                break
            if examined >= len(handles):
                stop = ('candidate_cap_100' if examined == MAX_FACT_CANDIDATES
                        else 'candidate_list_exhausted')
                break
            if result.too_far_from_answer(config.relevance_tail_size):
                stop = 'unrelated_tail'
                break
        coverage.append({'stop_reason': stop, 'examined_facts': examined,
            'selected_facts': len(result.found) + len(result.possible),
            'found_facts': len(result.found), 'possible_facts': len(result.possible),
            'candidate_facts': len(packets), 'is_partial': examined < len(packets)})
        return list(result.found.values()), list(result.possible.values()), coverage
