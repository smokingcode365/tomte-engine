"""Bounded, read-only question answering with model-owned semantic decisions."""
from copy import deepcopy
from datetime import datetime, timezone
import json

from tomte.engine.core.answering import AnswerSearchPlan, AnswerDecision, MAX_SEARCH_ROUNDS, MAX_SEARCHES_PER_ROUND
from tomte.engine.core.capacity import check_provider_capacity
from tomte.engine.core.config import settings
from tomte.engine.core.memory import public_entity
from tomte.engine.core.memory_text import resolve_memory_text_format
from tomte.engine.llm.prompts.answering import ANSWER_SEARCH_PROMPT, ANSWER_DECISION_PROMPT
from tomte.engine.workflows.memory_search import read_memory_views, render_retrieved_memory
from tomte.engine.workflows.pending_search import PendingMemorySearch


class AnswerWorkflow:
    def __init__(self, tenant, user, db, llm, cache, control):
        self.tenant, self.user = tenant, user
        self.db, self.llm, self.cache, self.control = db, llm, cache, control
        self.views, self.unresolved = {}, {}
        self.windows, self.seen_messages, self.coverage = [], set(), []
        self.searched = []
        self.output_format = resolve_memory_text_format()

    def _call(self, prompt, payload, schema):
        self.control.check()
        kwargs = dict(system_prompt=prompt, user_text=json.dumps(payload, ensure_ascii=False), schema=schema)
        # Custom providers get the same explicit capacity check as native ones.
        check_provider_capacity('parse_structured', kwargs)
        result = self.llm.parse_structured(**kwargs)
        self.control.check()
        return schema.model_validate(result)

    def _queries(self, proposed):
        result = []
        for text in proposed:
            text = text.strip()
            # These are structural limits, not relevance or language decisions.
            if (not text or '\x00' in text or len(text) > settings.max_input_length
                    or text in self.searched or text in result):
                continue
            result.append(text)
            if len(result) == MAX_SEARCHES_PER_ROUND:
                break
        return result

    def _remember(self, views, unresolved, pending):
        for view in views:
            incoming = deepcopy(view)
            previous = self.views.get(view['entity_id'])
            if previous is not None:
                old_facts = {f['fact_id']: f for f in previous['facts']}
                for fact in incoming['facts']:
                    fid = fact['fact_id']
                    old = old_facts.get(fid)
                    # Only union offered indices if the old history is still an
                    # unchanged prefix. Identical text in distinct entries stays distinct.
                    if old and fact['history_values'][:len(old['history_values'])] == old['history_values']:
                        selected = incoming['_history_selection'][fid]
                        incoming['_history_selection'][fid] = sorted(set(selected)
                            | set(previous['_history_selection'].get(fid, [])))
            self.views[view['entity_id']] = incoming
        for entity in unresolved:
            self.unresolved[entity['entity_id']] = deepcopy(entity)
        for window in pending['pending_message_windows']:
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
        self.coverage.append(pending['pending_search'])

    def _evidence(self):
        return {
            'entities': [public_entity(v, v['_history_selection']) for v in self.views.values()],
            'unresolved_memories': list(self.unresolved.values()),
            'pending_message_windows': self.windows,
        }

    def run(self, question):
        self.control.check()
        # Capture BEFORE any durable retrieval; retain this source for all rounds.
        pending = PendingMemorySearch(self.tenant, self.user, self.db, self.cache, self.control)
        reference_time = datetime.now(timezone.utc).isoformat()
        plan = self._call(ANSWER_SEARCH_PROMPT, {
            'question': question, 'reference_time_utc': reference_time,
        }, AnswerSearchPlan)
        queries = self._queries(plan.searches) or [question]
        for round_number in range(1, MAX_SEARCH_ROUNDS + 1):
            for query in queries:
                self.control.check()
                self.searched.append(query)
                vector = self.llm.embed_text(query)
                pending_result = pending.search(vector, self.llm)
                self.control.check()
                views, unresolved = read_memory_views(self.tenant, self.user, query, vector,
                    settings.semantic_retrieval_limit, self.db)
                self._remember(views, unresolved, pending_result)
            evidence = self._evidence()
            remaining = MAX_SEARCH_ROUNDS - round_number
            evidence['pending_search'] = {
                'complete': all(c['complete'] for c in self.coverage),
                'reasons': sorted({r for c in self.coverage for r in c['reasons']}),
                'snapshot_at': pending.snapshot_at,
            }
            decision = self._call(ANSWER_DECISION_PROMPT, {
                'question': question, 'reference_time_utc': reference_time,
                'previous_searches': list(self.searched), 'search_rounds_remaining': remaining,
                'memory': render_retrieved_memory(evidence, self.output_format),
            }, AnswerDecision)
            if decision.action == 'answer' or remaining == 0 or not decision.missing_information.strip():
                return decision.answer
            queries = self._queries(decision.searches)
            if not queries:
                return decision.answer
        raise AssertionError('Answer loop must return within its fixed round limit')
