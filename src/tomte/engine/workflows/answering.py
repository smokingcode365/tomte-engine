"""Plan information requirements, retrieve them, then answer or suggest gaps."""
from datetime import datetime, timezone

from tomte.engine.core.answering import AnswerDecision, MAX_SEARCH_ROUNDS, validate_memory_owner_name
from tomte.engine.core.config import settings
from tomte.engine.llm.calls import call_model
from tomte.engine.llm.prompts.answering import ANSWER_DECISION_PROMPT
from tomte.engine.workflows.information_planning import InformationPlanner
from tomte.engine.workflows.retrieving import MemoryRetrievalWorkflow
from tomte.engine.workflows.retrieval_evidence import EvidenceBuffer


class AnswerWorkflow:
    def __init__(self, tenant, user, db, llm, cache, control, *, retrieval=None, planner=None,
                 memory_owner_name=None):
        self.llm, self.control = llm, control
        self.memory_owner_name = validate_memory_owner_name(memory_owner_name)
        self.retrieval = retrieval if retrieval is not None else MemoryRetrievalWorkflow(
            tenant, user, db, llm, cache, control, memory_owner_name=self.memory_owner_name)
        self.reference_time = datetime.now(timezone.utc).isoformat()
        self.planner = planner if planner is not None else InformationPlanner(
            llm, control, self.reference_time, memory_owner_name=self.memory_owner_name)

    def decide_next_step(self, question, memory, remaining):
        payload = {
            'question': question, 'memory': memory,
            'retrievals_remaining': remaining, 'reference_time_utc': self.reference_time,
        }
        if self.memory_owner_name is not None:
            payload['memory_owner_name'] = self.memory_owner_name
        return call_model(self.llm, self.control, ANSWER_DECISION_PROMPT, payload, AnswerDecision)

    def run(self, question):
        evidence, requests, outcomes = EvidenceBuffer(), [], []
        suggestions, best_answer = [], None
        while True:
            results = list(outcomes)
            if evidence.received:
                # Include the merged evidence once, not once per overlapping search.
                results.append('Available information from these searches:\n' + evidence.render_for_answer())
            needs = self.planner.extract_required_info_from_question(question, results, suggestions)
            searched = False
            for need in needs:
                if len(requests) >= MAX_SEARCH_ROUNDS:
                    break
                request = need.strip()
                if (not request or '\x00' in request or len(request) > settings.max_input_length
                        or request in requests):
                    continue
                result = self.retrieval.retrieve_once(request)
                evidence.merge(result)
                requests.append(request)
                has_evidence = bool(result.packets or result.possible_packets or result.views or result.unresolved
                    or result.pending.get('pending_message_windows'))
                outcome = ('Evidence was returned; relevance must be assessed.' if has_evidence
                           else 'No information was returned; this does not prove absence.')
                outcomes.append(f'Requested: {request}\nResult: {outcome}')
                searched = True
            if not searched and best_answer is not None:
                return best_answer
            remaining = MAX_SEARCH_ROUNDS - len(requests)
            decision = self.decide_next_step(question, evidence.render_for_answer(), remaining)
            best_answer = decision.answer
            # Always read the final retrieval, including the third, before stopping.
            if decision.action == 'answer' or remaining == 0 or not decision.suggestions:
                return decision.answer
            suggestions = decision.suggestions
