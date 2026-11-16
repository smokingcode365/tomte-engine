"""Identify information requirements; retrieval and answering remain separate."""
from tomte.engine.core.answering import RequiredInformation, validate_memory_owner_name
from tomte.engine.llm.calls import call_model
from tomte.engine.llm.prompts.answering import REQUIRED_INFORMATION_PROMPT


class InformationPlanner:
    def __init__(self, llm, control, reference_time, *, memory_owner_name=None):
        self.llm, self.control, self.reference_time = llm, control, reference_time
        self.memory_owner_name = validate_memory_owner_name(memory_owner_name)

    def extract_required_info_from_question(
            self, question: str, retrieval_results: list[str], suggestions: list[str]) -> list[str]:
        payload = {'question': question, 'reference_time_utc': self.reference_time}
        if self.memory_owner_name is not None:
            payload['memory_owner_name'] = self.memory_owner_name
        if retrieval_results:
            payload['retrieval_results'] = list(retrieval_results)
        if suggestions:
            payload['suggestions'] = list(suggestions)
        return call_model(self.llm, self.control, REQUIRED_INFORMATION_PROMPT,
                          payload, RequiredInformation).information_needed
