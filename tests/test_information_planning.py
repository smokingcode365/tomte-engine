"""Planning receives actual outcomes; Python does not decide their relevance."""
import json
from unittest.mock import Mock

from test_answer_facts import packet
from tomte.engine.core.answering import AnswerDecision, RequiredInformation
from tomte.engine.core.config import settings
from tomte.engine.workflows.answering import AnswerWorkflow
from tomte.engine.workflows.information_planning import InformationPlanner
from tomte.engine.workflows.retrieval_evidence import RetrievalResult
from tomte.engine.workers.control import WorkControl


def test_empty_retrieval_is_recorded_before_follow_up_planning():
    llm, planner, retrieval = Mock(), Mock(), Mock()
    planner.extract_required_info_from_question.side_effect = [["Coco's breed"], []]
    retrieval.retrieve_once.return_value = RetrievalResult(
        pending={'pending_search': {'complete': True, 'reasons': []}})
    llm.parse_structured.return_value = AnswerDecision(
        action='retrieve', answer="Coco's breed is not established.", suggestions=["Coco's breed is missing."])
    workflow = AnswerWorkflow('tenant-a', 'user-a', Mock(), llm, Mock(), WorkControl(),
                              retrieval=retrieval, planner=planner)
    assert workflow.run("What breed is Coco?") == "Coco's breed is not established."
    retrieval.retrieve_once.assert_called_once_with("Coco's breed")
    question, results, suggestions = planner.extract_required_info_from_question.call_args.args
    assert question == 'What breed is Coco?'
    assert any("Coco's breed" in r and 'No information was returned' in r for r in results)
    assert suggestions == ["Coco's breed is missing."]
    assert llm.parse_structured.call_count == 1


def test_planner_returns_only_requirements_and_passes_outcomes_and_suggestions():
    llm = Mock()
    llm.parse_structured.side_effect = [RequiredInformation(information_needed=["Lucky's breed", "Coco's breed"]),
                                       RequiredInformation(information_needed=[])]
    planner = InformationPlanner(llm, WorkControl(), '2026-11-10T00:00:00Z')
    assert planner.extract_required_info_from_question('Same breed?', [], []) == ["Lucky's breed", "Coco's breed"]
    first = json.loads(llm.parse_structured.call_args.kwargs['user_text'])
    assert set(first) == {'question', 'reference_time_utc'}
    results = ["Lucky's breed: Labrador.", "Requested Coco's breed; no information returned."]
    suggestions = ["Coco's breed is missing."]
    assert planner.extract_required_info_from_question('Same breed?', results, suggestions) == []
    second = json.loads(llm.parse_structured.call_args.kwargs['user_text'])
    assert second['retrieval_results'] == results and second['suggestions'] == suggestions
    assert llm.parse_structured.call_args.kwargs['schema'] is RequiredInformation
    assert 'memory' not in second and 'answer' not in second


def test_list_requests_are_retrieved_in_order_and_final_allowed_result_is_answered():
    llm, planner, retrieval = Mock(), Mock(), Mock()
    planner.extract_required_info_from_question.return_value = ['first', 'second', 'third', 'fourth']
    retrieval.retrieve_once.side_effect = [RetrievalResult(packets=[packet(str(i), f'Evidence {i}')]) for i in range(3)]
    llm.parse_structured.return_value = AnswerDecision(action='retrieve', answer='Supported partial answer.', suggestions=['more'])
    workflow = AnswerWorkflow('t', 'u', Mock(), llm, Mock(), WorkControl(), retrieval=retrieval, planner=planner)
    assert workflow.run('Question') == 'Supported partial answer.'
    assert [c.args[0] for c in retrieval.retrieve_once.call_args_list] == ['first', 'second', 'third']
    payload = json.loads(llm.parse_structured.call_args.kwargs['user_text'])
    assert payload['retrievals_remaining'] == 0 and all(f'Evidence {i}' in payload['memory'] for i in range(3))
    planner.extract_required_info_from_question.assert_called_once()


def test_nonempty_result_is_not_marked_answered_and_new_supported_route_is_allowed():
    llm, planner, retrieval = Mock(), Mock(), Mock()
    planner.extract_required_info_from_question.side_effect = [["Coco's breed"], ["Coconut's breed"]]
    alias = packet('Coco', 'Coco was previously called Coconut.', attribute='alias')
    retrieval.retrieve_once.side_effect = [RetrievalResult(packets=[alias]),
                                           RetrievalResult(packets=[packet('Coco', 'Coconut is a Labrador.')])]
    llm.parse_structured.side_effect = [AnswerDecision(action='retrieve', answer='The breed is not established.', suggestions=['Search using Coconut.']),
                                       AnswerDecision(action='answer', answer='Labrador.')]
    workflow = AnswerWorkflow('t', 'u', Mock(), llm, Mock(), WorkControl(), retrieval=retrieval, planner=planner)
    assert workflow.run("What breed is Coco?") == 'Labrador.'
    _, results, suggestions = planner.extract_required_info_from_question.call_args.args
    assert 'relevance must be assessed' in results[0]
    assert results[-1].count('Coco was previously called Coconut.') == 1
    assert suggestions == ['Search using Coconut.']
    assert retrieval.retrieve_once.call_args.args == ("Coconut's breed",)


def test_structurally_invalid_and_exact_duplicate_items_do_not_discard_valid_needs():
    llm, planner, retrieval = Mock(), Mock(), Mock()
    planner.extract_required_info_from_question.return_value = ['', 'bad\x00request', 'x'*(settings.max_input_length+1), 'valid', ' valid ']
    retrieval.retrieve_once.return_value = RetrievalResult()
    llm.parse_structured.return_value = AnswerDecision(action='answer', answer='Unknown.')
    workflow = AnswerWorkflow('t', 'u', Mock(), llm, Mock(), WorkControl(), retrieval=retrieval, planner=planner)
    assert workflow.run('Question') == 'Unknown.'
    retrieval.retrieve_once.assert_called_once_with('valid')
