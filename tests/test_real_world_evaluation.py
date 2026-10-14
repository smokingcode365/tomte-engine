"""Offline checks for evaluation bookkeeping; never calls a model or service."""
import importlib
from pathlib import Path


def modules(monkeypatch):
    # Evaluation scripts import other scripts. This does not override Tomte's
    # installed package path or inject repository src into Python's path.
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    return importlib.import_module('evaluate_real_world_conversations'),importlib.import_module('evaluate_real_world_report')


def test_hidden_truth_is_separate_from_workflow_inputs(monkeypatch):
    harness,_=modules(monkeypatch)
    traces,truth=harness.load_traces()
    assert len(traces)>=12
    assert sum(len(t['turns']) for t in traces)>=600
    for trace in traces:
        assert trace['trace_id'] in truth
        assert all('hidden_labels' not in turn and 'required' not in turn for turn in trace['turns'])
        assert len({turn['session_id'] for turn in trace['turns']})==2


def test_sale_event_does_not_count_as_duplicate_vehicle(monkeypatch):
    harness,analysis=modules(monkeypatch)
    _,truth=harness.load_traces();labels=truth['long-01']
    event=dict(entity_name='Honda Civic sale',entity_domain='events_and_logistics',payload={'aliases':[]})
    vehicle=dict(entity_name='Honda Civic',entity_domain='assets_and_possessions',payload={'aliases':[]})
    assert analysis.identify(event,labels) is None
    assert analysis.identify(vehicle,labels)=='honda'


def test_exact_color_checklist_preserves_value_distinction(monkeypatch):
    harness,_=modules(monkeypatch)
    spec=dict(values=['blue'],attribute_contains=['color','colour','exterior'])
    assert harness.fact_matches(dict(attribute_name='exterior_color',value='blue'),spec)
    assert not harness.fact_matches(dict(attribute_name='exterior_color',value='dark blue'),spec)


def test_empty_candidate_sets_are_stable_not_disjoint(monkeypatch):
    _,analysis=modules(monkeypatch)
    assert analysis.jaccard([],[])==1.0
    assert analysis.jaccard(['a'],[])==0.0
    assert analysis.jaccard(['a','b'],['b','c'])==1/3
