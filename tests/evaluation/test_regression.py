"""Regression gate on the labelled evaluation set (deterministic pipeline, runs offline in ~1 s).

Thresholds are deliberately strict on safety: no critical misses and no missed BLOCKED conflicts.
"""

import pytest

from app.evaluation.harness import evaluate_cases, hallucination_injection, load_cases, safety_ablation
from tests.conftest import ROOT

CASES = load_cases(ROOT / "data" / "evaluation" / "cases.jsonl")


@pytest.fixture(scope="module")
def run(pipeline):
    return evaluate_cases(pipeline, CASES, "rule", "template")


def test_no_critical_misses(run):
    assert run["triage"]["critical_misses"] == []
    assert run["conflicts"]["blocked_recall"] == 1.0


def test_triage_and_conflicts(run):
    assert run["triage"]["accuracy"] >= 0.95
    assert run["conflicts"]["recall"] >= 0.95
    assert run["conflicts"]["precision"] >= 0.9


def test_entity_extraction(run):
    assert run["entities"]["micro"]["f1"] >= 0.95
    assert run["entities"]["micro"]["status_accuracy"] >= 0.95


def test_generated_notes_are_grounded(run):
    assert run["grounding"]["mean_hallucination_score"] == 0.0
    assert run["grounding"]["mean_entity_coverage"] >= 0.99


def test_hallucination_injection_detected(run):
    result = hallucination_injection(CASES, run["_analyses"])
    assert result["overall_detection_rate"] >= 0.95
    assert result["false_alarm_rate"] <= 0.05


def test_safety_engine_prevents_unsafe_leaks(pipeline, run):
    result = safety_ablation(pipeline, CASES, run["_analyses"])
    assert result["unsafe_leaks_with_safety_engine"] == 0
    assert result["unsafe_leaks_without_safety_engine"] > 0  # proves the ablation actually exercises the engine
