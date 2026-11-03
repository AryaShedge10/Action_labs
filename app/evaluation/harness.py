"""Evaluation harness for the Team B pipeline.

Three experiments:
1. Labelled-case evaluation: entity P/R/F1, conflict P/R/F1, triage (safety_status) accuracy,
   critical misses, grounding and latency.
2. Hallucination injection: adversarial claims (fabricated diagnosis / number / citation /
   medication, negation flip, family misattribution, blocked item) are fed to the grounding
   verifier; we measure detection rate, plus the false-alarm rate on generated claims.
3. Safety ablation: the same cases with the safety engine disabled, counting unsafe items that
   leak into the note or draft prescriptions.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from app.grounding.verifier import GroundingVerifier
from app.ingestion.adapter import to_transcript
from app.models.schemas import (
    Claim,
    ClaimKind,
    ClinicalEntity,
    ConflictStatus,
    ConsultationAnalysis,
    EntitySet,
    EntityStatus,
    EntityType,
    GroundingStatus,
    PrescriptionStatus,
    SOAPNote,
    SOAPSection,
)
from app.nlp.matcher import clinical_matcher
from app.nlp.text import numbers_in
from app.pipeline import ClinicalPipeline

_STATUS_RANK = {"SAFE": 0, "REVIEW_REQUIRED": 1, "CONFLICT_DETECTED": 2}
_DETECTED = {GroundingStatus.UNSUPPORTED, GroundingStatus.CONTRADICTED}


def load_cases(path: str | Path) -> list[dict[str, Any]]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def entity_key(e: ClinicalEntity) -> tuple[str, str]:
    key = f"{e.normalized}:{e.value}" if e.type == EntityType.DEMOGRAPHIC else e.normalized
    return e.type.value, key


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    p = tp / (tp + fp) if tp + fp else 1.0
    r = tp / (tp + fn) if tp + fn else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 3), "recall": round(r, 3), "f1": round(f, 3), "tp": tp, "fp": fp, "fn": fn}


def _match_conflicts(expected: list[dict], predicted: list[dict]) -> tuple[int, list[dict], list[dict]]:
    remaining = list(predicted)
    missed = []
    tp = 0
    for exp in expected:
        hit = next((p for p in remaining if p["type"] == exp["type"] and p["status"] == exp["status"]
                    and (exp.get("medication") is None or (p.get("medication") or "").lower() == exp["medication"])), None)
        if hit:
            tp += 1
            remaining.remove(hit)
        else:
            missed.append(exp)
    return tp, missed, remaining


# --------------------------------------------------------------------------- 1. labelled cases

def evaluate_cases(pipeline: ClinicalPipeline, cases: list[dict], extraction_mode: str = "rule",
                   generation_mode: str = "template") -> dict[str, Any]:
    ent_counts: dict[str, Counter] = defaultdict(Counter)
    status_correct = status_total = 0
    conflict_tp = conflict_fp = conflict_fn = 0
    blocked_expected = blocked_found = 0
    confusion: dict[str, Counter] = defaultdict(Counter)
    critical_misses, under_triage, over_triage = [], [], []
    grounding_scores, hallucination_scores, coverage, latencies = [], [], [], []
    failures, analyses = [], {}

    for case in cases:
        result = pipeline.analyze(case, extraction_mode, generation_mode)
        analyses[case["case_id"]] = result
        exp = case["expected"]
        latencies.append(result.metadata.processing_ms.get("total", 0.0))
        if result.grounding.fact_claims:
            grounding_scores.append(result.grounding.grounding_score)
            hallucination_scores.append(result.grounding.hallucination_score)
        if result.entities:
            coverage.append(result.grounding.entity_coverage)

        # entities
        predicted = {entity_key(e): e.status.value for e in result.entities}
        expected = {(t, k): s for t, k, s in exp["entities"]}
        for key in expected.keys() | predicted.keys():
            etype = key[0]
            if key in expected and key in predicted:
                ent_counts[etype]["tp"] += 1
                ent_counts[etype]["status_correct"] += int(expected[key] == predicted[key])
            elif key in expected:
                ent_counts[etype]["fn"] += 1
            else:
                ent_counts[etype]["fp"] += 1

        # conflicts
        pred_conflicts = [{"type": c.type.value, "status": c.status.value, "medication": c.medication}
                          for c in result.conflicts]
        tp, missed, extra = _match_conflicts(exp["conflicts"], pred_conflicts)
        conflict_tp, conflict_fn, conflict_fp = conflict_tp + tp, conflict_fn + len(missed), conflict_fp + len(extra)
        n_blocked = sum(c["status"] == "BLOCKED" for c in exp["conflicts"])
        blocked_expected += n_blocked
        blocked_found += n_blocked - sum(m["status"] == "BLOCKED" for m in missed)

        # triage
        want, got = exp["safety_status"], result.safety_status.value
        status_total += 1
        status_correct += int(want == got)
        confusion[want][got] += 1
        if want == "CONFLICT_DETECTED" and got == "SAFE":
            critical_misses.append(case["case_id"])
        if _STATUS_RANK[got] < _STATUS_RANK[want]:
            under_triage.append(case["case_id"])
        elif _STATUS_RANK[got] > _STATUS_RANK[want]:
            over_triage.append(case["case_id"])

        missing_ents = sorted(f"{t}:{k}" for (t, k) in expected.keys() - predicted.keys())
        extra_ents = sorted(f"{t}:{k}" for (t, k) in predicted.keys() - expected.keys())
        wrong_status = sorted(f"{t}:{k} expected {expected[(t, k)]} got {predicted[(t, k)]}"
                              for (t, k) in expected.keys() & predicted.keys() if expected[(t, k)] != predicted[(t, k)])
        if want != got or missed or extra or missing_ents or extra_ents or wrong_status:
            failures.append({"case_id": case["case_id"], "scenario": case["scenario"], "expected_status": want,
                             "predicted_status": got, "missed_conflicts": missed, "extra_conflicts": extra,
                             "missing_entities": missing_ents, "extra_entities": extra_ents, "wrong_status": wrong_status})

    per_type = {}
    totals = Counter()
    for etype, c in sorted(ent_counts.items()):
        per_type[etype] = {**_prf(c["tp"], c["fp"], c["fn"]),
                           "status_accuracy": round(c["status_correct"] / c["tp"], 3) if c["tp"] else None}
        totals.update(c)
    return {
        "config": {"extraction_mode": extraction_mode, "generation_mode": generation_mode},
        "n_cases": len(cases),
        "entities": {"per_type": per_type,
                     "micro": {**_prf(totals["tp"], totals["fp"], totals["fn"]),
                               "status_accuracy": round(totals["status_correct"] / totals["tp"], 3) if totals["tp"] else None}},
        "conflicts": {**_prf(conflict_tp, conflict_fp, conflict_fn),
                      "blocked_recall": round(blocked_found / blocked_expected, 3) if blocked_expected else None},
        "triage": {"accuracy": round(status_correct / status_total, 3) if status_total else None,
                   "critical_misses": critical_misses, "under_triage": under_triage, "over_triage": over_triage,
                   "confusion": {k: dict(v) for k, v in confusion.items()}},
        "grounding": {"mean_grounding_score": _mean(grounding_scores),
                      "mean_hallucination_score": _mean(hallucination_scores),
                      "mean_entity_coverage": _mean(coverage)},
        "latency_ms": {"p50": _pct(latencies, 50), "p95": _pct(latencies, 95), "max": round(max(latencies), 1) if latencies else None},
        "failures": failures,
        "_analyses": analyses,
    }


def _mean(values: list[float]) -> float | None:
    return round(statistics.fmean(values), 4) if values else None


def _pct(values: list[float], q: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1, int(round(q / 100 * (len(ordered) - 1))))], 1)


# --------------------------------------------------------------------------- 2. hallucination injection

_FAKE_DIAGNOSES = ["pneumonia", "migraine", "asthma", "hypertension", "influenza"]
_FAKE_DRUGS = ["prednisone", "doxycycline", "metoprolol", "omeprazole"]


def hallucination_injection(cases: list[dict], analyses: dict[str, ConsultationAnalysis]) -> dict[str, Any]:
    verifier = GroundingVerifier()
    matcher = clinical_matcher()
    detected: Counter = Counter()
    total: Counter = Counter()
    false_alarms = generated_facts = 0
    examples: dict[str, str] = {}

    for case in cases:
        analysis = analyses[case["case_id"]]
        transcript = to_transcript(case)
        entities = EntitySet(items=analysis.entities)
        blocked = {eid for c in analysis.conflicts if c.status == ConflictStatus.BLOCKED for eid in c.blocked_entity_ids}
        said = {m.canonical for s in transcript.segments for m in matcher.find(s.text)}
        transcript_numbers = numbers_in(" ".join(s.text for s in transcript.segments))
        first_seg = [transcript.segments[0].segment_id]

        for claim in analysis.soap_claims.all_claims():
            if claim.kind == ClaimKind.FACT:
                generated_facts += 1
                false_alarms += int(claim.grounding_status in _DETECTED)

        injections: list[tuple[str, Claim]] = []
        dx = next((d for d in _FAKE_DIAGNOSES if d not in said), None)
        if dx:
            injections.append(("fabricated_diagnosis", Claim(section=SOAPSection.ASSESSMENT,
                                                             text=f"Clinician assessment: {dx}.", segment_ids=first_seg)))
        drug = next((d for d in _FAKE_DRUGS if d not in said), None)
        if drug:
            injections.append(("fabricated_medication", Claim(section=SOAPSection.PLAN, text=f"Prescribe {drug}.",
                                                              segment_ids=first_seg)))
        base = next((c for c in analysis.soap_claims.subjective if c.kind == ClaimKind.FACT and c.entity_ids), None)
        if base and "14" not in transcript_numbers:
            injections.append(("fabricated_number", Claim(section=base.section, text=base.text.rstrip(".") + " for 14 days.",
                                                          entity_ids=base.entity_ids, segment_ids=base.segment_ids)))
        if base:
            injections.append(("fabricated_citation", Claim(section=base.section, text=base.text, segment_ids=["S999"])))
        for e in entities.items:
            if e.type == EntityType.SYMPTOM and e.status == EntityStatus.ABSENT:
                injections.append(("negation_flip", Claim(section=SOAPSection.SUBJECTIVE, text=f"Patient reports {e.normalized}.",
                                                          entity_ids=[e.entity_id], segment_ids=e.segment_ids)))
                break
        for e in entities.items:
            if e.status == EntityStatus.FAMILY_MEMBER and e.type in (EntityType.MEDICATION, EntityType.DIAGNOSIS):
                verb = "takes" if e.type == EntityType.MEDICATION else "has"
                injections.append(("family_misattribution", Claim(section=SOAPSection.SUBJECTIVE,
                                                                  text=f"Patient {verb} {e.normalized}.",
                                                                  entity_ids=[e.entity_id], segment_ids=e.segment_ids)))
                break
        for eid in sorted(blocked):
            e = entities.get(eid)
            if e is not None and e.type == EntityType.MEDICATION:
                injections.append(("blocked_item_asserted", Claim(section=SOAPSection.PLAN, text=f"Prescribe {e.normalized}.",
                                                                  entity_ids=[e.entity_id], segment_ids=e.segment_ids)))
                break

        for kind, claim in injections:
            note = SOAPNote()
            note.section(claim.section).append(claim)
            verifier.verify(note, transcript, entities, blocked)
            total[kind] += 1
            detected[kind] += int(claim.grounding_status in _DETECTED)
            examples.setdefault(kind, f"{claim.text}  ->  {claim.grounding_status.value}: {'; '.join(claim.issues[:1])}")

    per_kind = {k: {"detected": detected[k], "total": total[k], "rate": round(detected[k] / total[k], 3)}
                for k in sorted(total)}
    all_total = sum(total.values())
    return {
        "per_kind": per_kind,
        "overall_detection_rate": round(sum(detected.values()) / all_total, 3) if all_total else None,
        "generated_fact_claims": generated_facts,
        "false_alarm_rate": round(false_alarms / generated_facts, 3) if generated_facts else None,
        "examples": examples,
    }


# --------------------------------------------------------------------------- 3. safety ablation

def _leaks(case: dict, result: ConsultationAnalysis) -> list[str]:
    leaks = []
    fact_text = " ".join(c.text.lower() for c in result.soap_claims.all_claims() if c.kind == ClaimKind.FACT)
    for exp in case["expected"]["conflicts"]:
        if exp["status"] != "BLOCKED":
            continue
        term = exp.get("medication") or ("pregnan" if exp["type"] == "DEMOGRAPHIC_CONFLICT" else None)
        if term is None:
            continue
        in_note = term in fact_text
        in_rx = any(rx.medication.lower() == term and rx.status != PrescriptionStatus.BLOCKED
                    for rx in result.draft_prescriptions)
        if in_note or in_rx:
            leaks.append(term)
    return leaks


def safety_ablation(pipeline: ClinicalPipeline, cases: list[dict], analyses: dict[str, ConsultationAnalysis],
                    extraction_mode: str = "rule", generation_mode: str = "template") -> dict[str, Any]:
    risky = [c for c in cases if any(x["status"] == "BLOCKED" for x in c["expected"]["conflicts"])]
    with_engine = sum(len(_leaks(c, analyses[c["case_id"]])) for c in risky)
    without = []
    for c in risky:
        result = pipeline.analyze(c, extraction_mode, generation_mode, safety_enabled=False)
        without.extend(f"{c['case_id']}:{t}" for t in _leaks(c, result))
    return {"cases_with_expected_blocks": len(risky), "unsafe_leaks_with_safety_engine": with_engine,
            "unsafe_leaks_without_safety_engine": len(without), "leaked_without_engine": without}


# --------------------------------------------------------------------------- report

def render_markdown(runs: list[dict[str, Any]], injection: dict[str, Any], ablation: dict[str, Any],
                    n_cases: int) -> str:
    lines = ["# Team 2B - Evaluation Report", "",
             "> Generated by `python scripts/evaluate.py`. All numbers below are computed, not estimated.", "",
             "## Dataset", "",
             f"- **{n_cases} labelled consultations** in Team A's transcript format (`data/evaluation/cases.jsonl`).",
             "- Sources: Team A's real sample transcript, the sample agreed by email with Team A, the 12 legacy "
             "gold/adversarial cases (migrated and relabelled), and authored adversarial cases covering every safety rule.",
             "- **Limitation:** the cases were written by the same team that wrote the rules, so these results measure "
             "*regression safety and rule coverage*, not real-world generalisation. A held-out set (e.g. MTS-Dialog "
             "converted with `scripts/convert_mts_dialog.py`, labelled by someone else) is the next step.", ""]

    lines += ["## Pipeline configurations", "",
              "| Config | Entity P | Entity R | Entity F1 | Status acc. | Conflict P | Conflict R | Blocked recall | "
              "Triage acc. | Critical misses | Grounding | Hallucination | Coverage | p50 ms | p95 ms |",
              "|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for run in runs:
        e, c, t, g, lat = run["entities"]["micro"], run["conflicts"], run["triage"], run["grounding"], run["latency_ms"]
        name = f"{run['config']['extraction_mode']} + {run['config']['generation_mode']}"
        lines.append(f"| {name} | {e['precision']:.2f} | {e['recall']:.2f} | {e['f1']:.2f} | {e['status_accuracy']:.2f} | "
                     f"{c['precision']:.2f} | {c['recall']:.2f} | {c['blocked_recall']:.2f} | {t['accuracy']:.2f} | "
                     f"{len(t['critical_misses'])} | {g['mean_grounding_score']:.2f} | {g['mean_hallucination_score']:.2f} | "
                     f"{g['mean_entity_coverage']:.2f} | {lat['p50']} | {lat['p95']} |")
    lines += ["", "*Critical miss* = a case that should be CONFLICT_DETECTED but was reported SAFE.", ""]

    for run in runs:
        name = f"{run['config']['extraction_mode']} + {run['config']['generation_mode']}"
        lines += [f"### Entity extraction by type ({name})", "",
                  "| Type | Precision | Recall | F1 | Status accuracy | TP | FP | FN |", "|---|--:|--:|--:|--:|--:|--:|--:|"]
        for etype, m in run["entities"]["per_type"].items():
            sa = f"{m['status_accuracy']:.2f}" if m["status_accuracy"] is not None else "-"
            lines.append(f"| {etype} | {m['precision']:.2f} | {m['recall']:.2f} | {m['f1']:.2f} | {sa} | "
                         f"{m['tp']} | {m['fp']} | {m['fn']} |")
        lines += ["", f"Triage confusion (rows = expected, columns = predicted): `{json.dumps(run['triage']['confusion'])}`", ""]
        if run["failures"]:
            lines += [f"**Cases with any deviation ({len(run['failures'])}):**", ""]
            for f in run["failures"]:
                detail = []
                if f["expected_status"] != f["predicted_status"]:
                    detail.append(f"status {f['expected_status']} -> {f['predicted_status']}")
                for k in ("missing_entities", "extra_entities", "wrong_status"):
                    if f[k]:
                        detail.append(f"{k.replace('_', ' ')}: {', '.join(f[k])}")
                if f["missed_conflicts"]:
                    detail.append("missed conflicts: " + ", ".join(m["type"] for m in f["missed_conflicts"]))
                if f["extra_conflicts"]:
                    detail.append("extra conflicts: " + ", ".join(m["type"] for m in f["extra_conflicts"]))
                lines.append(f"- `{f['case_id']}` ({f['scenario']}): " + "; ".join(detail))
            lines.append("")

    lines += ["## Hallucination detection (claim-level grounding verifier)", "",
              "Adversarial claims were injected into notes and verified against the transcript.", "",
              "| Injected hallucination | Detected | Total | Rate |", "|---|--:|--:|--:|"]
    for kind, m in injection["per_kind"].items():
        lines.append(f"| {kind.replace('_', ' ')} | {m['detected']} | {m['total']} | {m['rate']:.2f} |")
    lines += ["", f"- **Overall detection rate:** {injection['overall_detection_rate']:.2f}",
              f"- **False-alarm rate on generated claims:** {injection['false_alarm_rate']:.2f} "
              f"({injection['generated_fact_claims']} generated FACT claims)", "", "Examples:", ""]
    lines += [f"- `{k}`: {v}" for k, v in injection["examples"].items()]

    lines += ["", "## Ablation: safety engine on vs off", "",
              "Unsafe items (blocked medications / contradicted conditions) that reach the note or a non-blocked draft prescription.", "",
              "| Architecture | Unsafe leaks |", "|---|--:|",
              f"| Pipeline without safety engine | {ablation['unsafe_leaks_without_safety_engine']} / "
              f"{ablation['cases_with_expected_blocks']} risky cases |",
              f"| Full pipeline (safety engine on) | {ablation['unsafe_leaks_with_safety_engine']} / "
              f"{ablation['cases_with_expected_blocks']} risky cases |", ""]
    if ablation["leaked_without_engine"]:
        lines.append(f"Leaked without the engine: {', '.join(ablation['leaked_without_engine'])}")
    return "\n".join(lines) + "\n"
