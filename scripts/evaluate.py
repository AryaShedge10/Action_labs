"""Run the Team B evaluation suite and write reports/evaluation_report.md + reports/metrics.json.

Examples:
    python scripts/evaluate.py                         # deterministic pipeline (offline)
    python scripts/evaluate.py --with-llm              # also evaluate hybrid extraction + LLM SOAP (needs Ollama)
    python scripts/evaluate.py --configs rule:template,llm:llm
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.evaluation.harness import (  # noqa: E402
    evaluate_cases,
    hallucination_injection,
    load_cases,
    render_markdown,
    safety_ablation,
)
from app.pipeline import ClinicalPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default=str(ROOT / "data" / "evaluation" / "cases.jsonl"))
    parser.add_argument("--configs", default="rule:template", help="comma-separated extraction:generation pairs")
    parser.add_argument("--with-llm", action="store_true", help="add hybrid:llm to the configurations")
    parser.add_argument("--out-dir", default=str(ROOT / "reports"))
    args = parser.parse_args()

    configs = [tuple(c.split(":")) for c in args.configs.split(",") if c]
    if args.with_llm and ("hybrid", "llm") not in configs:
        configs.append(("hybrid", "llm"))

    pipeline = ClinicalPipeline(Settings.from_env())
    needs_llm = any(e in ("llm", "hybrid") or g == "llm" for e, g in configs)
    if needs_llm and not pipeline.llm_status()["available"]:
        print("Local LLM is not available - start Ollama or drop the LLM configurations.", file=sys.stderr)
        return 2

    cases = load_cases(args.cases)
    runs = []
    for extraction_mode, generation_mode in configs:
        print(f"Evaluating {extraction_mode}:{generation_mode} on {len(cases)} cases ...", flush=True)
        runs.append(evaluate_cases(pipeline, cases, extraction_mode, generation_mode))

    deterministic = runs[0]
    injection = hallucination_injection(cases, deterministic["_analyses"])
    ablation = safety_ablation(pipeline, cases, deterministic["_analyses"], *configs[0])

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    metrics = {"runs": [{k: v for k, v in r.items() if k != "_analyses"} for r in runs],
               "hallucination_injection": injection, "safety_ablation": ablation}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out / "evaluation_report.md").write_text(render_markdown(runs, injection, ablation, len(cases)))

    for r in runs:
        e, t = r["entities"]["micro"], r["triage"]
        print(f"  {r['config']}: entity F1={e['f1']} triage acc={t['accuracy']} critical misses={t['critical_misses']} "
              f"conflict R={r['conflicts']['recall']} p50={r['latency_ms']['p50']}ms")
    print(f"  hallucination detection={injection['overall_detection_rate']} false alarms={injection['false_alarm_rate']}")
    print(f"  ablation leaks: without engine={ablation['unsafe_leaks_without_safety_engine']} "
          f"with engine={ablation['unsafe_leaks_with_safety_engine']}")
    print(f"Wrote {out / 'evaluation_report.md'} and {out / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
