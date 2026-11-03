"""Convert MTS-Dialog (https://github.com/abachaa/MTS-Dialog, CC BY-NC-SA 4.0) into Team A's transcript format.

MTS-Dialog has no audio, so timestamps are synthesised from word counts (~2.6 words/s) and ASR
confidence is left empty. Output: one Team A JSON transcript per line (JSONL), usable with
POST /v1/analyze, the Streamlit console, or as a held-out set to be labelled.

    python scripts/convert_mts_dialog.py MTS-Dialog-TrainingSet.csv data/external/mts_dialog.jsonl --limit 200
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

_TURN = re.compile(r"^\s*(Doctor|Patient|Guest_family|Guest_clinician|Nurse)\s*:\s*(.*)$", re.IGNORECASE)
_SPEAKERS = {"doctor": "DOCTOR", "patient": "PATIENT", "nurse": "NURSE", "guest_clinician": "DOCTOR",
             "guest_family": "UNKNOWN"}


def dialogue_to_segments(dialogue: str) -> list[dict]:
    segments, t = [], 0.0
    current: list[str] | None = None
    for line in dialogue.splitlines():
        m = _TURN.match(line)
        if m:
            current = [_SPEAKERS[m.group(1).lower()], m.group(2).strip()]
            segments.append(current)
        elif current is not None and line.strip():
            current[1] += " " + line.strip()
    out = []
    for speaker, text in segments:
        if not text:
            continue
        duration = round(max(1.0, len(text.split()) / 2.6), 2)
        out.append({"speaker": speaker, "start_time": round(t, 2), "end_time": round(t + duration, 2),
                    "text": text, "confidence": None})
        t += duration + 0.4
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv_path")
    parser.add_argument("out_path")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(args.csv_path, newline="", encoding="utf-8") as src, out.open("w") as dst:
        for row in csv.DictReader(src):
            segments = dialogue_to_segments(row.get("dialogue", ""))
            if not segments:
                continue
            dst.write(json.dumps({
                "consultation_id": f"mts_{row.get('ID', written)}",
                "segments": segments,
                "reference_note": {"section": row.get("section_header"), "text": row.get("section_text")},
            }) + "\n")
            written += 1
            if args.limit and written >= args.limit:
                break
    print(f"wrote {written} transcripts to {out}")


if __name__ == "__main__":
    main()
