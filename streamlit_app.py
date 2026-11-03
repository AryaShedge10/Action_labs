"""Team 2B developer console (Streamlit).

Run:  streamlit run streamlit_app.py

Not the physician dashboard (that is Team C's deliverable): this is a tool to exercise the
Team B pipeline on Team A transcripts and inspect exactly what Team C will receive.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from app.config import Settings  # noqa: E402
from app.evaluation.harness import load_cases  # noqa: E402
from app.ingestion.adapter import TranscriptFormatError  # noqa: E402
from app.ingestion.team_a_client import TeamAClient, TeamAError  # noqa: E402
from app.models.schemas import ClaimKind, ConflictStatus, ConsultationAnalysis, GroundingStatus  # noqa: E402
from app.pipeline import ClinicalPipeline  # noqa: E402
from app.utils.llm import LLMError  # noqa: E402

st.set_page_config(page_title="Team 2B - Clinical NLP & Safety", page_icon=":stethoscope:", layout="wide")

STATUS_UI = {
    "SAFE": ("success", "SAFE - no conflicts or grounding issues detected (physician sign-off still required)"),
    "REVIEW_REQUIRED": ("warning", "REVIEW REQUIRED - items need physician review before sign-off"),
    "CONFLICT_DETECTED": ("error", "CONFLICT DETECTED - unsafe items were blocked from the draft"),
}
GROUNDING_ICON = {
    GroundingStatus.SUPPORTED: ":white_check_mark:",
    GroundingStatus.PARTIALLY_SUPPORTED: ":large_yellow_circle:",
    GroundingStatus.UNSUPPORTED: ":x:",
    GroundingStatus.CONTRADICTED: ":no_entry:",
    GroundingStatus.NOT_APPLICABLE: ":information_source:",
}


@st.cache_resource
def get_pipeline() -> ClinicalPipeline:
    return ClinicalPipeline(Settings.from_env().with_overrides(store_results=False))


@st.cache_data
def load_samples() -> dict[str, dict]:
    samples = {f"Sample - {p.stem}": json.loads(p.read_text()) for p in sorted((ROOT / "data" / "samples").glob("*.json"))}
    cases_path = ROOT / "data" / "evaluation" / "cases.jsonl"
    if cases_path.exists():
        for case in load_cases(cases_path):
            samples[f"Eval - {case['case_id']}: {case['scenario']}"] = {
                "consultation_id": case["consultation_id"], "segments": case["segments"]}
    return samples


def ts(seconds: float) -> str:
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def citation_line(claim) -> str:
    return " | ".join(f"{c.segment_id} - {c.speaker.value} - {ts(c.start_time)}-{ts(c.end_time)}" for c in claim.citations)


# --------------------------------------------------------------------------- sidebar

pipeline = get_pipeline()
llm = pipeline.llm_status()

with st.sidebar:
    st.header("Input")
    source = st.radio("Transcript source", ["Sample", "Upload JSON", "Paste JSON", "Fetch from Team A"], index=0)
    payload: dict | None = None

    if source == "Sample":
        samples = load_samples()
        payload = samples[st.selectbox("Choose a transcript", list(samples))]
    elif source == "Upload JSON":
        uploaded = st.file_uploader("Team A transcript (.json)", type="json")
        if uploaded:
            try:
                payload = json.loads(uploaded.read())
            except ValueError:
                st.error("File is not valid JSON")
    elif source == "Paste JSON":
        text = st.text_area("Team A transcript JSON", height=220,
                            placeholder='{"consultation_id": "...", "segments": [{"speaker": "DOCTOR", ...}]}')
        if text.strip():
            try:
                payload = json.loads(text)
            except ValueError:
                st.error("Not valid JSON")
    else:
        base_url = st.text_input("Team A base URL", pipeline.settings.team_a_base_url)
        consultation_id = st.text_input("Consultation id", "20261031_143022")
        if st.button("Fetch transcript"):
            try:
                st.session_state["fetched"] = TeamAClient(base_url, pipeline.settings.team_a_timeout_s).fetch_transcript(consultation_id)
                st.session_state["fetched"].setdefault("consultation_id", consultation_id)
            except TeamAError as exc:
                st.error(str(exc))
        payload = st.session_state.get("fetched")

    st.header("Pipeline")
    extraction_mode = st.selectbox("Extraction", ["rule", "hybrid", "llm", "auto"], index=0,
                                   help="rule = deterministic (instant). hybrid = rules + quote-verified LLM additions.")
    generation_mode = st.selectbox("SOAP generation", ["template", "llm", "auto"], index=0,
                                   help="template = deterministic, cannot hallucinate. llm = fluent text, then scrubbed and verified.")
    if llm["available"]:
        st.caption(f":green_circle: Local LLM `{llm['model']}` available (LLM modes take ~1 min on a laptop)")
    else:
        st.caption(f":red_circle: Local LLM unavailable - use rule/template ({llm.get('reason', 'Ollama not reachable')})")
    run = st.button("Analyse transcript", type="primary", disabled=payload is None, width="stretch")

# --------------------------------------------------------------------------- run

st.title("Team 2B - Clinical NLP & Safety Logic")
st.caption("Team A transcript -> verified entities -> deterministic safety checks -> grounded SOAP note -> Team C JSON")

if run and payload is not None:
    with st.spinner("Running pipeline..."):
        try:
            st.session_state["result"] = pipeline.analyze(payload, extraction_mode, generation_mode)
            st.session_state["payload"] = payload
        except (TranscriptFormatError, LLMError) as exc:
            st.session_state.pop("result", None)
            st.error(str(exc))

result: ConsultationAnalysis | None = st.session_state.get("result")
if result is None:
    st.info("Choose a transcript in the sidebar and press **Analyse transcript**.")
    if payload:
        with st.expander("Transcript preview", expanded=True):
            for s in payload.get("segments", []):
                st.markdown(f"`{ts(float(s.get('start_time', 0)))}` **{s.get('speaker')}**: {s.get('text')}")
    st.stop()

# --------------------------------------------------------------------------- summary

kind, label = STATUS_UI[result.safety_status.value]
getattr(st, kind)(f"**{label}**  \nConsultation `{result.consultation_id}`")

blocked = sum(c.status == ConflictStatus.BLOCKED for c in result.conflicts)
review = len(result.conflicts) - blocked
g, conf = result.grounding, result.confidence
cols = st.columns(6)
cols[0].metric("Conflicts", f"{blocked} blocked / {review} review")
cols[1].metric("Grounding score", f"{g.grounding_score:.0%}", help="(supported + 0.5 x partial) / fact claims")
cols[2].metric("Hallucination score", f"{g.hallucination_score:.0%}", help="(unsupported + contradicted) / fact claims")
cols[3].metric("Entity coverage", f"{g.entity_coverage:.0%}", help="Extracted facts referenced in the note")
cols[4].metric("Overall confidence", f"{conf.overall:.0%}" if conf.overall is not None else "-")
cols[5].metric("Processing", f"{result.metadata.processing_ms.get('total', 0):,.0f} ms")
st.caption(f"Extraction: **{result.metadata.extraction_mode}** | Generation: **{result.metadata.generation_mode}** | "
           f"Model: {result.metadata.model or 'none (deterministic)'} | Rules v{result.metadata.rule_version} | "
           f"Transcript SHA-256 `{result.metadata.transcript_sha256[:16]}...`")
for w in result.metadata.warnings:
    st.caption(f":warning: {w}")

tabs = st.tabs(["SOAP note", f"Safety conflicts ({len(result.conflicts)})",
                f"Draft prescriptions ({len(result.draft_prescriptions)})", f"Entities ({len(result.entities)})",
                "Transcript", "Team C JSON"])

with tabs[0]:
    pi = result.patient_info
    st.markdown(f"**Patient:** sex {pi.demographics.sex or '-'} | age {pi.demographics.age or '-'} | "
                f"allergies {', '.join(pi.allergies) or 'none recorded'} | "
                f"current medications {', '.join(pi.medications) or 'none recorded'}")
    for section in ("subjective", "objective", "assessment", "plan"):
        st.subheader(section.capitalize())
        for claim in getattr(result.soap_claims, section):
            if claim.kind == ClaimKind.SAFETY:
                (st.error if claim.text.startswith("SAFETY BLOCK") else st.warning)(claim.text)
                continue
            icon = GROUNDING_ICON[claim.grounding_status]
            st.markdown(f"{icon} {claim.text}")
            detail = []
            if claim.kind == ClaimKind.FACT:
                detail.append(f"{claim.grounding_status.value.replace('_', ' ').lower()} | confidence {claim.confidence:.2f}")
            if claim.citations:
                detail.append(citation_line(claim))
            if detail:
                st.caption(" | ".join(detail))
            for issue in claim.issues:
                st.caption(f":warning: {issue}")
    if g.removed_claims:
        with st.expander(f"{len(g.removed_claims)} generated claim(s) removed by the safety scrub"):
            for rc in g.removed_claims:
                st.markdown(f"- ~~{rc.text}~~ - {rc.reason}")

with tabs[1]:
    if not result.conflicts:
        st.success("No safety conflicts detected.")
    for c in result.conflicts:
        box = st.error if c.status == ConflictStatus.BLOCKED else st.warning
        box(f"**{c.conflict_id} | {c.type.value} | {c.severity.value} | {c.status.value}**  \n{c.message}")
        if c.recommendation:
            st.markdown(f"*Recommendation:* {c.recommendation}")
        for ev in c.evidence:
            st.caption(f"{ev.segment_id} - {ev.speaker.value} - {ts(ev.start_time)}: \"{ev.quote}\"")
        st.caption(f"rule `{c.rule_id}` | confidence {c.confidence:.2f}")

with tabs[2]:
    if not result.draft_prescriptions:
        st.info("No medications were prescribed or recommended.")
    else:
        st.dataframe([{
            "id": rx.prescription_id, "medication": rx.medication, "intent": rx.intent, "status": rx.status.value,
            "dose": rx.dose, "frequency": rx.frequency, "route": rx.route, "duration": rx.duration,
            "missing": ", ".join(rx.missing_fields), "conflicts": ", ".join(rx.conflict_ids),
            "confidence": rx.confidence} for rx in result.draft_prescriptions], width="stretch", hide_index=True)
        st.caption("All drafts require physician signature. BLOCKED drafts are listed so the reason is visible.")

with tabs[3]:
    st.dataframe([{
        "id": e.entity_id, "type": e.type.value, "term": e.normalized, "said": e.text, "status": e.status.value,
        "value": f"{e.value} {e.unit or ''}".strip() if e.value else "",
        "details": ", ".join(f"{k}={v}" for k, v in e.attributes.items()),
        "confidence": e.confidence, "source": e.source, "segments": ", ".join(e.segment_ids)}
        for e in result.entities], width="stretch", hide_index=True)

with tabs[4]:
    threshold = pipeline.settings.low_asr_confidence
    for s in result.transcript:
        low = s.confidence is not None and s.confidence < threshold
        conf_txt = f"ASR {s.confidence:.2f}" if s.confidence is not None else "ASR n/a"
        st.markdown(f"`{s.segment_id}` `{ts(s.start_time)}-{ts(s.end_time)}` **{s.speaker.value}** "
                    f"{':warning: ' if low else ''}{s.text}")
        st.caption(f"{conf_txt} | sha256 {s.sha256[:16]}...")

with tabs[5]:
    body = result.model_dump(mode="json")
    st.download_button("Download JSON for Team C", json.dumps(body, indent=2),
                       file_name=f"{result.consultation_id}_team_b.json", mime="application/json")
    st.json(body, expanded=False)
