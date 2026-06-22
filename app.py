from __future__ import annotations

import streamlit as st

from src.app_support import load_metrics, run_triage, symptom_catalog_for_ui
from src.model_assets import ensure_models

st.set_page_config(page_title="Symptom Triage", layout="wide", initial_sidebar_state="collapsed")

ALIGNED = {
    "symptoms": ["E_91", "E_201", "E_97"],
    "text": "34-year-old female with fever, cough, and sore throat for a few days.",
}
MISMATCH = {
    "symptoms": ["E_66", "E_148"],
    "text": "34-year-old female with high fever, bad cough, and sore throat for three days.",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:opsz,wght@9..40,400;0,9..40,600;0,9..40,700&family=Libre+Baskerville:wght@700&display=swap');

html, body, [class*="css"] {
  font-family: "DM Sans", system-ui, sans-serif;
}

.block-container {
  padding-top: 1.25rem;
  max-width: 1180px;
}

.hero {
  background: linear-gradient(165deg, #1a3d2f 0%, #245c42 52%, #2d6b4f 100%);
  color: #fff;
  border-radius: 16px;
  padding: 1.6rem 1.75rem 1.35rem;
  margin-bottom: 1.5rem;
  border: 1px solid rgba(255,255,255,0.1);
}
.hero h1 {
  font-family: "Libre Baskerville", Georgia, serif;
  font-size: 2rem;
  margin: 0 0 0.45rem;
  line-height: 1.15;
}
.hero p {
  margin: 0;
  color: rgba(255,255,255,0.74);
  font-size: 0.95rem;
  max-width: 40rem;
}
.tags {
  display: flex;
  flex-wrap: wrap;
  gap: 0.45rem;
  margin-top: 1rem;
}
.tag {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  padding: 0.3rem 0.6rem;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
  border: 1px solid rgba(255,255,255,0.12);
}

.section-kicker {
  font-size: 0.7rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #5a7566;
  margin: 0 0 0.75rem;
}

.stButton > button[kind="primary"] {
  border-radius: 999px;
  font-weight: 600;
}
.stButton > button[kind="secondary"] {
  border-radius: 999px;
  font-weight: 600;
  border-color: #b5cfc0;
  color: #2f6b4f;
}
</style>
"""


def _inject_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _hero(metrics: dict | None) -> None:
    metric_line = ""
    if metrics:
        acc = metrics["multimodal"]["top3_accuracy"]
        metric_line = (
            f'<p style="margin-top:0.65rem;font-size:0.82rem;color:rgba(255,255,255,0.55)">'
            f"Fusion top-3 on test split: {acc:.1%}</p>"
        )
    st.markdown(
        f"""
        <div class="hero">
          <h1>Symptom Triage</h1>
          <p>Rank conditions from a symptom checklist and what the patient actually says.</p>
          {metric_line}
          <div class="tags">
            <span class="tag">DDXPlus</span>
            <span class="tag">Structured + text + fusion</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _branch_card(branch: dict) -> None:
    top_prob = branch["differential"][0]["probability"] if branch["differential"] else 1.0
    conf = branch["confidence"] * 100

    with st.container(border=True):
        title_col, conf_col = st.columns([3, 1])
        with title_col:
            st.markdown(f"**{branch['model']}**")
        with conf_col:
            st.caption(f"{conf:.0f}% conf")

        if branch.get("abstain") and branch.get("abstain_message"):
            st.warning(branch["abstain_message"])

        for i, dx in enumerate(branch["differential"], start=1):
            pct = dx["probability"] * 100
            bar_val = min(max(dx["probability"] / top_prob if top_prob else 0.0, 0.0), 1.0)
            row_rank, row_name, row_pct = st.columns([0.35, 3, 0.9])
            with row_rank:
                st.markdown(f"**{i}**")
            with row_name:
                st.write(dx["pathology"])
            with row_pct:
                st.write(f"{pct:.1f}%")
            st.progress(bar_val)

        weights = branch.get("modality_weights")
        if weights:
            st.caption(f"text {weights['text']:.2f} · structured {weights['structured']:.2f}")


def _show_result(result: dict) -> None:
    if result["safety"]["abstain_recommended"]:
        st.warning(result["safety"]["rationale"])

    analysis = result["modality_analysis"]
    disagree_pct = analysis["disagreement_score"] * 100
    msg = f"{analysis['interpretation']} Disagreement {disagree_pct:.0f}%."
    if analysis["modalities_conflict"]:
        st.error(msg)
    else:
        st.success(msg)

    for key in ("multimodal", "structured", "text_only"):
        _branch_card(result[key])


@st.cache_resource
def _bootstrap():
    ensure_models()
    return True


_inject_styles()
_bootstrap()

metrics = load_metrics()
_hero(metrics)

catalog = symptom_catalog_for_ui(24)
catalog_ids = {s["id"] for s in catalog}

if "symptoms" not in st.session_state:
    st.session_state.symptoms = [s for s in ALIGNED["symptoms"] if s in catalog_ids]
    st.session_state.text = ALIGNED["text"]
    st.session_state.result = None

col_form, col_out = st.columns([1, 1], gap="large")

with col_form:
    st.markdown('<p class="section-kicker">Patient intake</p>', unsafe_allow_html=True)
    age_col, sex_col = st.columns(2)
    with age_col:
        age = st.number_input("Age", 1, 110, 34)
    with sex_col:
        sex = st.selectbox("Sex", ["F", "M"])
    symptoms = st.multiselect(
        "Symptoms",
        [s["id"] for s in catalog],
        default=st.session_state.symptoms,
        format_func=lambda i: next(x["label"] for x in catalog if x["id"] == i),
        key="symptom_picker",
    )
    st.session_state.symptoms = symptoms
    text = st.text_area(
        "Patient description",
        value=st.session_state.text,
        height=140,
        placeholder="What they'd say at intake — can disagree with the checklist.",
    )
    st.session_state.text = text

    btn_run, btn_mismatch, btn_reset = st.columns(3)
    run_clicked = btn_run.button("Run triage", type="primary", use_container_width=True)
    if btn_mismatch.button("Mismatch example", use_container_width=True):
        st.session_state.symptoms = [s for s in MISMATCH["symptoms"] if s in catalog_ids]
        st.session_state.text = MISMATCH["text"]
        st.session_state.result = None
        st.rerun()
    if btn_reset.button("Reset", use_container_width=True):
        st.session_state.symptoms = [s for s in ALIGNED["symptoms"] if s in catalog_ids]
        st.session_state.text = ALIGNED["text"]
        st.session_state.result = None
        st.rerun()

with col_out:
    st.markdown('<p class="section-kicker">Differential</p>', unsafe_allow_html=True)

    if run_clicked:
        if not symptoms and len(text.strip()) < 12:
            st.session_state.result = None
            st.error("Need symptoms or a longer description.")
        else:
            with st.spinner("Running structured, text, and fusion models…"):
                st.session_state.result = run_triage(age, sex, symptoms, text.strip())

    result = st.session_state.result
    if result:
        _show_result(result)
    elif not run_clicked:
        st.info("No results yet. Fill in the intake form and run triage, or load the mismatch example.")

st.caption("Sania Thankan · Penn State CDS · not medical advice")