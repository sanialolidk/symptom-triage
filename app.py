from __future__ import annotations

import html

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
  background: linear-gradient(165deg, #1c2434 0%, #2a3548 55%, #1f2a3d 100%);
  color: #fff;
  border-radius: 16px;
  padding: 1.6rem 1.75rem 1.35rem;
  margin-bottom: 1.5rem;
  border: 1px solid rgba(255,255,255,0.08);
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
  color: #6b7a8f;
  margin: 0 0 0.75rem;
}

.card {
  background: #fff;
  border: 1px solid #e4ddd3;
  border-radius: 14px;
  padding: 1rem 1.1rem;
  margin-bottom: 0.75rem;
  box-shadow: 0 1px 2px rgba(28,36,52,0.05);
}
.card.featured {
  border-color: #d4b0a6;
  box-shadow: 0 8px 24px rgba(28,36,52,0.08);
}
.card-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 0.75rem;
  margin-bottom: 0.65rem;
}
.card-head h3 {
  margin: 0;
  font-family: "Libre Baskerville", Georgia, serif;
  font-size: 1.05rem;
  color: #1c2434;
}
.pill {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.03em;
  text-transform: uppercase;
  padding: 0.25rem 0.55rem;
  border-radius: 999px;
  background: #e8f2ed;
  color: #3d6b5a;
  white-space: nowrap;
}
.pill.low {
  background: #fdf4e7;
  color: #7a5518;
}

.dx-row {
  display: grid;
  grid-template-columns: 1.4rem 1fr auto;
  gap: 0.45rem;
  align-items: center;
  font-size: 0.84rem;
  margin-bottom: 0.35rem;
}
.dx-rank { font-weight: 700; color: #7a2e1c; font-size: 0.78rem; }
.dx-name { color: #1c2434; font-weight: 500; }
.dx-pct { font-weight: 700; color: #3d4a5c; font-variant-numeric: tabular-nums; }
.bar-track {
  grid-column: 2 / -1;
  height: 5px;
  border-radius: 999px;
  background: #e4ddd3;
  overflow: hidden;
  margin-bottom: 0.35rem;
}
.bar-fill {
  display: block;
  height: 100%;
  border-radius: inherit;
  background: linear-gradient(90deg, #b84a32, #d4765f);
}

.banner {
  border-radius: 10px;
  padding: 0.8rem 0.95rem;
  font-size: 0.88rem;
  margin-bottom: 0.75rem;
  border: 1px solid;
}
.banner strong { display: block; margin-bottom: 0.15rem; }
.banner.warn { background: #fdf4e7; border-color: #e8c98a; color: #7a5518; }
.banner.conflict { background: #f3e4df; border-color: #ddb8ae; color: #7a2e1c; }
.banner.ok { background: #e8f2ed; border-color: #b9d5c8; color: #3d6b5a; }

.empty {
  text-align: center;
  color: #6b7a8f;
  padding: 2.5rem 1.5rem;
  border: 1px dashed #cfc4b6;
  border-radius: 14px;
  background: rgba(255,255,255,0.55);
}
.empty strong {
  display: block;
  font-family: "Libre Baskerville", Georgia, serif;
  color: #3d4a5c;
  font-size: 1.1rem;
  margin-bottom: 0.35rem;
}

div[data-testid="stVerticalBlock"] > div:has(> div.intake-panel) {
  background: #fff;
  border: 1px solid #e4ddd3;
  border-radius: 14px;
  padding: 0.25rem 0.75rem 0.75rem;
}

.stButton > button[kind="primary"] {
  border-radius: 999px;
  font-weight: 600;
}
.stButton > button[kind="secondary"] {
  border-radius: 999px;
  font-weight: 600;
  border-color: #cfc4b6;
  color: #7a2e1c;
}
</style>
"""


def _inject_styles() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def _hero(metrics: dict | None) -> None:
    metric_line = ""
    if metrics:
        acc = metrics["multimodal"]["top3_accuracy"]
        metric_line = f'<p style="margin-top:0.65rem;font-size:0.82rem;color:rgba(255,255,255,0.55)">Fusion top-3 on test split: {acc:.1%}</p>'
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


def _banner(kind: str, title: str, body: str) -> None:
    st.markdown(
        f'<div class="banner {kind}"><strong>{html.escape(title)}</strong>{html.escape(body)}</div>',
        unsafe_allow_html=True,
    )


def _branch_card(branch: dict, featured: bool = False) -> None:
    top_prob = branch["differential"][0]["probability"] if branch["differential"] else 1.0
    conf = branch["confidence"] * 100
    pill_class = "pill low" if branch.get("abstain") else "pill"
    rows = []
    for i, dx in enumerate(branch["differential"], start=1):
        pct = dx["probability"] * 100
        width = (dx["probability"] / top_prob) * 100 if top_prob else 0
        rows.append(
            f"""
            <div class="dx-row">
              <span class="dx-rank">{i}</span>
              <span class="dx-name">{html.escape(dx["pathology"])}</span>
              <span class="dx-pct">{pct:.1f}%</span>
            </div>
            <div class="bar-track"><span class="bar-fill" style="width:{width:.1f}%"></span></div>
            """
        )
    weights_html = ""
    weights = branch.get("modality_weights")
    if weights:
        weights_html = (
            f'<p style="margin:0.6rem 0 0;font-size:0.8rem;color:#6b7a8f">'
            f"text {weights['text']:.2f} · structured {weights['structured']:.2f}</p>"
        )
    abstain_html = ""
    if branch.get("abstain") and branch.get("abstain_message"):
        abstain_html = f'<p style="margin:0 0 0.5rem;font-size:0.82rem;color:#7a5518">{html.escape(branch["abstain_message"])}</p>'

    st.markdown(
        f"""
        <div class="card{" featured" if featured else ""}">
          <div class="card-head">
            <h3>{html.escape(branch["model"])}</h3>
            <span class="{pill_class}">{conf:.0f}% conf</span>
          </div>
          {abstain_html}
          {''.join(rows)}
          {weights_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    with st.container(border=False):
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
        if result["safety"]["abstain_recommended"]:
            _banner("warn", "Low confidence", result["safety"]["rationale"])

        analysis = result["modality_analysis"]
        banner_kind = "conflict" if analysis["modalities_conflict"] else "ok"
        banner_title = "Modalities conflict" if analysis["modalities_conflict"] else "Modalities agree"
        _banner(
            banner_kind,
            banner_title,
            f"{analysis['interpretation']} Disagreement {analysis['disagreement_score'] * 100:.0f}%.",
        )

        for key in ("multimodal", "structured", "text_only"):
            _branch_card(result[key], featured=(key == "multimodal"))
    elif not run_clicked:
        st.markdown(
            """
            <div class="empty">
              <strong>No results yet</strong>
              Fill in the intake form and run triage, or load the mismatch example.
            </div>
            """,
            unsafe_allow_html=True,
        )

st.caption("Sania Thankan · Penn State CDS · not medical advice")