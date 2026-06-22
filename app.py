import streamlit as st

from src.app_support import load_metrics, run_triage, symptom_catalog_for_ui
from src.model_assets import ensure_models

st.set_page_config(page_title="Symptom Triage", layout="wide")

ALIGNED = {
    "symptoms": ["E_91", "E_201", "E_97"],
    "text": "34-year-old female with fever, cough, and sore throat for a few days.",
}
MISMATCH = {
    "symptoms": ["E_66", "E_148"],
    "text": "34-year-old female with high fever, bad cough, and sore throat for three days.",
}


@st.cache_resource
def _bootstrap():
    ensure_models()
    return True


def _branch_card(branch: dict) -> None:
    st.markdown(f"**{branch['model']}**")
    if branch.get("abstain"):
        st.caption(branch["abstain_message"])
    for i, dx in enumerate(branch["differential"], start=1):
        st.write(f"{i}. {dx['pathology']} — {dx['probability'] * 100:.1f}%")
    weights = branch.get("modality_weights")
    if weights:
        st.caption(
            f"text weight {weights['text']:.2f}, structured {weights['structured']:.2f}"
        )


_bootstrap()
st.title("Symptom Triage")
st.caption(
    "Structured symptoms + patient narrative → top-3 differential. "
    "Try the mismatch preset to see the models disagree."
)

metrics = load_metrics()
catalog = symptom_catalog_for_ui(24)
catalog_ids = {s["id"] for s in catalog}
if metrics:
    st.caption(f"test top-3 (fusion): {metrics['multimodal']['top3_accuracy']:.1%}")

if "symptoms" not in st.session_state:
    st.session_state.symptoms = [s for s in ALIGNED["symptoms"] if s in catalog_ids]
    st.session_state.text = ALIGNED["text"]

col_form, col_out = st.columns([1, 1], gap="large")

with col_form:
    age = st.number_input("Age", 1, 110, 34)
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
        placeholder="what they'd say at intake",
    )
    st.session_state.text = text

    btn_run, btn_mismatch, btn_reset = st.columns(3)
    run_clicked = btn_run.button("Run", type="primary")
    if btn_mismatch.button("Mismatch example"):
        st.session_state.symptoms = [s for s in MISMATCH["symptoms"] if s in catalog_ids]
        st.session_state.text = MISMATCH["text"]
        st.rerun()
    if btn_reset.button("Reset"):
        st.session_state.symptoms = [s for s in ALIGNED["symptoms"] if s in catalog_ids]
        st.session_state.text = ALIGNED["text"]
        st.rerun()

with col_out:
    if run_clicked:
        if not symptoms and len(text.strip()) < 12:
            st.error("Need symptoms or a longer description.")
        else:
            with st.spinner("Running models…"):
                result = run_triage(age, sex, symptoms, text.strip())

            if result["safety"]["abstain_recommended"]:
                st.warning(result["safety"]["rationale"])

            analysis = result["modality_analysis"]
            st.info(
                f"{analysis['interpretation']} "
                f"(disagreement {analysis['disagreement_score'] * 100:.0f}%)"
            )

            for key in ("multimodal", "structured", "text_only"):
                with st.container(border=True):
                    _branch_card(result[key])