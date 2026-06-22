import streamlit as st

from src.app_support import load_metrics, run_triage, symptom_catalog_for_ui

st.set_page_config(page_title="Symptom Triage", layout="wide")
st.title("Symptom Triage")
st.caption("Quick backend test — the React app in frontend/ is what I actually show people.")

metrics = load_metrics()
catalog = symptom_catalog_for_ui(30)
if metrics:
    st.caption(f"last test top-3 (fusion): {metrics['multimodal']['top3_accuracy']:.1%}")

age = st.number_input("Age", 1, 110, 34)
sex = st.selectbox("Sex", ["F", "M"])
symptoms = st.multiselect(
    "Symptoms",
    [s["id"] for s in catalog],
    format_func=lambda i: next(x["label"] for x in catalog if x["id"] == i),
)
text = st.text_area("Patient description", height=120, placeholder="what they'd actually say at intake")

if st.button("Run", type="primary"):
    st.json(run_triage(age, sex, symptoms, text))