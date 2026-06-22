from src.data import _parse_evidence_list, augment_narrative
from src.inference import symptom_catalog_for_ui


def test_parse_evidence_list_from_string():
    raw = "['E_91', 'E_48', 'E_54_@_V_161']"
    parsed = _parse_evidence_list(raw)
    assert parsed == ["E_91", "E_48", "E_54_@_V_161"]


def test_parse_evidence_list_from_list():
    assert _parse_evidence_list(["E_1", "E_2"]) == ["E_1", "E_2"]


def test_augment_narrative_can_change_text():
    text = "Patient is 20. Reports fever. Reports cough. Reports fatigue."
    out = augment_narrative(text, noise_prob=1.0)
    assert isinstance(out, str)
    assert len(out) > 0


def test_ui_symptom_catalog_pins_demo_items():
    catalog = symptom_catalog_for_ui(10)
    assert catalog[0]["id"] == "E_91"
    assert catalog[1]["id"] == "E_201"