import { useMemo, useState } from "react";
import { api } from "../api/client";
import type { Meta, TriageResponse } from "../types";

interface Props {
  meta: Meta;
}

const ALIGNED = {
  symptoms: ["E_91", "E_201", "E_97"],
  text: "34-year-old female with fever, cough, and sore throat for a few days.",
};

const MISMATCH = {
  symptoms: ["E_66", "E_148"],
  text: "34-year-old female with high fever, bad cough, and sore throat for three days.",
};

export function TriagePage({ meta }: Props) {
  const defaultSymptoms = useMemo(
    () => ALIGNED.symptoms.filter((id) => meta.symptom_catalog.some((s) => s.id === id)),
    [meta],
  );
  const [age, setAge] = useState(34);
  const [sex, setSex] = useState<"M" | "F">("F");
  const [symptoms, setSymptoms] = useState<string[]>(defaultSymptoms);
  const [text, setText] = useState(ALIGNED.text);
  const [result, setResult] = useState<TriageResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const applyPreset = (preset: typeof ALIGNED) => {
    setSymptoms(preset.symptoms.filter((id) => meta.symptom_catalog.some((s) => s.id === id)));
    setText(preset.text);
    setResult(null);
    setError(null);
  };

  const run = () => {
    setLoading(true);
    setError(null);
    api
      .triage({ age, sex, symptoms, text })
      .then(setResult)
      .catch((err: Error) => {
        setResult(null);
        setError(err.message);
      })
      .finally(() => setLoading(false));
  };

  return (
    <section>
      <h2 className="page-title">Triage</h2>
      <p className="page-desc">
        Form and story usually agree on DDXPlus. Hit &ldquo;Mismatch example&rdquo; to see structured
        vs text pull apart — that&apos;s the case I built this for.
      </p>

      <div className="grid-triage">
        <div className="panel">
          <p className="section-label">Intake</p>
          <div className="field">
            <label htmlFor="age">Age</label>
            <input id="age" type="number" min={1} max={110} value={age} onChange={(e) => setAge(Number(e.target.value))} />
          </div>
          <div className="field">
            <label htmlFor="sex">Sex</label>
            <select id="sex" value={sex} onChange={(e) => setSex(e.target.value as "M" | "F")}>
              <option value="F">Female</option>
              <option value="M">Male</option>
            </select>
          </div>
          <div className="field">
            <label>Symptoms</label>
            <div className="symptom-grid">
              {meta.symptom_catalog.map((s) => (
                <label key={s.id} className="symptom-chip">
                  <input
                    type="checkbox"
                    checked={symptoms.includes(s.id)}
                    onChange={(e) => {
                      setSymptoms((prev) =>
                        e.target.checked ? [...prev, s.id] : prev.filter((id) => id !== s.id),
                      );
                    }}
                  />
                  <span>{s.label}</span>
                </label>
              ))}
            </div>
          </div>
          <div className="field">
            <label htmlFor="narrative">Patient description</label>
            <textarea id="narrative" rows={6} value={text} onChange={(e) => setText(e.target.value)} />
          </div>
          <div className="btn-row">
            <button type="button" className="btn-primary" onClick={run} disabled={loading}>
              {loading ? "Running…" : "Run"}
            </button>
            <button type="button" className="btn-secondary" onClick={() => applyPreset(MISMATCH)}>
              Mismatch example
            </button>
            <button type="button" className="btn-secondary" onClick={() => applyPreset(ALIGNED)}>
              Reset
            </button>
          </div>
        </div>

        <div>
          {error && <div className="error">{error}</div>}
          {result && (
            <>
              {result.safety.abstain_recommended && (
                <div className="warn-box">
                  <strong>Low confidence</strong>
                  <p>{result.safety.rationale}</p>
                </div>
              )}
              <div className="info-box">
                {result.modality_analysis.interpretation}
                {" "}
                (disagreement {(result.modality_analysis.disagreement_score * 100).toFixed(0)}%)
              </div>
              {(["multimodal", "structured", "text_only"] as const).map((key) => {
                const branch = result[key];
                return (
                  <div key={key} className="result-card">
                    <h3>{branch.model}</h3>
                    {branch.abstain && <p className="abstain">{branch.abstain_message}</p>}
                    <ol className="dx-list">
                      {branch.differential.map((dx, i) => (
                        <li key={dx.pathology}>
                          <span className="rank">{i + 1}.</span>
                          <span className="name">{dx.pathology}</span>
                          <span className="prob">{(dx.probability * 100).toFixed(1)}%</span>
                        </li>
                      ))}
                    </ol>
                    {branch.modality_weights && (
                      <p className="muted">
                        text weight {branch.modality_weights.text.toFixed(2)}, structured{" "}
                        {branch.modality_weights.structured.toFixed(2)}
                      </p>
                    )}
                  </div>
                );
              })}
            </>
          )}
        </div>
      </div>
    </section>
  );
}