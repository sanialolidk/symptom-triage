import { useMemo, useState } from "react";
import { api } from "../api/client";
import type { BranchResult, Meta, TriageResponse } from "../types";

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

function DifferentialList({ branch }: { branch: BranchResult }) {
  const topProb = branch.differential[0]?.probability ?? 0;

  return (
    <>
      {branch.abstain && <p className="abstain">{branch.abstain_message}</p>}
      <ol className="dx-list">
        {branch.differential.map((dx, i) => (
          <li key={dx.pathology} className="dx-item">
            <span className="rank">{i + 1}</span>
            <span className="name">{dx.pathology}</span>
            <span className="prob">{(dx.probability * 100).toFixed(1)}%</span>
            <div className="prob-bar" aria-hidden>
              <span style={{ width: `${(dx.probability / topProb) * 100}%` }} />
            </div>
          </li>
        ))}
      </ol>
      {branch.modality_weights && (
        <div className="modality-weights">
          <div>Modality weights</div>
          <div className="weight-track">
            <span>Text</span>
            <div className="bar">
              <span style={{ width: `${branch.modality_weights.text * 100}%` }} />
            </div>
            <span>{branch.modality_weights.text.toFixed(2)}</span>
          </div>
          <div className="weight-track">
            <span>Struct</span>
            <div className="bar">
              <span style={{ width: `${branch.modality_weights.structured * 100}%` }} />
            </div>
            <span>{branch.modality_weights.structured.toFixed(2)}</span>
          </div>
        </div>
      )}
    </>
  );
}

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
      <div className="page-intro">
        <h2 className="page-title">Run a case</h2>
        <p className="page-desc">
          Form and story usually agree on DDXPlus. Use the mismatch preset to watch structured vs
          text models diverge — that&apos;s the scenario this was built for.
        </p>
      </div>

      <div className="grid-triage">
        <div className="panel">
          <p className="section-label">Patient intake</p>

          <div className="field-row">
            <div className="field">
              <label htmlFor="age">Age</label>
              <input
                id="age"
                type="number"
                min={1}
                max={110}
                value={age}
                onChange={(e) => setAge(Number(e.target.value))}
              />
            </div>
            <div className="field">
              <label htmlFor="sex">Sex</label>
              <select id="sex" value={sex} onChange={(e) => setSex(e.target.value as "M" | "F")}>
                <option value="F">Female</option>
                <option value="M">Male</option>
              </select>
            </div>
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
            <textarea
              id="narrative"
              rows={6}
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="What they'd say at intake — can disagree with the checklist above."
            />
          </div>

          <div className="btn-row">
            <button type="button" className="btn-primary" onClick={run} disabled={loading}>
              {loading ? "Running…" : "Run triage"}
            </button>
            <button type="button" className="btn-secondary" onClick={() => applyPreset(MISMATCH)}>
              Mismatch example
            </button>
            <button type="button" className="btn-ghost" onClick={() => applyPreset(ALIGNED)}>
              Reset
            </button>
          </div>
        </div>

        <div className="panel panel-sticky">
          <p className="section-label">Differential</p>

          {error && <div className="error">{error}</div>}

          {!result && !error && !loading && (
            <div className="empty-state">
              <strong>No results yet</strong>
              <span>Fill in the intake form and run triage, or load the mismatch example.</span>
            </div>
          )}

          {loading && <p className="loading">Running structured, text, and fusion models…</p>}

          {result && (
            <div className="results-stack">
              {result.safety.abstain_recommended && (
                <div className="status-banner warn">
                  <div>
                    <strong>Low confidence</strong>
                    {result.safety.rationale}
                  </div>
                </div>
              )}

              <div
                className={`status-banner ${
                  result.modality_analysis.modalities_conflict ? "conflict" : "ok"
                }`}
              >
                <div>
                  <strong>
                    {result.modality_analysis.modalities_conflict
                      ? "Modalities conflict"
                      : "Modalities agree"}
                  </strong>
                  {result.modality_analysis.interpretation} Disagreement{" "}
                  {(result.modality_analysis.disagreement_score * 100).toFixed(0)}%.
                </div>
              </div>

              {(["multimodal", "structured", "text_only"] as const).map((key) => {
                const branch = result[key];
                return (
                  <div key={key} className={`result-card ${key === "multimodal" ? "featured" : ""}`}>
                    <div className="result-card-header">
                      <h3>{branch.model}</h3>
                      <span className={`confidence-pill ${branch.abstain ? "low" : ""}`}>
                        {(branch.confidence * 100).toFixed(0)}% conf
                      </span>
                    </div>
                    <DifferentialList branch={branch} />
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </section>
  );
}