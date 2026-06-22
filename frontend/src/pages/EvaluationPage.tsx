import { useEffect, useMemo, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api } from "../api/client";
import type { MetricsPayload } from "../types";

export function EvaluationPage() {
  const [metrics, setMetrics] = useState<MetricsPayload | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.metrics().then(setMetrics).catch((err: Error) => setError(err.message));
  }, []);

  const comparison = useMemo(() => {
    if (!metrics) return [];
    return [
      { model: "Structured", top3: metrics.structured.top3_accuracy, macro: metrics.structured.macro_f1 },
      { model: "Text", top3: metrics.text_only.top3_accuracy, macro: metrics.text_only.macro_f1 },
      { model: "Fusion", top3: metrics.multimodal.top3_accuracy, macro: metrics.multimodal.macro_f1 },
    ];
  }, [metrics]);

  if (error) return <div className="error">{error}</div>;
  if (!metrics) return <p className="loading">Loading metrics…</p>;

  return (
    <section>
      <h2 className="page-title">Results</h2>
      <p className="page-desc">
        Numbers from the held-out test split. Threshold for abstaining was tuned on validate.
      </p>

      <div className="grid-2">
        <div className="panel">
          <p className="section-label">Data</p>
          <ul className="plain-list">
            <li>{metrics.dataset.name}</li>
            <li>
              train {metrics.dataset.train_samples}, test {metrics.dataset.test_samples}
              {metrics.dataset.val_samples != null ? `, val ${metrics.dataset.val_samples}` : ""}
            </li>
            <li>{metrics.dataset.n_classes} classes</li>
          </ul>
          {metrics.eval_notes && (
            <>
              <p className="section-label">Notes</p>
              <ul className="plain-list">
                {Object.entries(metrics.eval_notes).map(([k, v]) => (
                  <li key={k}>
                    {k.replace(/_/g, " ")}: {v}
                  </li>
                ))}
              </ul>
            </>
          )}
        </div>
        <div className="panel">
          <p className="section-label">Noisy text ablation</p>
          {metrics.ablation_noisy_text ? (
            <ul className="plain-list">
              <li>modality disagreement: {(metrics.ablation_noisy_text.mean_modality_disagreement * 100).toFixed(1)}%</li>
              <li>text top-3 with noise: {(metrics.ablation_noisy_text.text_top3_under_noise * 100).toFixed(1)}%</li>
              <li>structured top-3: {(metrics.ablation_noisy_text.structured_top3_stable * 100).toFixed(1)}%</li>
            </ul>
          ) : (
            <p className="muted">Run main.py to populate this.</p>
          )}
        </div>
      </div>

      <div className="chart-panel">
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={comparison}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="model" />
            <YAxis domain={[0, 1]} />
            <Tooltip formatter={(v: number) => `${(v * 100).toFixed(1)}%`} />
            <Legend />
            <Bar dataKey="top3" name="Top-3 acc" fill="#1a4480" />
            <Bar dataKey="macro" name="Macro F1" fill="#2e6ba8" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}