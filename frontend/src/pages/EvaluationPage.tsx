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
      <div className="page-intro">
        <h2 className="page-title">Evaluation</h2>
        <p className="page-desc">
          Held-out test split. Abstention threshold was tuned on the validation set.
        </p>
      </div>

      <div className="stat-grid">
        <div className="stat-card">
          <div className="label">Fusion top-3</div>
          <div className="value">{(metrics.multimodal.top3_accuracy * 100).toFixed(1)}%</div>
          <div className="sub">macro F1 {(metrics.multimodal.macro_f1 * 100).toFixed(1)}%</div>
        </div>
        <div className="stat-card">
          <div className="label">Test cases</div>
          <div className="value">{metrics.dataset.test_samples}</div>
          <div className="sub">{metrics.dataset.n_classes} pathology classes</div>
        </div>
        <div className="stat-card">
          <div className="label">Abstain rate</div>
          <div className="value">{(metrics.multimodal.abstain_rate * 100).toFixed(1)}%</div>
          <div className="sub">threshold {metrics.multimodal.abstain_threshold.toFixed(2)}</div>
        </div>
      </div>

      <div className="grid-2">
        <div className="panel">
          <p className="section-label">Dataset</p>
          <ul className="plain-list">
            <li>{metrics.dataset.name}</li>
            <li>
              train {metrics.dataset.train_samples}, test {metrics.dataset.test_samples}
              {metrics.dataset.val_samples != null ? `, val ${metrics.dataset.val_samples}` : ""}
            </li>
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
              <li>
                modality disagreement:{" "}
                {(metrics.ablation_noisy_text.mean_modality_disagreement * 100).toFixed(1)}%
              </li>
              <li>
                text top-3 with noise:{" "}
                {(metrics.ablation_noisy_text.text_top3_under_noise * 100).toFixed(1)}%
              </li>
              <li>
                structured top-3:{" "}
                {(metrics.ablation_noisy_text.structured_top3_stable * 100).toFixed(1)}%
              </li>
            </ul>
          ) : (
            <p className="muted">Run main.py to populate this.</p>
          )}
        </div>
      </div>

      <div className="chart-panel">
        <p className="section-label">Model comparison</p>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={comparison} barGap={6} barCategoryGap="18%">
            <CartesianGrid strokeDasharray="3 3" stroke="#e4ddd3" vertical={false} />
            <XAxis dataKey="model" tick={{ fill: "#6b7a8f", fontSize: 12 }} axisLine={false} tickLine={false} />
            <YAxis
              domain={[0, 1]}
              tick={{ fill: "#6b7a8f", fontSize: 12 }}
              axisLine={false}
              tickLine={false}
              tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip
              formatter={(v: number) => `${(v * 100).toFixed(1)}%`}
              contentStyle={{
                borderRadius: 10,
                border: "1px solid #e4ddd3",
                boxShadow: "0 8px 24px rgba(28, 36, 52, 0.08)",
              }}
            />
            <Legend />
            <Bar dataKey="top3" name="Top-3 acc" fill="#b84a32" radius={[6, 6, 0, 0]} />
            <Bar dataKey="macro" name="Macro F1" fill="#2f5f8a" radius={[6, 6, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}