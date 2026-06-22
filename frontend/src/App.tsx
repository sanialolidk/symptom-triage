import { useEffect, useState } from "react";
import { api } from "./api/client";
import { Footer } from "./components/Footer";
import { Header } from "./components/Header";
import { EvaluationPage } from "./pages/EvaluationPage";
import { TriagePage } from "./pages/TriagePage";
import type { Meta, Page } from "./types";

const TABS: { id: Page; label: string }[] = [
  { id: "triage", label: "Triage" },
  { id: "evaluation", label: "Results" },
];

export default function App() {
  const [page, setPage] = useState<Page>("triage");
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.meta().then(setMeta).catch((err: Error) => setError(err.message));
  }, []);

  return (
    <div className="app-shell">
      <Header />
      <nav className="nav-tabs">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            className={`nav-tab ${page === tab.id ? "active" : ""}`}
            onClick={() => setPage(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {error && (
        <div className="error">
          API not up — run <code>uvicorn api.main:app --port 8001</code>
          <br />
          {error}
        </div>
      )}

      {!meta && !error && <p className="loading">Loading…</p>}
      {meta && page === "triage" && <TriagePage meta={meta} />}
      {page === "evaluation" && <EvaluationPage />}
      <Footer />
    </div>
  );
}