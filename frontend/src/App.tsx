import { useEffect, useState } from "react";
import { api } from "./api/client";
import { Footer } from "./components/Footer";
import { Header } from "./components/Header";
import { EvaluationPage } from "./pages/EvaluationPage";
import { TriagePage } from "./pages/TriagePage";
import type { Meta, Page } from "./types";

export default function App() {
  const [page, setPage] = useState<Page>("triage");
  const [meta, setMeta] = useState<Meta | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.meta().then(setMeta).catch((err: Error) => setError(err.message));
  }, []);

  return (
    <div className="app-shell">
      <Header page={page} onPageChange={setPage} />

      <main className="main-content">
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
      </main>

      <Footer />
    </div>
  );
}