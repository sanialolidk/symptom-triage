import type { Meta, MetricsPayload, TriageResponse } from "../types";

const BASE = import.meta.env.VITE_API_URL ?? "";

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`);
  if (!res.ok) throw new Error(await res.text() || res.statusText);
  return res.json() as Promise<T>;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text() || res.statusText);
  return res.json() as Promise<T>;
}

export const api = {
  meta: () => get<Meta>("/api/meta"),
  metrics: () => get<MetricsPayload>("/api/metrics"),
  triage: (payload: { age: number; sex: "M" | "F"; symptoms: string[]; text: string }) =>
    post<TriageResponse>("/api/triage", payload),
};