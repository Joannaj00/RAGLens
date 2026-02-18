import type { ConfigSummary, ConfigReport, QueryResponse } from "./types"

const API = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`, { cache: "no-store" })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail || `Request failed: ${res.status}`)
  }
  return res.json()
}

export const getSummary = () => get<ConfigSummary[]>("/api/summary")

export const getConfigs = () => get<string[]>("/api/configs")

export const getConfigResults = (config: string) =>
  get<ConfigReport>(`/api/results/${config}`)

export async function runQuery(opts: {
  question: string
  chunking: string
  chunk_size: number
  chunk_overlap: number
  top_k: number
}): Promise<QueryResponse> {
  const res = await fetch(`${API}/api/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(opts),
  })
  if (!res.ok) {
    const detail = await res.text()
    throw new Error(detail || `Query failed: ${res.status}`)
  }
  return res.json()
}
