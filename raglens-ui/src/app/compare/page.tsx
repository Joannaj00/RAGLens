"use client"
import { useEffect, useState } from "react"
import {
  LineChart, Line, ScatterChart, Scatter,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ReferenceLine,
} from "recharts"
import { getConfigs, getConfigResults } from "@/lib/api"
import type { ConfigReport } from "@/lib/types"

const KS = ["1", "3", "5", "10"]

function Delta({ label, a, b }: { label: string; a: number; b: number }) {
  const diff = b - a
  const pos = diff > 0
  const neutral = Math.abs(diff) < 0.001
  return (
    <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm text-center">
      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-1">{label}</p>
      <p className="text-2xl font-bold text-slate-800">{(b * 100).toFixed(1)}%</p>
      {!neutral && (
        <p className={`text-xs font-semibold mt-1 ${pos ? "text-emerald-600" : "text-red-500"}`}>
          {pos ? "▲" : "▼"} {Math.abs(diff * 100).toFixed(1)}pp vs A
        </p>
      )}
      {neutral && <p className="text-xs text-slate-400 mt-1">tied</p>}
      <p className="text-xs text-slate-400 mt-0.5">A: {(a * 100).toFixed(1)}%</p>
    </div>
  )
}

export default function ComparePage() {
  const [configs, setConfigs] = useState<string[]>([])
  const [cfgA, setCfgA] = useState("")
  const [cfgB, setCfgB] = useState("")
  const [repA, setRepA] = useState<ConfigReport | null>(null)
  const [repB, setRepB] = useState<ConfigReport | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    getConfigs().then((cs) => {
      setConfigs(cs)
      if (cs.length >= 2) { setCfgA(cs[0]); setCfgB(cs[1]) }
    })
  }, [])

  useEffect(() => {
    if (!cfgA || !cfgB) return
    setLoading(true)
    Promise.all([getConfigResults(cfgA), getConfigResults(cfgB)])
      .then(([a, b]) => { setRepA(a); setRepB(b) })
      .finally(() => setLoading(false))
  }, [cfgA, cfgB])

  const select = (cls: string) => (
    `border border-slate-200 rounded-lg px-3 py-2 text-sm bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400 ${cls}`
  )

  // Recall@k line data
  const recallData = KS.map((k) => ({
    k: `@${k}`,
    A: repA ? repA.macro_recall_at_k[k] : 0,
    B: repB ? repB.macro_recall_at_k[k] : 0,
  }))

  // Scatter data: per-question RR of A vs B
  const scatterData: { x: number; y: number; winner: string }[] = []
  if (repA && repB) {
    const mapA = Object.fromEntries(repA.per_question.map((q) => [q.question, q.reciprocal_rank]))
    repB.per_question.forEach((qr) => {
      const rrA = mapA[qr.question] ?? 0
      const rrB = qr.reciprocal_rank
      scatterData.push({
        x: rrA,
        y: rrB,
        winner: rrB > rrA ? "B wins" : rrA > rrB ? "A wins" : "tied",
      })
    })
  }

  const winnerColors: Record<string, string> = { "B wins": "#10b981", "A wins": "#3b82f6", "tied": "#94a3b8" }

  // Group scatter by winner for colored series
  const grouped = ["B wins", "A wins", "tied"].map((w) => ({
    name: w,
    data: scatterData.filter((d) => d.winner === w),
    color: winnerColors[w],
  }))

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Strategy Comparison</h1>
        <p className="text-slate-500 text-sm mt-1">Compare two configurations head-to-head.</p>
      </div>

      {/* Selectors */}
      <div className="flex items-center gap-6 bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-blue-600 w-6">A</span>
          <select className={select("")} value={cfgA} onChange={(e) => setCfgA(e.target.value)}>
            {configs.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
        <span className="text-slate-400 font-bold">vs</span>
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-emerald-600 w-6">B</span>
          <select className={select("")} value={cfgB} onChange={(e) => setCfgB(e.target.value)}>
            {configs.map((c) => <option key={c} value={c}>{c}</option>)}
          </select>
        </div>
      </div>

      {loading && <div className="text-slate-400 text-center py-10">Loading…</div>}

      {repA && repB && (
        <>
          {/* Metric deltas */}
          <div className="grid grid-cols-5 gap-4">
            <Delta label="MRR" a={repA.macro_mrr} b={repB.macro_mrr} />
            {KS.map((k) => (
              <Delta
                key={k}
                label={`Recall@${k}`}
                a={repA.macro_recall_at_k[k] ?? 0}
                b={repB.macro_recall_at_k[k] ?? 0}
              />
            ))}
          </div>

          {/* Charts */}
          <div className="grid grid-cols-2 gap-6">
            {/* Recall@k overlay */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <h2 className="text-sm font-semibold text-slate-700 mb-4">Recall@k: A vs B</h2>
              <ResponsiveContainer width="100%" height={260}>
                <LineChart data={recallData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis dataKey="k" />
                  <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                  <Tooltip formatter={(v: any) => `${(+v * 100).toFixed(1)}%`} />
                  <Legend />
                  <Line type="monotone" dataKey="A" name={cfgA} stroke="#3b82f6" strokeWidth={2} dot={{ r: 4 }} />
                  <Line type="monotone" dataKey="B" name={cfgB} stroke="#10b981" strokeWidth={2} dot={{ r: 4 }} />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Per-question scatter */}
            <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
              <h2 className="text-sm font-semibold text-slate-700 mb-1">Per-question Reciprocal Rank</h2>
              <p className="text-xs text-slate-400 mb-4">Above diagonal = B wins · Below = A wins</p>
              <ResponsiveContainer width="100%" height={260}>
                <ScatterChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                  <XAxis type="number" dataKey="x" name={cfgA} domain={[-0.05, 1.1]}
                    label={{ value: "A", position: "insideBottom", offset: -2 }} />
                  <YAxis type="number" dataKey="y" name={cfgB} domain={[-0.05, 1.1]}
                    label={{ value: "B", angle: -90, position: "insideLeft" }} />
                  <Tooltip cursor={{ strokeDasharray: "3 3" }} />
                  <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]}
                    stroke="#94a3b8" strokeDasharray="5 5" />
                  {grouped.map(({ name, data, color }) => (
                    <Scatter key={name} name={name} data={data} fill={color} opacity={0.7} />
                  ))}
                  <Legend />
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
