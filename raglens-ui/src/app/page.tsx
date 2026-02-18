"use client"
import { useEffect, useState } from "react"
import {
  LineChart, Line, BarChart, Bar,
  XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from "recharts"
import { getSummary } from "@/lib/api"
import type { ConfigSummary } from "@/lib/types"

const COLORS = ["#3b82f6", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"]
const KS = [1, 3, 5, 10]

function StatCard({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-200">
      <p className="text-xs font-semibold text-slate-400 uppercase tracking-wide mb-1">{label}</p>
      <p className="text-3xl font-bold text-slate-800">{value}</p>
      {sub && <p className="text-xs text-slate-400 mt-1">{sub}</p>}
    </div>
  )
}

export default function OverviewPage() {
  const [data, setData] = useState<ConfigSummary[]>([])
  const [error, setError] = useState("")

  useEffect(() => {
    getSummary()
      .then(setData)
      .catch((e) => setError(e.message))
  }, [])

  if (error) return (
    <div className="bg-red-50 border border-red-200 rounded-xl p-6 text-red-700">
      <p className="font-semibold mb-1">Could not load results</p>
      <p className="text-sm">{error}</p>
      <p className="text-sm mt-2">Make sure the FastAPI backend is running:<br />
        <code className="bg-red-100 px-2 py-0.5 rounded">uvicorn dashboard.api:app --reload --port 8000</code>
      </p>
    </div>
  )

  if (!data.length) return (
    <div className="flex items-center justify-center h-64 text-slate-400">Loading…</div>
  )

  const bestMRR = Math.max(...data.map((d) => d.mrr))
  const bestR5 = Math.max(...data.map((d) => d["recall@5"]))
  const bestConfig = data.find((d) => d.mrr === bestMRR)?.config ?? "—"

  // Recall@k line chart data: one point per k, one line per config
  const recallChartData = KS.map((k) => {
    const point: Record<string, number | string> = { k }
    data.forEach((d) => { point[d.config] = +d[`recall@${k}` as keyof ConfigSummary] })
    return point
  })

  // MRR bar chart data
  const mrrData = [...data].sort((a, b) => b.mrr - a.mrr)

  return (
    <div className="space-y-8">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Overview</h1>
        <p className="text-slate-500 text-sm mt-1">
          Aggregated retrieval metrics across all evaluated configurations.
        </p>
      </div>

      {/* Stat cards */}
      <div className="grid grid-cols-3 gap-4">
        <StatCard label="Best MRR" value={(bestMRR * 100).toFixed(1) + "%"} sub={bestConfig} />
        <StatCard label="Best Recall@5" value={(bestR5 * 100).toFixed(1) + "%"} />
        <StatCard label="Configs Tested" value={String(data.length)} />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-2 gap-6">
        <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-200">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">Recall@k by Configuration</h2>
          <ResponsiveContainer width="100%" height={260}>
            <LineChart data={recallChartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
              <XAxis dataKey="k" label={{ value: "k", position: "insideBottom", offset: -2 }} />
              <YAxis domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <Tooltip formatter={(v: any) => `${(+v * 100).toFixed(1)}%`} />
              <Legend />
              {data.map((d, i) => (
                <Line
                  key={d.config}
                  type="monotone"
                  dataKey={d.config}
                  stroke={COLORS[i % COLORS.length]}
                  strokeWidth={2}
                  dot={{ r: 4 }}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-white rounded-xl p-5 shadow-sm border border-slate-200">
          <h2 className="text-sm font-semibold text-slate-700 mb-4">MRR by Configuration</h2>
          <ResponsiveContainer width="100%" height={260}>
            <BarChart data={mrrData} layout="vertical" margin={{ left: 16 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" horizontal={false} />
              <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <YAxis type="category" dataKey="config" width={160} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(v: any) => `${(+v * 100).toFixed(2)}%`} />
              <Bar dataKey="mrr" fill="#3b82f6" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Metrics table */}
      <div className="bg-white rounded-xl shadow-sm border border-slate-200 overflow-hidden">
        <div className="px-5 py-4 border-b border-slate-100">
          <h2 className="text-sm font-semibold text-slate-700">Full Metrics Table</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 text-slate-500 text-xs uppercase tracking-wide">
              <tr>
                {["Config", "MRR", "R@1", "R@3", "R@5", "R@10", "Build (s)", "Eval (s)"].map((h) => (
                  <th key={h} className="px-4 py-3 text-right first:text-left font-semibold">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {data.map((row, i) => (
                <tr key={row.config} className={i % 2 === 0 ? "bg-white" : "bg-slate-50/50"}>
                  <td className="px-4 py-3 font-mono text-xs font-medium text-slate-700">{row.config}</td>
                  {[row.mrr, row["recall@1"], row["recall@3"], row["recall@5"], row["recall@10"]].map((v, j) => (
                    <td key={j} className="px-4 py-3 text-right font-semibold text-blue-700">
                      {(v * 100).toFixed(1)}%
                    </td>
                  ))}
                  <td className="px-4 py-3 text-right text-slate-500">{row.build_s}s</td>
                  <td className="px-4 py-3 text-right text-slate-500">{row.eval_s}s</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}
