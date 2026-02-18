"use client"
import { useEffect, useState } from "react"
import { getConfigs, getConfigResults } from "@/lib/api"
import type { ConfigReport, QuestionResult } from "@/lib/types"
import ChunkCard from "@/components/ChunkCard"

function MetricPill({ label, value }: { label: string; value: number }) {
  return (
    <span className="inline-flex items-center gap-1 bg-slate-100 px-2.5 py-1 rounded-full text-xs font-semibold text-slate-600">
      <span className="text-slate-400">{label}</span>
      <span className="text-blue-600">{(value * 100).toFixed(0)}%</span>
    </span>
  )
}

export default function DrillDownPage() {
  const [configs, setConfigs] = useState<string[]>([])
  const [selectedConfig, setSelectedConfig] = useState("")
  const [report, setReport] = useState<ConfigReport | null>(null)
  const [selectedQ, setSelectedQ] = useState<QuestionResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")

  useEffect(() => {
    getConfigs().then((cs) => {
      setConfigs(cs)
      if (cs.length) setSelectedConfig(cs[0])
    })
  }, [])

  useEffect(() => {
    if (!selectedConfig) return
    setLoading(true)
    setSelectedQ(null)
    getConfigResults(selectedConfig)
      .then((r) => { setReport(r); setLoading(false) })
      .catch((e) => { setError(e.message); setLoading(false) })
  }, [selectedConfig])

  const rrColor = (rr: number) =>
    rr === 1 ? "text-emerald-600" : rr > 0.3 ? "text-amber-500" : "text-red-500"

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Drill-down</h1>
        <p className="text-slate-500 text-sm mt-1">
          Select a configuration and question to inspect exactly which chunks were retrieved.
        </p>
      </div>

      {/* Config selector */}
      <div className="flex items-center gap-3">
        <label className="text-sm font-medium text-slate-600">Configuration</label>
        <select
          className="border border-slate-200 rounded-lg px-3 py-2 text-sm bg-white shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
          value={selectedConfig}
          onChange={(e) => setSelectedConfig(e.target.value)}
        >
          {configs.map((c) => <option key={c} value={c}>{c}</option>)}
        </select>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">{error}</div>
      )}

      {loading && (
        <div className="flex items-center justify-center h-40 text-slate-400">Loading results…</div>
      )}

      {report && (
        <div className="grid grid-cols-5 gap-6">
          {/* Question list */}
          <div className="col-span-2 bg-white rounded-xl border border-slate-200 shadow-sm overflow-hidden">
            <div className="px-4 py-3 border-b border-slate-100 bg-slate-50">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wide">
                {report.per_question.length} Questions
              </p>
            </div>
            <ul className="divide-y divide-slate-100 max-h-[600px] overflow-y-auto">
              {report.per_question.map((qr, i) => (
                <li
                  key={i}
                  onClick={() => setSelectedQ(qr)}
                  className={`px-4 py-3 cursor-pointer hover:bg-blue-50 transition-colors ${
                    selectedQ?.question === qr.question ? "bg-blue-50 border-l-4 border-blue-400" : ""
                  }`}
                >
                  <p className="text-sm text-slate-700 line-clamp-2 mb-1">{qr.question}</p>
                  <div className="flex items-center gap-2">
                    <span className={`text-xs font-bold ${rrColor(qr.reciprocal_rank)}`}>
                      RR {qr.reciprocal_rank.toFixed(2)}
                    </span>
                    <MetricPill label="R@5" value={qr.recall_at_k["5"] ?? 0} />
                  </div>
                </li>
              ))}
            </ul>
          </div>

          {/* Chunk detail */}
          <div className="col-span-3">
            {!selectedQ ? (
              <div className="flex items-center justify-center h-64 text-slate-400 bg-white rounded-xl border border-slate-200">
                ← Select a question
              </div>
            ) : (
              <div className="space-y-4">
                {/* Question header */}
                <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
                  <p className="text-sm font-semibold text-slate-700 mb-3">{selectedQ.question}</p>
                  <div className="flex flex-wrap gap-2">
                    <span className="text-xs text-slate-400">
                      Relevant doc: <code className="bg-slate-100 px-1 rounded">{selectedQ.relevant_doc_ids.join(", ")}</code>
                    </span>
                  </div>
                  <div className="flex flex-wrap gap-2 mt-3">
                    <span className={`font-semibold text-sm ${rrColor(selectedQ.reciprocal_rank)}`}>
                      RR {selectedQ.reciprocal_rank.toFixed(3)}
                    </span>
                    {[1, 3, 5, 10].map((k) => (
                      <MetricPill key={k} label={`R@${k}`} value={selectedQ.recall_at_k[String(k)] ?? 0} />
                    ))}
                  </div>
                </div>

                {/* Chunks */}
                <div className="space-y-3 max-h-[520px] overflow-y-auto pr-1">
                  {selectedQ.retrieved.map((chunk, i) => (
                    <ChunkCard key={i} chunk={chunk} rank={i + 1} showRelevance />
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
