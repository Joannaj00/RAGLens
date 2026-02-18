"use client"
import { useState } from "react"
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts"
import { runQuery } from "@/lib/api"
import type { QueryResponse } from "@/lib/types"
import ChunkCard from "@/components/ChunkCard"

const SAMPLE_QUESTIONS = [
  "What is gamma-ray emission region?",
  "How do neural networks learn representations?",
  "What methods are used for object detection?",
  "Explain the attention mechanism in transformers.",
]

export default function InspectorPage() {
  const [question, setQuestion] = useState("")
  const [chunking, setChunking] = useState("recursive")
  const [chunkSize, setChunkSize] = useState(1000)
  const [chunkOverlap, setChunkOverlap] = useState(150)
  const [topK, setTopK] = useState(5)
  const [result, setResult] = useState<QueryResponse | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [buildNote, setBuildNote] = useState("")

  async function handleQuery() {
    if (!question.trim()) return
    setLoading(true)
    setError("")
    setBuildNote("Building index if not cached — this may take up to 60 seconds on first run…")
    try {
      const res = await runQuery({ question, chunking, chunk_size: chunkSize, chunk_overlap: chunkOverlap, top_k: topK })
      setResult(res)
    } catch (e: any) {
      setError(e.message)
    } finally {
      setLoading(false)
      setBuildNote("")
    }
  }

  const simBarData = result?.chunks.map((c) => ({
    name: `#${c.rank}`,
    similarity: +(c.similarity * 100).toFixed(1),
    title: c.title.slice(0, 40),
  })) ?? []

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">Retrieval Inspector</h1>
        <p className="text-slate-500 text-sm mt-1">
          Run a live query against any pipeline configuration and inspect the retrieved chunks.
        </p>
      </div>

      <div className="grid grid-cols-4 gap-6">
        {/* Sidebar controls */}
        <div className="col-span-1 space-y-4">
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm space-y-4">
            <h2 className="text-sm font-semibold text-slate-700">Pipeline Config</h2>

            <div>
              <label className="text-xs font-medium text-slate-500 block mb-1">Chunking Strategy</label>
              <select
                className="w-full border border-slate-200 rounded-lg px-3 py-2 text-sm bg-white"
                value={chunking}
                onChange={(e) => setChunking(e.target.value)}
              >
                <option value="recursive">Recursive</option>
                <option value="fixed">Fixed</option>
              </select>
            </div>

            <div>
              <label className="text-xs font-medium text-slate-500 block mb-1">
                Chunk Size: <strong>{chunkSize}</strong>
              </label>
              <input type="range" min={200} max={2000} step={100} value={chunkSize}
                onChange={(e) => setChunkSize(+e.target.value)}
                className="w-full accent-blue-500" />
            </div>

            <div>
              <label className="text-xs font-medium text-slate-500 block mb-1">
                Chunk Overlap: <strong>{chunkOverlap}</strong>
              </label>
              <input type="range" min={0} max={400} step={25} value={chunkOverlap}
                onChange={(e) => setChunkOverlap(+e.target.value)}
                className="w-full accent-blue-500" />
            </div>

            <div>
              <label className="text-xs font-medium text-slate-500 block mb-1">
                Top-k: <strong>{topK}</strong>
              </label>
              <input type="range" min={1} max={15} step={1} value={topK}
                onChange={(e) => setTopK(+e.target.value)}
                className="w-full accent-blue-500" />
            </div>
          </div>

          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
            <h2 className="text-xs font-semibold text-slate-500 uppercase tracking-wide mb-2">Sample Questions</h2>
            <ul className="space-y-1">
              {SAMPLE_QUESTIONS.map((q) => (
                <li key={q}>
                  <button
                    onClick={() => setQuestion(q)}
                    className="text-left text-xs text-blue-600 hover:underline leading-snug"
                  >
                    {q}
                  </button>
                </li>
              ))}
            </ul>
          </div>
        </div>

        {/* Main query area */}
        <div className="col-span-3 space-y-5">
          <div className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm">
            <div className="flex gap-3">
              <input
                type="text"
                placeholder="Ask any question about the corpus…"
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleQuery()}
                className="flex-1 border border-slate-200 rounded-lg px-4 py-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
              />
              <button
                onClick={handleQuery}
                disabled={loading || !question.trim()}
                className="px-5 py-2.5 bg-blue-600 text-white rounded-lg text-sm font-semibold hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {loading ? "Retrieving…" : "Retrieve"}
              </button>
            </div>
            {buildNote && (
              <p className="text-xs text-amber-600 mt-2">{buildNote}</p>
            )}
            {error && (
              <p className="text-xs text-red-600 mt-2">{error}</p>
            )}
          </div>

          {result && (
            <>
              {/* Similarity bar chart */}
              <div className="bg-white rounded-xl border border-slate-200 p-5 shadow-sm">
                <h2 className="text-sm font-semibold text-slate-700 mb-4">Similarity Scores</h2>
                <ResponsiveContainer width="100%" height={200}>
                  <BarChart data={simBarData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
                    <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                    <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} />
                    <Tooltip
                      formatter={(v: any) => [`${v}%`, "Similarity"]}
                      labelFormatter={(_, payload) => payload?.[0]?.payload?.title || ""}
                    />
                    <Bar dataKey="similarity" radius={[4, 4, 0, 0]}>
                      {simBarData.map((_, i) => (
                        <Cell key={i} fill={`hsl(${220 - i * 18}, 80%, ${55 + i * 4}%)`} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Chunk cards */}
              <div className="space-y-3">
                {result.chunks.map((chunk, i) => (
                  <ChunkCard key={i} chunk={chunk} rank={chunk.rank ?? i + 1} />
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  )
}
