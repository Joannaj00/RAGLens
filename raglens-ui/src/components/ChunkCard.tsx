import type { RetrievedChunk } from "@/lib/types"

interface Props {
  chunk: RetrievedChunk
  rank: number
  showRelevance?: boolean
}

export default function ChunkCard({ chunk, rank, showRelevance = false }: Props) {
  const isRelevant = chunk.is_relevant

  const borderColor = showRelevance
    ? isRelevant
      ? "border-emerald-400"
      : "border-red-300"
    : "border-slate-200"

  const badge = showRelevance ? (
    isRelevant ? (
      <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-emerald-100 text-emerald-700">
        ✓ Relevant
      </span>
    ) : (
      <span className="px-2 py-0.5 rounded-full text-xs font-semibold bg-red-100 text-red-600">
        ✗ Not relevant
      </span>
    )
  ) : null

  return (
    <div className={`bg-white rounded-xl border-2 ${borderColor} p-4 shadow-sm`}>
      <div className="flex items-start justify-between gap-4 mb-2">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-bold text-slate-400 uppercase tracking-wide">
            Rank {rank}
          </span>
          {badge}
        </div>
        <div className="text-right shrink-0">
          <span className="text-sm font-semibold text-blue-600">
            {(chunk.similarity * 100).toFixed(1)}%
          </span>
          <span className="text-xs text-slate-400 ml-1">sim</span>
        </div>
      </div>

      <p className="text-sm font-semibold text-slate-700 mb-1 truncate" title={chunk.title}>
        {chunk.title || "(no title)"}
      </p>
      <p className="text-xs text-slate-400 mb-3 font-mono">{chunk.doc_id}</p>

      <p className="text-sm text-slate-600 leading-relaxed line-clamp-4">{chunk.content}</p>

      <div className="mt-3 pt-3 border-t border-slate-100 flex gap-4 text-xs text-slate-400">
        <span>L2 dist: <strong>{chunk.l2_distance.toFixed(4)}</strong></span>
        <span>Chunk ID: <strong>{chunk.chunk_id ?? "—"}</strong></span>
      </div>
    </div>
  )
}
