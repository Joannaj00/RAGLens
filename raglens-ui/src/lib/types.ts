export interface ConfigSummary {
  config: string
  mrr: number
  "recall@1": number
  "recall@3": number
  "recall@5": number
  "recall@10": number
  build_s: number
  eval_s: number
}

export interface RetrievedChunk {
  rank?: number
  doc_id: string
  chunk_id: number | null
  title: string
  l2_distance: number
  similarity: number
  content: string
  is_relevant?: boolean // present in eval results, absent in live query
}

export interface QuestionResult {
  question: string
  relevant_doc_ids: string[]
  reciprocal_rank: number
  recall_at_k: Record<string, number>
  precision_at_k: Record<string, number>
  retrieved: RetrievedChunk[]
}

export interface ConfigReport {
  config_name: string
  macro_mrr: number
  macro_recall_at_k: Record<string, number>
  macro_precision_at_k: Record<string, number>
  per_question: QuestionResult[]
}

export interface QueryResponse {
  question: string
  config: {
    chunking: string
    chunk_size: number
    chunk_overlap: number
    top_k: number
  }
  chunks: RetrievedChunk[]
}
