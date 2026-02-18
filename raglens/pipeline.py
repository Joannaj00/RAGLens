"""
Configurable RAG pipeline.

Wraps document loading, chunking, embedding, and retrieval behind a single
RAGPipeline class.  An optional LLM can be attached for answer generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from raglens.chunking import chunk_docs_fixed, chunk_docs_recursive
from raglens.ingest import load_jsonl


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    """All knobs that define one RAG configuration."""
    chunking_strategy: str = "recursive"   # "fixed" | "recursive"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    embedding_model: str = "intfloat/e5-base-v2"
    top_k: int = 5
    config_name: str = ""

    def __post_init__(self):
        if not self.config_name:
            self.config_name = (
                f"{self.chunking_strategy}-{self.chunk_size}-ol{self.chunk_overlap}"
            )


# ---------------------------------------------------------------------------
# Output types
# ---------------------------------------------------------------------------

@dataclass
class RetrievedChunk:
    doc_id: str
    chunk_id: Optional[int]
    title: str
    l2_distance: float
    similarity: float   # 1 / (1 + l2_distance)
    content: str


@dataclass
class RAGResult:
    question: str
    retrieved_chunks: List[RetrievedChunk]
    generated_answer: str
    context_used: str


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

_GENERATION_PROMPT = """\
Answer the question based solely on the provided context.
If the context does not contain enough information, say so.

Context:
{context}

Question: {question}

Answer:"""


class RAGPipeline:
    """
    Build a retrieval pipeline from a JSONL document file and a RAGConfig.

    Usage:
        config = RAGConfig(chunking_strategy="recursive", chunk_size=1000)
        pipeline = RAGPipeline(config).build()
        result = pipeline.run("What is attention mechanism?")
    """

    def __init__(
        self,
        config: RAGConfig,
        data_path: str = "data/raw/docs.jsonl",
    ):
        self.config = config
        self.data_path = data_path
        self.vectorstore: Optional[FAISS] = None
        self._llm = None

    # ------------------------------------------------------------------
    # Builder
    # ------------------------------------------------------------------

    def build(self) -> "RAGPipeline":
        """Load docs, chunk, embed, and build the FAISS index."""
        docs = load_jsonl(self.data_path)

        if self.config.chunking_strategy == "fixed":
            chunks = chunk_docs_fixed(
                docs, self.config.chunk_size, self.config.chunk_overlap
            )
        else:
            chunks = chunk_docs_recursive(
                docs, self.config.chunk_size, self.config.chunk_overlap
            )

        embeddings = HuggingFaceEmbeddings(model_name=self.config.embedding_model)
        self.vectorstore = FAISS.from_documents(chunks, embeddings)
        return self

    def set_llm(self, llm) -> "RAGPipeline":
        """Attach a LangChain chat model for answer generation."""
        self._llm = llm
        return self

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def retrieve(self, question: str, k: Optional[int] = None) -> List[RetrievedChunk]:
        """Return top-k chunks most similar to the question."""
        if self.vectorstore is None:
            raise RuntimeError("Pipeline not built yet. Call .build() first.")
        k = k or self.config.top_k
        raw = self.vectorstore.similarity_search_with_score(question, k=k)
        chunks = []
        for doc, dist in raw:
            chunks.append(RetrievedChunk(
                doc_id=doc.metadata.get("id", ""),
                chunk_id=doc.metadata.get("chunk_id"),
                title=doc.metadata.get("title", ""),
                l2_distance=float(dist),
                similarity=1.0 / (1.0 + float(dist)),
                content=doc.page_content,
            ))
        return chunks

    def generate(self, question: str, context: str) -> str:
        """Generate an answer given a question and retrieved context."""
        if self._llm is None:
            return "[No LLM attached — retrieval-only mode]"
        prompt = _GENERATION_PROMPT.format(context=context, question=question)
        response = self._llm.invoke(prompt)
        return response.content

    def run(self, question: str) -> RAGResult:
        """Full RAG pass: retrieve then generate."""
        chunks = self.retrieve(question)
        context = "\n\n---\n\n".join(c.content for c in chunks)
        answer = self.generate(question, context)
        return RAGResult(
            question=question,
            retrieved_chunks=chunks,
            generated_answer=answer,
            context_used=context,
        )
