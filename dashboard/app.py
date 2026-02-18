"""
RAGLens Visualization Dashboard
================================
Streamlit app with four tabs:

  1. Retrieval Inspector  — run a live query and inspect retrieved chunks
  2. Eval Metrics         — aggregate Recall@k / MRR charts from saved results
  3. Question Browser     — browse per-question retrieval results from eval runs
  4. Strategy Comparison  — side-by-side metric comparison across configs

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make sure the project root is on the path when running from any directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_env(path: Path) -> None:
    """Minimal .env loader — no third-party deps required."""
    if not path.exists():
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())

_load_env(ROOT / ".env")

# .env uses OPENAI-KEY; LangChain/OpenAI SDK expects OPENAI_API_KEY
_raw_key = os.environ.get("OPENAI-KEY", "")
if _raw_key:
    os.environ["OPENAI_API_KEY"] = _raw_key

from raglens.pipeline import RAGConfig, RAGPipeline

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="RAGLens",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EVAL_RESULTS_DIR = ROOT / "data" / "eval_results"
DATA_PATH = str(ROOT / "data" / "raw" / "docs.jsonl")
KS = [1, 3, 5, 10]

CHUNK_COLORS = {
    True:  "#d4edda",   # green tint for relevant
    False: "#f8d7da",   # red tint for irrelevant
}
RANK_BADGE = {
    True:  "✅",
    False: "❌",
}


# ---------------------------------------------------------------------------
# Cached helpers
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Building pipeline — this may take a minute…")
def get_pipeline(chunking: str, chunk_size: int, chunk_overlap: int) -> RAGPipeline:
    config = RAGConfig(
        chunking_strategy=chunking,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return RAGPipeline(config, data_path=DATA_PATH).build()


def load_eval_results() -> dict[str, dict]:
    """Load all per-config JSON result files from data/eval_results/."""
    results = {}
    if not EVAL_RESULTS_DIR.exists():
        return results
    for path in sorted(EVAL_RESULTS_DIR.glob("*.json")):
        if path.stem == "summary":
            continue
        with open(path, encoding="utf-8") as f:
            results[path.stem] = json.load(f)
    return results


def load_summary() -> list[dict] | None:
    path = EVAL_RESULTS_DIR / "summary.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔍 RAGLens")
    st.caption("RAG Pipeline Inspector")
    st.divider()

    st.subheader("Pipeline Config")
    chunking = st.selectbox(
        "Chunking strategy",
        ["recursive", "fixed"],
        help="recursive: respects paragraph/line boundaries; fixed: hard character splits",
    )
    chunk_size = st.slider("Chunk size (chars)", 200, 2000, 1000, 100)
    chunk_overlap = st.slider("Chunk overlap (chars)", 0, 400, 150, 25)
    top_k = st.slider("Top-k retrieved chunks", 1, 15, 5)

    st.divider()
    st.caption("Run `scripts/run_eval.py` to generate evaluation results.")


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_inspector, tab_metrics, tab_browser, tab_compare = st.tabs([
    "🔎 Retrieval Inspector",
    "📊 Eval Metrics",
    "📋 Question Browser",
    "⚖️ Strategy Comparison",
])


# ===========================================================================
# TAB 1 — Retrieval Inspector
# ===========================================================================

with tab_inspector:
    st.header("Retrieval Inspector")
    st.write(
        "Enter any question to see which chunks the pipeline retrieves, "
        "their similarity scores, and which source documents they come from."
    )

    query = st.text_input(
        "Your question",
        placeholder="e.g. What is gamma-ray emission region?",
        key="inspector_query",
    )

    col_run, col_info = st.columns([1, 4])
    run_btn = col_run.button("Retrieve", type="primary", use_container_width=True)

    if run_btn and query.strip():
        pipeline = get_pipeline(chunking, chunk_size, chunk_overlap)
        chunks = pipeline.retrieve(query.strip(), k=top_k)

        st.divider()
        st.subheader(f"Top {top_k} retrieved chunks")

        # Similarity score bar chart
        df_scores = pd.DataFrame([
            {"Rank": i + 1, "Similarity": c.similarity, "Title": c.title[:60]}
            for i, c in enumerate(chunks)
        ])
        fig = px.bar(
            df_scores, x="Rank", y="Similarity", text="Title",
            color="Similarity", color_continuous_scale="Blues",
            title="Chunk similarity scores (higher = more relevant)",
            height=300,
        )
        fig.update_traces(textposition="outside")
        fig.update_layout(coloraxis_showscale=False, yaxis_range=[0, 1])
        st.plotly_chart(fig, use_container_width=True)

        # Detailed chunk cards
        for i, chunk in enumerate(chunks):
            with st.expander(
                f"Rank {i+1} — {chunk.title[:80]}  "
                f"(sim={chunk.similarity:.3f})",
                expanded=(i == 0),
            ):
                col_meta, col_content = st.columns([1, 3])
                with col_meta:
                    st.markdown(f"**Doc ID:** `{chunk.doc_id}`")
                    st.markdown(f"**Chunk ID:** `{chunk.chunk_id}`")
                    st.markdown(f"**L2 distance:** `{chunk.l2_distance:.4f}`")
                    st.markdown(f"**Similarity:** `{chunk.similarity:.4f}`")
                with col_content:
                    st.markdown(chunk.content)

    elif run_btn:
        st.warning("Please enter a question first.")


# ===========================================================================
# TAB 2 — Eval Metrics
# ===========================================================================

with tab_metrics:
    st.header("Evaluation Metrics")
    st.write(
        "Aggregate Recall@k, Precision@k, and MRR across all eval questions "
        "for each saved configuration."
    )

    results = load_eval_results()

    if not results:
        st.info(
            "No evaluation results found in `data/eval_results/`.\n\n"
            "Run the following commands first:\n"
            "```bash\n"
            "python scripts/generate_eval_set.py\n"
            "python scripts/run_eval.py\n"
            "```"
        )
    else:
        # ---- Recall@k line chart ----------------------------------------
        recall_rows = []
        for cfg_name, report in results.items():
            for k_str, val in report["macro_recall_at_k"].items():
                recall_rows.append({"Config": cfg_name, "k": int(k_str), "Recall@k": val})

        df_recall = pd.DataFrame(recall_rows)
        fig_recall = px.line(
            df_recall, x="k", y="Recall@k", color="Config",
            markers=True,
            title="Recall@k by Configuration",
            labels={"k": "k (number of retrieved chunks)", "Recall@k": "Recall@k"},
        )
        fig_recall.update_layout(yaxis_range=[0, 1.05])
        st.plotly_chart(fig_recall, use_container_width=True)

        # ---- MRR bar chart -----------------------------------------------
        mrr_rows = [
            {"Config": cfg_name, "MRR": report["macro_mrr"]}
            for cfg_name, report in results.items()
        ]
        df_mrr = pd.DataFrame(mrr_rows).sort_values("MRR", ascending=False)
        fig_mrr = px.bar(
            df_mrr, x="Config", y="MRR", color="MRR",
            color_continuous_scale="Greens",
            title="Mean Reciprocal Rank (MRR) by Configuration",
        )
        fig_mrr.update_layout(yaxis_range=[0, 1], coloraxis_showscale=False)
        st.plotly_chart(fig_mrr, use_container_width=True)

        # ---- Per-question MRR distribution --------------------------------
        st.subheader("Per-question Reciprocal Rank distribution")
        dist_rows = []
        for cfg_name, report in results.items():
            for qr in report["per_question"]:
                dist_rows.append({
                    "Config": cfg_name,
                    "Reciprocal Rank": qr["reciprocal_rank"],
                })
        df_dist = pd.DataFrame(dist_rows)
        fig_dist = px.histogram(
            df_dist, x="Reciprocal Rank", color="Config",
            barmode="overlay", opacity=0.6,
            title="Distribution of Reciprocal Rank across Questions",
            nbins=12,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # ---- Summary table -----------------------------------------------
        st.subheader("Summary table")
        summary_rows = []
        for cfg_name, report in results.items():
            row = {"Config": cfg_name, "MRR": f"{report['macro_mrr']:.3f}"}
            for k_str, val in sorted(report["macro_recall_at_k"].items(), key=lambda x: int(x[0])):
                row[f"Recall@{k_str}"] = f"{val:.3f}"
            summary_rows.append(row)
        st.dataframe(pd.DataFrame(summary_rows).set_index("Config"), use_container_width=True)


# ===========================================================================
# TAB 3 — Question Browser
# ===========================================================================

with tab_browser:
    st.header("Question Browser")
    st.write(
        "Select a configuration and a question to inspect exactly which chunks "
        "were retrieved and whether the correct document was found."
    )

    results = load_eval_results()

    if not results:
        st.info("Run `python scripts/run_eval.py` first to generate results.")
    else:
        cfg_names = list(results.keys())
        selected_cfg = st.selectbox("Configuration", cfg_names, key="browser_cfg")
        report = results[selected_cfg]

        questions = [qr["question"] for qr in report["per_question"]]
        selected_q_idx = st.selectbox(
            "Question",
            range(len(questions)),
            format_func=lambda i: f"Q{i+1}: {questions[i][:90]}",
            key="browser_q",
        )
        qr = report["per_question"][selected_q_idx]

        st.divider()

        # Metrics for this question
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Reciprocal Rank", f"{qr['reciprocal_rank']:.3f}")
        for col, k in zip([m_col2, m_col3, m_col4], [1, 3, 5]):
            col.metric(f"Recall@{k}", f"{qr['recall_at_k'].get(str(k), 0):.3f}")

        st.write(f"**Relevant doc IDs:** {', '.join(qr['relevant_doc_ids'])}")

        st.divider()
        st.subheader("Retrieved chunks")

        for i, chunk in enumerate(qr["retrieved"]):
            is_rel = chunk["is_relevant"]
            badge = RANK_BADGE[is_rel]
            bg = CHUNK_COLORS[is_rel]
            label = "RELEVANT" if is_rel else "not relevant"

            with st.container():
                st.markdown(
                    f"""<div style="background:{bg}; border-radius:6px; padding:10px; margin-bottom:8px;">
                    <b>Rank {i+1} {badge} {label.upper()}</b> &nbsp;|&nbsp;
                    <code>{chunk['doc_id']}</code> &nbsp;|&nbsp;
                    sim=<b>{chunk['similarity']:.3f}</b><br>
                    <i>{chunk['title'][:100]}</i>
                    </div>""",
                    unsafe_allow_html=True,
                )
                with st.expander("Show chunk content"):
                    st.write(chunk["content"])


# ===========================================================================
# TAB 4 — Strategy Comparison
# ===========================================================================

with tab_compare:
    st.header("Strategy Comparison")
    st.write(
        "Compare two configurations side by side to understand which chunking "
        "strategy and parameters perform better."
    )

    results = load_eval_results()

    if len(results) < 2:
        st.info(
            "Need at least two evaluation results to compare.\n\n"
            "Run `python scripts/run_eval.py` to evaluate all built-in configs."
        )
    else:
        cfg_names = list(results.keys())

        col_a, col_b = st.columns(2)
        with col_a:
            cfg_a = st.selectbox("Config A", cfg_names, index=0, key="cmp_a")
        with col_b:
            cfg_b = st.selectbox("Config B", cfg_names, index=min(1, len(cfg_names) - 1), key="cmp_b")

        rep_a = results[cfg_a]
        rep_b = results[cfg_b]

        # Metric deltas
        st.divider()
        st.subheader("Metric comparison")

        metrics_col = st.columns(len(KS) + 1)
        labels = ["MRR"] + [f"Recall@{k}" for k in KS]
        vals_a = [rep_a["macro_mrr"]] + [rep_a["macro_recall_at_k"].get(str(k), 0) for k in KS]
        vals_b = [rep_b["macro_mrr"]] + [rep_b["macro_recall_at_k"].get(str(k), 0) for k in KS]

        for col, label, va, vb in zip(metrics_col, labels, vals_a, vals_b):
            delta = vb - va
            col.metric(label, f"{vb:.3f}", delta=f"{delta:+.3f}", delta_color="normal")
            col.caption(f"A: {va:.3f}")

        # Overlapping Recall@k chart
        st.divider()
        compare_rows = []
        for k_str in sorted(rep_a["macro_recall_at_k"].keys(), key=int):
            compare_rows.append({
                "k": int(k_str),
                cfg_a: rep_a["macro_recall_at_k"][k_str],
                cfg_b: rep_b["macro_recall_at_k"][k_str],
            })
        df_cmp = pd.DataFrame(compare_rows)
        fig_cmp = go.Figure()
        fig_cmp.add_trace(go.Scatter(
            x=df_cmp["k"], y=df_cmp[cfg_a], mode="lines+markers", name=cfg_a
        ))
        fig_cmp.add_trace(go.Scatter(
            x=df_cmp["k"], y=df_cmp[cfg_b], mode="lines+markers", name=cfg_b
        ))
        fig_cmp.update_layout(
            title="Recall@k: A vs B",
            xaxis_title="k",
            yaxis_title="Recall@k",
            yaxis_range=[0, 1.05],
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Per-question winner scatter
        st.divider()
        st.subheader("Per-question reciprocal rank: A vs B")

        qs_a = {qr["question"]: qr["reciprocal_rank"] for qr in rep_a["per_question"]}
        qs_b = {qr["question"]: qr["reciprocal_rank"] for qr in rep_b["per_question"]}
        common_qs = sorted(set(qs_a) & set(qs_b))

        scatter_rows = []
        for q in common_qs:
            rr_a = qs_a[q]
            rr_b = qs_b[q]
            winner = cfg_b if rr_b > rr_a else (cfg_a if rr_a > rr_b else "tie")
            scatter_rows.append({
                "question": q[:60] + "…" if len(q) > 60 else q,
                cfg_a: rr_a,
                cfg_b: rr_b,
                "winner": winner,
            })

        if scatter_rows:
            df_scatter = pd.DataFrame(scatter_rows)
            fig_scatter = px.scatter(
                df_scatter,
                x=cfg_a,
                y=cfg_b,
                color="winner",
                hover_data=["question"],
                title=f"Reciprocal Rank per question: {cfg_a} vs {cfg_b}",
                labels={cfg_a: f"RR ({cfg_a})", cfg_b: f"RR ({cfg_b})"},
            )
            # Diagonal y=x line
            fig_scatter.add_shape(
                type="line", x0=0, y0=0, x1=1, y1=1,
                line=dict(color="gray", dash="dot"),
            )
            fig_scatter.update_layout(xaxis_range=[-0.05, 1.1], yaxis_range=[-0.05, 1.1])
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption(
                f"Points above the diagonal: Config B ({cfg_b}) wins. "
                f"Points below: Config A ({cfg_a}) wins."
            )
