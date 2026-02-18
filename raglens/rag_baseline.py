# conda activate raglens
# conda install -c conda-forge faiss-cpu -y
# pip install -r requirements.txt
# python3 -m raglens.rag_baseline

from raglens.ingest import load_jsonl
from raglens.chunking import chunk_docs_fixed, chunk_docs_recursive

# from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

def main():
    docs = load_jsonl("data/raw/docs.jsonl")
    chunks = chunk_docs_recursive(docs, chunk_size=1000, chunk_overlap=150)
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/e5-base-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)

    query = "What is gamma-ray emission region?"
    results = vectorstore.similarity_search_with_score(query, k=5)

    for rank, (result, score) in enumerate(results):
        print(f"Rank {rank + 1} (score: {score:.4f}):")
        print(result.page_content)
        print("Metadata:", result.metadata)
        print("-" * 50)

if __name__ == "__main__":
    main()