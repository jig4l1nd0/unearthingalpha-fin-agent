import argparse
import os
from typing import List

# Local imports
from download_data import download_and_process
from src.ingestion import IngestionPipeline, RecursiveTokenChunking
from src.retrieval import RetrievalPipeline


def maybe_download_data(data_dir: str, force: bool) -> None:
    """
    Download and preprocess SEC 10-K filings if needed.
    If `force` is False and text files already exist in `data_dir`, skip download.
    """
    has_txt = any(
        f.endswith(".txt") for f in os.listdir(data_dir)
    ) if os.path.exists(data_dir) else False

    if force or not has_txt:
        print("[Download] Starting data download and preprocessing...")
        download_and_process()
    else:
        print(f"[Download] Found existing text files in '{data_dir}'. Skipping download.")


def ingest_data(data_dir: str):
    print("[Ingestion] Loading and chunking documents...")
    ingestion = IngestionPipeline(
        data_dir=data_dir,
        chunking_strategy=RecursiveTokenChunking(chunk_size=1000, chunk_overlap=200),
    )
    chunks = ingestion.run()
    if not chunks:
        raise RuntimeError("No chunks produced. Make sure data exists in the 'data/' folder.")
    return chunks


def build_retriever(chunks, mode: str = "dense") -> RetrievalPipeline:
    print("[Retrieval] Building indices (Dense + BM25)...")
    retriever = RetrievalPipeline(persist_dir="./chroma_db")
    retriever.index_documents(chunks)
    retriever.set_mode(mode)
    return retriever


def run_retrieval(retriever: RetrievalPipeline, query: str, top_k: int = 5) -> List[str]:
    print(f"[Retrieval] Query -- {query}")
    docs = retriever.query(query)
    results = []
    for i, d in enumerate(docs[:top_k], start=1):
        src = d.metadata.get("source", "unknown")
        if len(d.page_content) > 300:
            snippet = (d.page_content[:300] + "...")
        else:
            d.page_content
        results.append(f"{i}. [source: {src}] {snippet}")
    return results


def main():
    parser = argparse.ArgumentParser(description="Simple RAG pipeline: download -> ingest -> retrieve -> (generate placeholder)")
    parser.add_argument("--data-dir", default="data", help="Directory containing cleaned .txt files")
    parser.add_argument("--download", action="store_true", help="Force downloading and preprocessing filings")
    parser.add_argument("--mode", choices=["dense", "hybrid"], default="dense", help="Retrieval mode")
    parser.add_argument("--query", default="What risk factors does Apple report?", help="User query for retrieval")
    parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to display")

    args = parser.parse_args()

    # 1. Download 
    maybe_download_data(args.data_dir, force=args.download)

    # 2. Ingestion
    chunks = ingest_data(args.data_dir)

    # 3. Retrieval
    retriever = build_retriever(chunks, mode=args.mode)
    retrieved_snippets = run_retrieval(retriever, query=args.query, top_k=args.top_k)

    print("\n____ Retrieved Snippets ___")
    for line in retrieved_snippets:
        print(line)

    # 4) Generation (placeholder)
    print("\n____ Generation (Placeholder) ___")


if __name__ == "__main__":
    main()
