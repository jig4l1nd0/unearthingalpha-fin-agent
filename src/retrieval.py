import shutil
import os
from abc import ABC, abstractmethod
from typing import List, Optional

# Core Imports
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever


class SearchComponent(ABC):
    """
    Abstract class to handle search strategies 
    Any search strategy must implement this methods.
    This allows the Pipeline to treat Dense, Hybrid, and Reranking identically.
    """
    @abstractmethod
    def search(self, query: str) -> List[Document]:
        pass


class DenseRetriever(SearchComponent):
    """
    Wraps ChromaDB. Performs K-Nearest Neighbor search on vectors.
    """
    def __init__(self, vectorstore):
        """
        retriever: LangChain retriever object created from the ChromaDB vectorstore
        """
        # We retrieve top 10 results to ensure we catch enough context
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

    def search(self, query: str) -> List[Document]:
        return self.retriever.invoke(query)


class HybridRetriever(SearchComponent):
    """
    Combines Dense (Vectors) and Sparse (BM25) using a weighted average.
    """
    def __init__(self, bm25_retriever, vectorstore):
        dense = vectorstore.as_retriever(search_kwargs={"k": 10})

        # JUSTIFICATION: We weight BM25 (0.6) higher than Dense (0.4).
        # In finance, if the user asks for "Project XYZ", they want that exact phrase.
        # Dense search might give generic "Project Management" results.
        self.ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, dense],
            weights=[0.6, 0.4]
        )

    def search(self, query: str) -> List[Document]:
        return self.ensemble.invoke(query)


class RetrievalPipeline:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir

        # JUSTIFICATION: 'all-MiniLM-L6-v2' is the industry standard for CPU-based RAG.
        # It is 5x faster than 'mpnet' with only a 2% drop in accuracy.
        print("Loading Embedding Model (MiniLM)...")
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.vectorstore = None
        self.bm25_retriever = None  # <--- NEW State
        self.active_retriever: Optional[SearchComponent] = None

    def index_documents(self, chunks: List[Document]):
        """Builds the Vector Index."""
        # Cleanup ensures idempotency (running the script twice doesn't crash)
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)

        # 1. Build Dense
        print(f"Indexing {len(chunks)} chunks (Dense)...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks, 
            embedding=self.embedding_model,
            persist_directory=self.persist_dir
        )

        # 2. Build Sparse (NEW)
        print("Indexing chunks (BM25)...")
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 10

        # Default to Dense (Baseline)
        self.set_mode("dense")

    def set_mode(self, mode: str):
        """Allows toggling strategies for evaluation."""
        if not self.vectorstore:
            raise ValueError("Index not built!")

        print(f"-> Switching to {mode} mode")
        if mode == "dense":
            self.active_retriever = DenseRetriever(self.vectorstore)
        elif mode == "hybrid":
            self.active_retriever = HybridRetriever(self.bm25_retriever, self.vectorstore)
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def query(self, query_text: str) -> List[Document]:
        if not self.active_retriever:
            raise ValueError("Run index_documents() first.")
        return self.active_retriever.search(query_text)
