import shutil
import os
from abc import ABC, abstractmethod
from typing import List, Optional

# Core Imports
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


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


class RetrievalPipeline:
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir

        # JUSTIFICATION: 'all-MiniLM-L6-v2' is the industry standard for CPU-based RAG.
        # It is 5x faster than 'mpnet' with only a 2% drop in accuracy.
        print("Loading Embedding Model (MiniLM)...")
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        self.vectorstore = None
        self.active_retriever: Optional[SearchComponent] = None

    def index_documents(self, chunks: List[Document]):
        """Builds the Vector Index."""
        # Cleanup ensures idempotency (running the script twice doesn't crash)
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)

        print(f"Indexing {len(chunks)} chunks...")
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embedding_model,
            persist_directory=self.persist_dir
        )

        # Initialize Dense Retrieval by default
        self.active_retriever = DenseRetriever(self.vectorstore)
        print("Dense Indexing Complete.")

    def query(self, query_text: str) -> List[Document]:
        if not self.active_retriever:
            raise ValueError("Index not built! Run index_documents() first.")
        return self.active_retriever.search(query_text)
