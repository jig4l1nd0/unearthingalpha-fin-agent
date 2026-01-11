import os
from abc import ABC, abstractmethod
from typing import List
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document


class ChunkingStrategy(ABC):
    """
    Abstract Base Class for chunking strategies. Allows swapping
    algorithms ( Token, Character. ...) without changing the pipeline.
    """
    @abstractmethod
    def chunk(self, documents: List[Document]) -> List[Document]:
        pass


class RecursiveTokenChunking(ChunkingStrategy):
    """
    Splits text while respecting paragraph boundaries.

    Attributes:
        chunk_size: Target size of each chunk (in characters).
        chunk_overlap: Amount of overlap to preserve context at boundaries.
    """
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def chunk(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


class IngestionPipeline:
    def __init__(self, data_dir: str, chunking_strategy: ChunkingStrategy):
        self.data_dir = data_dir
        self.chunker = chunking_strategy

    def load_documents(self) -> List[Document]:
        """Loads all .txt files from the data directory."""
        documents = []

        # Validation
        if not os.path.exists(self.data_dir):
            print(f"[WARNING] Directory '{self.data_dir}' not found.")
            return []

        files = [f for f in os.listdir(self.data_dir) if f.endswith(".txt")]
        if not files:
            print(f"[WARNING] No .txt files found in '{self.data_dir}'.")
            return []

        print(f"Loading {len(files)} documents from '{self.data_dir}'...")

        for filename in files:
            path = os.path.join(self.data_dir, filename)
            try:
                # Load text
                loader = TextLoader(path, encoding='utf-8') # for $ symbols
                loaded_docs = loader.load()

                # Clean and Tag
                for doc in loaded_docs:
                    doc.metadata["source"] = filename
                    # Cleaning: Collapse multiple spaces/newlines into one
                    doc.page_content = " ".join(doc.page_content.split())

                documents.extend(loaded_docs)
            except Exception as e:
                print(f"Error loading {filename}: {e}")

        return documents

    def run(self) -> List[Document]:
        """Executes the full pipeline: Load -> Clean -> Chunk"""
        raw_docs = self.load_documents()
        if not raw_docs:
            return []

        chunks = self.chunker.chunk(raw_docs)
        print(f"-> Successfully created {len(chunks)} chunks.")
        return chunks
