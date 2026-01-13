# UnearthingAlpha Financial RAG Agent — Technical Design and Engineering Decisions

UnearthingAlpha is an open‑source, single‑CPU RAG pipeline that downloads recent Apple 10‑K filings, cleans inline XBRL HTML into plain text, chunks the text, builds a hybrid dense+BM25 index, and retrieves relevant passages — all locally without paid APIs. Answers can be generated with a local Ollama model (generation wiring is not yet complete). The project prioritizes transparency, exact citations, reproducibility, and explainable choices over maximum accuracy.

----------------------------------------------------------------
The name UnearthingAlpha reflects our goal to dig through dense SEC filings and expose actionable insights. This repo implements a pragmatic, open-source RAG pipeline that runs end-to-end on a single CPU, emphasizing transparency, citations, and reproducibility. It favors simple, explainable engineering that can be audited and improved over time.

----------------------------------------------------------------

This document explains the engineering decisions, architecture, and trade-offs in this repository, aligned with the challenge goals and constraints.

- Clean, production-quality Python
- Modern open-source RAG techniques on real SEC 10-Ks
- End-to-end runnable with basic evaluation and proposed improvements

Repo entry points:
- CLI: [main.py](main.py)

Key modules:
- Data acquisition/cleaning: [download_data.py](download_data.py)
- Ingestion/chunking: [src/ingestion.py](src/ingestion.py)
- Retrieval (dense + BM25 hybrid): [src/retrieval.py](src/retrieval.py)
- Generation (local LLM via Ollama): [src/generation.py](src/generation.py) WIP

----------------------------------------------------------------

## 1) Goals and Constraints

Challenge goals
- Clean, production-grade Python
- Apply modern LLM + retrieval techniques to real documents
- Evaluate rigorously and propose improvements

Hard constraints satisfied
- Open-source only (Hugging Face embeddings, FAISS, BM25, LangChain community, bs4, lxml, sec_edgar_downloader)
- Local/CPU friendly by default
- End-to-end execution via a single command (CLI) or notebook
- All code is original; external references kept to library docs

Environment
- Python 3.10+
- CPU laptop or single CPU VM
- No proprietary APIs required; generation uses local Ollama

Allowed libraries used
- sentence-transformers, faiss-cpu, rank-bm25, langchain-community, langchain_huggingface, bs4, lxml, sec_edgar_downloader

----------------------------------------------------------------

## 2) Quickstart

The fastest way to see the system working is to run one of these commands:

Single command (end-to-end ingestion + retrieval):
- Download SEC filings (if needed), ingest, index, and query
- Toggle retrieval mode: dense or hybrid (BM25+dense)

Examples:
```bash
# Most common first run (downloads filings if missing, builds index, runs hybrid search)
python main.py --download --mode hybrid --query "List the main risk factors disclosed by Apple" --top-k 6
# Prints ~6 matching text chunks with filenames

# Subsequent runs (reuse cleaned data)
python main.py --mode dense --query "Describe revenue recognition policies"
# Prints ~5 matching text chunks with filenames
```

Expected runtime: first run (download + ingest + index build) usually takes 3–12 minutes depending on internet speed and number of filings. Later queries usually take 1–4 seconds on a typical laptop CPU.

### Setup from a fresh clone

Prerequisites
- Python 3.10+ on macOS/Linux (CPU-only environment is fine)

Commands
```bash
# 1) Clone and enter the project
git clone https://github.com/jig4l1nd0/unearthingalpha-fin-agent.git
cd unearthingalpha-fin-agent

# 2) Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# 4) Run end-to-end (downloads filings if needed, builds index, queries)
python main.py --download --mode hybrid --query "List the main risk factors disclosed by Apple" --top-k 6
```

 

----------------------------------------------------------------

## 3) System Architecture

High-level flow
1) Download raw SEC submissions (Workiva inline XBRL HTML) to temp folder
2) Extract readable text from inline XBRL into clean .txt files under data/
3) Ingest and chunk texts into overlapping segments
4) Index chunks into FAISS (dense) and BM25 (sparse), support hybrid retrieval
5) Optionally generate a cited answer via local Ollama (module implemented; CLI wiring is a TODO)

Components
- Data acquisition/cleaning: [`download_and_process`](download_data.py), [`find_html_content_in_sec_filing`](download_data.py), [`extract_readable_content_from_xbrl`](download_data.py)
- Ingestion/chunking: [`src.ingestion.IngestionPipeline`](src/ingestion.py), [`src.ingestion.RecursiveTokenChunking`](src/ingestion.py)
- Retrieval: [`src.retrieval.RetrievalPipeline`](src/retrieval.py), [`src.retrieval.DenseRetriever`](src/retrieval.py), [`src.retrieval.HybridRetriever`](src/retrieval.py)
- Generation: [`src.generation.RAGGenerator`](src/generation.py)
- CLI: [`main.maybe_download_data`](main.py), [`main.ingest_data`](main.py), [`main.build_retriever`](main.py), [`main.run_retrieval`](main.py), [`main.main`](main.py)

----------------------------------------------------------------

## 4) Data Acquisition and Cleaning

File: [download_data.py](download_data.py)

- Downloader: sec_edgar_downloader
  - Stores raw submissions under temp_sec_raw/sec-edgar-filings/TICKER/10-K/ACCESSION-ID
- HTML extraction: [`find_html_content_in_sec_filing`](download_data.py)
  - Scans each <DOCUMENT> <TEXT> block to find the inline HTML payload
  - Heuristic length check (>10,000 chars) to select the main 10-K
- Readable text extraction: [`extract_readable_content_from_xbrl`](download_data.py)
  - Parses with BeautifulSoup (lxml)
  - Unwraps inline XBRL tags (e.g., ix:nonNumeric) but keeps text
  - Drops script/style/head/title/meta/link blocks
  - Filters boilerplate lines and normalizes whitespace
- Output: Saved as data/{TICKER}_10K_{ACCESSION}.txt
  - See examples: [data/AAPL_10K_0000320193-21-000105.txt](data/AAPL_10K_0000320193-21-000105.txt), [data/AAPL_10K_0000320193-24-000123.txt](data/AAPL_10K_0000320193-24-000123.txt)

Key decisions
- Workiva inline XBRL handling: strip namespaces but preserve readable text
- Conservative filtering of lines/URLs to avoid stripping content-heavy lines
- Minimum content length guard to discard failed parses
- Open-source stack (bs4, lxml) and file-based pipeline for reproducibility

Trade-offs
- Heuristic HTML selection might miss edge cases
- No table-to-text semantic reconstruction yet (kept simple for CPU/local)

----------------------------------------------------------------

## 5) Ingestion and Chunking

File: [src/ingestion.py](src/ingestion.py)

Abstractions
- Strategy pattern for chunking:
  - [`src.ingestion.ChunkingStrategy`](src/ingestion.py) (abstract)
  - [`src.ingestion.RecursiveTokenChunking`](src/ingestion.py): uses LangChain RecursiveCharacterTextSplitter

Pipeline
- [`src.ingestion.IngestionPipeline`](src/ingestion.py)
  - Loads .txt files
  - Normalizes whitespace
  - Tags each document with filename as metadata source
  - Splits into overlapping chunks

Defaults
- Chunk size: 1000 chars
- Overlap: 200 chars
- Separators: paragraph → line → sentence → word → char

Rationale
- Overlap preserves cross-boundary context for retrieval
- Character-based splitter avoids tokenizer dependencies and remains CPU-only
- File-level source metadata supports downstream citation

Alternatives
- Token-based splitting (e.g., tiktoken) for tighter control
- Section-aware splitting (headings/tables) to improve retrieval precision

----------------------------------------------------------------

### Why recursive character chunking was chosen (trade-offs)

Given the constraints (CPU-only, open-source, 3–4 hours), we compare common chunking approaches and select a pragmatic default.

Strategy comparison

| Strategy | How it works | Advantages | Disadvantages | Good fit | OSS building blocks | Sources |
|---|---|---|---|---|---|---|
| Recursive character/paragraph splitter | Recursively split by larger→smaller separators (paragraph → line → sentence → word → char) until below size | Simple, fast, CPU-only; robust on mixed-format text; strong baseline | May cut sentences; uneven semantic units; table structures not preserved | Baseline RAG on long prose (10‑Ks) under CPU constraints | LangChain `RecursiveCharacterTextSplitter`; LlamaIndex splitters | [1], [2] |
| Token-based splitter | Split by token counts using a tokenizer (e.g., `tiktoken`) | Model-aligned sizing; better control vs context window | Tokenizer dependency; slightly slower; tokenizer-specific | Tight control for long-context models and budgeted calls | LangChain `TokenTextSplitter`; `tiktoken` | [1] |
| Sentence/semantic splitter | Segment by sentences or semantic similarity (e.g., Punkt/senter; similarity-based merges) | Higher coherence; often improves first-hit precision | Extra preprocessing; variable chunk sizes; punctuation sensitivity | Narrative sections (Risk Factors, MD&A) | NLTK Punkt, spaCy senter; LlamaIndex semantic splitters | [2], [3], [4] |
| Section/header-aware splitter | Use headings (e.g., “Item 1A. Risk Factors”) and hierarchy to form chunks | Aligns with 10‑K structure; boosts precision and citation clarity | Heavier parsing; long sections require sub-splitting; formatting edge cases | Financial filings, manuals, papers | Regex + BeautifulSoup; LlamaIndex header/hierarchical splitters | [2] |
| Fixed-size sliding window | Fixed window with stride/overlap | Very simple; good recall with overlap; deterministic | Poor semantic boundaries; duplication; larger index | Noisy/unstructured text; quick baseline | LangChain `CharacterTextSplitter` (fixed) | [1] |

Chunk size and overlap guidance

| Use case | Chunk size | Overlap | Rationale | Sources |
|---|---|---|---|---|
| Baseline finance RAG (this repo) | 800–1200 chars | 150–250 chars | Balance recall/coherence/index size; overlap guards mid-paragraph splits | [1], [2] |
| Long-context local LLMs | 2000–4000 tokens | 10–15% of size | Fewer calls, self-contained context; mind “lost in the middle” | [5] |
| Section-aware pipelines | Variable per section | 0–10% | Use headings as primary boundary; sub-split long sections | [2] |

Best choice under current constraints (decision)

- Choice: Recursive character splitter with `chunk_size=1000` and `chunk_overlap=200` (already implemented).
- Pros:
  - CPU-only, no tokenizer dependency; fast to implement and run.
  - Strong baseline for long prose (10‑Ks) with minimal engineering.
  - Overlap preserves context across paragraph boundaries, improving recall.
- Cons:
  - Occasionally splits sentences/tables; some coherence loss vs sentence/section-aware.
  - Not perfectly aligned with model token limits.

Upgrade path (when time/compute allow)
- Header-aware primary splits (detect “Item X” headings) + token-aware sub-splitting to fit model windows.
- Optional sentence-aware merges at chunk boundaries to improve coherence.

Sources (verifiable)
1) LangChain Text Splitters: https://python.langchain.com/docs/modules/data_connection/document_transformers/text_splitters  
2) LlamaIndex chunking overview: https://docs.llamaindex.ai/en/stable/module_guides/loading/ingestion/chunking  
3) NLTK sentence tokenization (Punkt): https://www.nltk.org/api/nltk.tokenize.html  
4) spaCy sentence segmentation: https://spacy.io/usage/linguistic-features#sbd  
5) “Lost in the Middle” (Liu et al., 2023): https://arxiv.org/abs/2307.03172

----------------------------------------------------------------

## 6) Retrieval (Dense + BM25 Hybrid)

File: [src/retrieval.py](src/retrieval.py)

Abstractions
- Search strategy interface: [`src.retrieval.SearchComponent`](src/retrieval.py)
- Dense retriever: [`src.retrieval.DenseRetriever`](src/retrieval.py)
- Hybrid retriever: [`src.retrieval.HybridRetriever`](src/retrieval.py)

Pipeline
- [`src.retrieval.RetrievalPipeline`](src/retrieval.py)
  - Embeddings: sentence-transformers all-MiniLM-L6-v2 (CPU-friendly, strong baseline)
  - Vector store: FAISS (CPU-friendly vector index; can be saved/loaded via LangChain FAISS wrappers)
  - Sparse retriever: BM25 (rank-bm25)
  - Ensemble: LangChain EnsembleRetriever with weights [BM25=0.6, Dense=0.4]
  - Modes: set_mode("dense" | "hybrid")
  - Idempotent index build (cleans persist_dir before rebuild)

Note on vector store choice
- Chroma was initially evaluated but discarded due to pydantic compatibility issues in our environment. We switched to FAISS to keep the stack fully functional under the stated constraints.

Rationale
- CPU-first: MiniLM is 5x faster than mpnet with small accuracy delta
- Hybrid retrieval improves exact-term matching (SEC jargon) + semantic recall
- Weighted ensemble biases towards lexical precision for finance use cases

Trade-offs
- No cross-encoder re-ranking step yet (kept simple, CPU-only)
- Weights are fixed; could be tuned via evaluation

----------------------------------------------------------------

## 7) Generation (Local LLM, Citations) — status

File: [src/generation.py](src/generation.py)

- Generator class: [`src.generation.RAGGenerator`](src/generation.py)
  - Backend: LangChain ChatOllama (local, no API key)
  - Deterministic settings: temperature=0 for reduced hallucinations
  - Strict prompt requiring citations and honesty
  - Formats retrieved docs with explicit [Source: filename] tags
  - Output structure: { answer_text, citations, retrieved_chunks }

Prompt highlights
- “Answer ONLY from context”
- “If unknown, say you don’t have enough information”
- “Cite sources for every key fact”

Rationale
- Open-source, offline generation consistent with constraints
- Explicit citation scaffolding to enable traceability

Status
- The generation module is implemented, but CLI integration is not yet complete. The current CLI prints retrieved snippets followed by a generation placeholder.


----------------------------------------------------------------

## 8) CLI Orchestration

File: [main.py](main.py)

- Download: [`main.maybe_download_data`](main.py)
- Ingestion: [`main.ingest_data`](main.py)
- Retrieval: [`main.build_retriever`](main.py) and [`main.run_retrieval`](main.py)
- Entry point: [`main.main`](main.py)

Behavior
- Optional download on first run (or with --download)
- Builds both dense and BM25 indices; supports switching modes
- Prints top-k retrieved chunks with source filenames
- Generation currently a placeholder (see “Improvements”)

Example:
```bash
python main.py --download --mode hybrid --query "What are principal market risks?" --top-k 5
```

----------------------------------------------------------------

## 9) Evaluation Approach (current & planned)

Current
- Manual side-by-side comparison of `dense` vs `hybrid` on 8–12 finance-related questions.
- Log latency and chunk counts; note CPU model and Python version.

Planned
- Curated question set with gold passages; standard retrieval metrics (Recall@k, MRR).
- ragas-based generation faithfulness and relevance (once generation is wired into CLI).
- Optional ablations: ensemble weights sweep and cross-encoder reranking toggle.

Quick check commands
```bash
python main.py --mode hybrid --query "What risk factors does Apple report?" --top-k 5
python main.py --mode dense  --query "What risk factors does Apple report?" --top-k 5
```

----------------------------------------------------------------

## 10) Design Patterns and Engineering Practices

- Strategy pattern
  - Chunking: interchangeable via [`src.ingestion.ChunkingStrategy`](src/ingestion.py)
  - Retrieval strategies: [`src.retrieval.SearchComponent`](src/retrieval.py) with dense/hybrid implementations
- Pipeline composition
  - Clear stages: acquire → clean → ingest → retrieve → generate
- Idempotent builds
  - Vector index directory cleaned before rebuild to ensure determinism
- Separation of concerns
  - Data acquisition/cleaning vs ingestion vs retrieval vs generation
- Reproducibility
  - Local-only stack; deterministic generation settings
- Observability
  - Explicit prints/logging for each stage, guards for common failure modes

----------------------------------------------------------------

## 11) Performance and Resource Considerations

- CPU-first model: all-MiniLM-L6-v2 for embeddings
- FAISS for fast, CPU-only vector similarity search
- BM25 for precise lexical matches in finance text
- Chunk size 1000 with 200 overlap balances recall vs index size
- k=10 retrieval by default; adjustable

Known costs
- Inline XBRL parsing and HTML cleanup can be I/O bound
- Hybrid retrieval performs two queries (BM25 + vector), slight overhead

----------------------------------------------------------------

## 12) Limitations and Improvement Plan

Near-term
- Current gap: generation is not yet wired into the CLI (placeholder output shown after retrieval).
- Wire generation into CLI:
  - Add a --generate flag and call [`src.generation.RAGGenerator`](src/generation.py)
- Token-aware chunking:
  - Replace character splitter with token splitter for improved boundary quality
- Re-ranking:
  - Add open-source cross-encoder (e.g., bge-reranker) after hybrid retrieval
- Weight tuning:
  - Empirically tune BM25 vs dense weights per query class

Mid-term
- Section-aware splitting:
  - Use headings and tables to create semantically coherent chunks
- Metadata enrichment:
  - Add section titles, dates, and doc types to improve retrieval filtering
- Caching and persistence:
  - Persistent embedding cache keyed by file hash to avoid re-embedding
- Evaluation suite:
  - Add ragas/evaluate-based tests with question-answer sets and citation checks

Long-term
- Multi-issuer ingestion:
  - Scale to multiple tickers with config-driven pipelines
- Advanced parsing:
  - Inline tables to text, selective footnote handling

----------------------------------------------------------------

## 13) Security and Privacy

- Local-only by default; no external API calls required
- Data is public (SEC 10-Ks), stored locally
- If using external notebooks or scripts, ensure no proprietary APIs are required; the generator uses local Ollama by default.

----------------------------------------------------------------

## 14) References (Libraries)

- sentence-transformers (all-MiniLM-L6-v2)
- faiss-cpu
- rank-bm25
- langchain-community, langchain-huggingface, langchain-text-splitters
- bs4, lxml
- sec_edgar_downloader

----------------------------------------------------------------

## 15) File Index (Key Symbols)

- Ingestion
  - [`src.ingestion.ChunkingStrategy`](src/ingestion.py)
  - [`src.ingestion.RecursiveTokenChunking`](src/ingestion.py)
  - [`src.ingestion.IngestionPipeline`](src/ingestion.py)
- Retrieval
  - [`src.retrieval.SearchComponent`](src/retrieval.py)
  - [`src.retrieval.DenseRetriever`](src/retrieval.py)
  - [`src.retrieval.HybridRetriever`](src/retrieval.py)
  - [`src.retrieval.RetrievalPipeline`](src/retrieval.py)
- Generation
  - [`src.generation.RAGGenerator`](src/generation.py)
- CLI
  - [`main.maybe_download_data`](main.py)
  - [`main.ingest_data`](main.py)
  - [`main.build_retriever`](main.py)
  - [`main.run_retrieval`](main.py)
  - [`main.main`](main.py)
- Data pipeline
  - [`download_and_process`](download_data.py)
  - [`find_html_content_in_sec_filing`](download_data.py)
  - [`extract_readable_content_from_xbrl`](download_data.py)


## 16) Vector Store Trade-offs: FAISS vs Chroma

We evaluated both FAISS and Chroma. The project currently uses FAISS. Chroma was initially trialed but was discarded in this repo due to a pydantic dependency compatibility issue encountered locally (environment-specific).

Summary decision
- Use FAISS for reliability under CPU-only constraints and minimal dependencies.
- What we lose vs Chroma: first-class metadata filters and built-in collection persistence (we compensate via wrappers and simple pre-filtering).
- Detailed comparison: see Appendix A.

----------------------------------------------------------------

## Appendix A — Detailed FAISS vs Chroma comparison

Comparison

| Aspect | FAISS | Chroma | Sources |
|---|---|---|---|
| Core strength | High-performance vector similarity (CPU/GPU), mature C++/Python library | Developer-friendly vector DB with persistence, metadata filtering, and client API | [FAISS](https://faiss.ai/) · [FAISS GitHub](https://github.com/facebookresearch/faiss) · [LangChain FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss/) · [Chroma Docs](https://docs.trychroma.com/) · [LangChain Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/) |
| Persistence | Save/load indexes (plus docstore via wrappers like LangChain) | Built-in persistent storage and collections | [LangChain FAISS save/load](https://python.langchain.com/docs/integrations/vectorstores/faiss/) · [Chroma persistence](https://docs.trychroma.com/usage-guide#persisting-data) |
| Metadata filtering | Not native in FAISS itself; done via wrapper pre-filtering or custom pipelines | First-class metadata filters (`where` queries) | [LangChain FAISS notes](https://python.langchain.com/docs/integrations/vectorstores/faiss/) · [Chroma filtering](https://docs.trychroma.com/usage-guide#querying) |
| Index types / performance | Multiple index types (IVF, PQ, HNSW via integrations), optimized for speed/recall trade-offs | Single-engine abstraction; focuses on usability over algorithm variety | [FAISS docs](https://faiss.ai/) |
| Operational model | Embedded library; no server required; great for CPU-only notebooks/CLIs | Embedded DB/server-like behaviors; simple local persistence, easy to inspect | [Chroma Docs](https://docs.trychroma.com/) |
| Ecosystem integration | Broad adoption in research/industry; widely supported in toolkits | Rich Python developer UX; growing RAG ecosystem adoption | [LangChain Integrations](https://python.langchain.com/docs/integrations/vectorstores/) |
| Dependency surface | Light runtime deps; simple to vendor into CPU environments | Heavier Python dep chain (e.g., pydantic-based models). Note: this repo hit a pydantic compatibility issue locally | Project note (env-specific) · [Chroma install](https://docs.trychroma.com/getting-started) |

Advantages/Disadvantages (summary)

| Library | Advantages | Disadvantages | Sources |
|---|---|---|---|
| FAISS | Fast CPU/GPU similarity search; mature; flexible index choices; easy to embed in Python pipelines | No native metadata filtering; requires wrapper-managed persistence/docstores | [FAISS](https://faiss.ai/) · [LangChain FAISS](https://python.langchain.com/docs/integrations/vectorstores/faiss/) |
| Chroma | Simple persistence and collections; first-class metadata filters; developer-friendly UX | Additional dependency surface; environment compatibility can vary; performance depends on workload | [Chroma Docs](https://docs.trychroma.com/) · [LangChain Chroma](https://python.langchain.com/docs/integrations/vectorstores/chroma/) |