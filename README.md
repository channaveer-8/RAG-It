<h1 align="center">
  <br>
  <img src="ui/assets/logo.svg" alt="RAG It! Logo" width="80">
  <br>
  RAG It!
  <br>
</h1>

<h4 align="center">
  A local, multimodal Retrieval-Augmented Generation system that understands<br>
  <strong>text · images · tables</strong> from complex documents — all running on your machine.
</h4>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11%2B-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.13-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/PySide6-6.11-41CD52?style=for-the-badge&logo=qt&logoColor=white"/>
  <img src="https://img.shields.io/badge/Qdrant-1.19-DC244C?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Ollama-Local%20LLM-000000?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge"/>
</p>

<p align="center">
  <a href="#features">Features</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#prerequisites">Prerequisites</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#usage">Usage</a> •
  <a href="#development">Development</a>
</p>

---

## What Is RAG It!?

**RAG It!** is a **fully local**, **privacy-first** Retrieval-Augmented Generation pipeline that goes beyond plain-text RAG. It ingests complex documents (PDFs, DOCX, PPTX, Excel, Markdown) and understands every element inside them:

| Element | How It's Handled |
|:---|:---|
| 📄 **Text sections** | Recursively chunked and embedded with BGE-M3 |
| 🖼️ **Figures, charts, diagrams** | Cropped, stored as PNG, embedded with Visualized-BGE (shared semantic space) |
| 📊 **Tables** | Exported as structured JSON + Markdown, embedded and retrievable |

Answers are grounded in retrieved evidence with clickable source citations that open the original PDF at the exact page.

**Zero cloud dependencies** — everything runs on your local hardware.

---

## Features

- 🔍 **Multi-query retrieval + Reciprocal Rank Fusion (RRF)** — decomposes complex questions into targeted sub-queries for higher recall
- 🖼️ **True cross-modal search** — text queries can retrieve relevant figures without OCR (shared 1024-dim embedding space)
- 🏗️ **IBM Docling parsing** — layout-aware document decomposition preserving heading hierarchy, tables, and visual assets
- ⚡ **Streaming token generation** — real-time answer streaming directly in the UI at 60 FPS
- 📎 **Clickable source citations** — every answer includes document name, page numbers, and chunk-level provenance
- 🧮 **Two-stage reranking** — Cross-Encoder and BGE reranker for precision after recall
- 🖥️ **Native desktop GUI** — PySide6/QML dark-themed application, no browser required
- 🔒 **100% local** — no API keys, no cloud, no data leaving your machine
- 🗄️ **Persistent vector store** — embedded Qdrant database survives restarts

---

## Architecture

```
Document (PDF/DOCX/PPTX)
    │
    ▼
IBM Docling ──── Layout AST ────► Text Chunks
                                ► Cropped Image PNGs
                                ► Structured Table JSON
    │
    ▼
Visualized-BGE (BGE-M3 + ViT)
    │  1024-dim shared semantic space
    ▼
Local Qdrant Vector DB
    │
    ▼
User Query
    │
    ├─► RetrievalPlanner (Ollama) ──► 2–5 Sub-Queries
    │
    ├─► Multi-Query Dense Search (Qdrant)
    │
    ├─► Reciprocal Rank Fusion (RRF, k=60)
    │
    ├─► Cross-Encoder Reranker (optional Stage-2)
    │
    ▼
ContextBuilder ──► PromptBuilder ──► Ollama LLM ──► Streaming Answer + Citations
```

For full architecture diagrams, see [`ARCHITECTURE.md`](ARCHITECTURE.md).

---

## Prerequisites

Before installation, you need to set up **two external dependencies** manually:

### 1. Ollama (Local LLM Runtime)

Download and install from **https://ollama.com**

After installation, pull the default model:
```bash
ollama pull qwen3:4b
```

> Any Ollama-compatible model works. The app will auto-start the Ollama daemon on launch.

### 2. Visualized-BGE Model Weights

Download the **Visualized_m3.pth** weights (~1.74 GB):

```
https://huggingface.co/BAAI/bge-visualized/resolve/main/Visualized_m3.pth
```

Place the downloaded file at:
```
multi-model-rag-pipeline/models/Visualized_m3.pth
```

### 3. Python 3.11+

Download from **https://python.org** (3.11 or 3.12 recommended).

---

## Installation

### Option A: One-Click Automated Setup (Recommended)

Simply run:

```bash
python setup.py
```

This script automatically creates `.venv`, installs all dependencies, sets up the directory layout, and verifies your Ollama and model weight configurations.

---

### Option B: Manual Installation

#### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-it.git
cd rag-it
```

#### 2. Create and activate a virtual environment

```bash
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# Windows (Command Prompt)
.venv\Scripts\activate.bat

# macOS / Linux
source .venv/bin/activate
```

#### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **CUDA users**: For GPU acceleration, install the CUDA build of PyTorch:
> ```bash
> pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu121
> ```
> Then update `DEVICE = "cuda"` in `src/rag/config.py`.

#### 4. Place your model weights

Download `Visualized_m3.pth` (~1.74 GB) and place it at:
```
models/Visualized_m3.pth
```

#### 5. Verify Ollama

Make sure Ollama is running and has the model:
```bash
ollama pull qwen3.5:4b-q4_K_M
```

---

## Quick Start

### Launch the Desktop App

```bash
cd multi-model-rag-pipeline
python ui/main.py
```

The app will:
1. Auto-start Ollama if not running
2. Load the Visualized-BGE embedding model
3. Connect to the local Qdrant vector database
4. Open the RAG It! desktop interface

### Add a Document

1. Click the **"+ Add Document"** button in the left sidebar
2. Select any PDF, DOCX, PPTX, or other supported file
3. Wait for ingestion to complete (Docling parses, BGE embeds)
4. Start asking questions in the chat!

---

## Usage

### Desktop UI — Full Feature Mode

```bash
cd multi-model-rag-pipeline
python ui/main.py
```

**UI Features:**
- 📁 **Left sidebar**: Document manager — add, remove, view indexed documents
- 💬 **Center**: Chat with streaming token display and markdown rendering
- 📌 **Right sidebar**:
  - **Sources** tab: Ranked evidence chunks with page-click deep-links
  - **Images** tab: Retrieved figures with zoom modal
  - **Tables** tab: Structured table viewer
  - **Prompt** drawer: Customize the RAG system prompt on the fly

### CLI — Batch Ingestion

```bash
cd multi-model-rag-pipeline
python scripts/ingest.py
```

Ingests all documents from the `documents/` folder.

### CLI — Interactive RAG Chat

```bash
cd multi-model-rag-pipeline
python tests/test_rag.py
```

A terminal-based question-answering loop for quick testing.

---

## Supported Document Formats

| Format | Extension |
|:---|:---|
| PDF | `.pdf` |
| Word Document | `.docx` |
| PowerPoint | `.pptx` |
| Excel | `.xlsx` |
| HTML | `.html` |
| Markdown | `.md` |
| Plain Text | `.txt` |

---

## Development

### Run the Test Suite

```bash
cd multi-model-rag-pipeline

# Unit tests for individual components
python tests/test_embedding.py
python tests/test_chunker.py
python tests/test_qdrant_index.py
python tests/test_query_planner.py
python tests/test_context_builder.py
python tests/test_prompt_builder.py
python tests/test_ollama_manager.py
python tests/test_cross_encoder_reranker.py
python tests/test_single_document_index.py

# Benchmark retrieval quality (Hit@1, Hit@3, Hit@5, MRR)
python tests/test_retrieval.py

# Benchmark reranker improvement
python tests/test_reranker.py
```

### Project Structure

```
multi-model-rag-pipeline/
├── src/rag/                    # Core RAG library
│   ├── config.py               # Global config (paths, model names, hyperparams)
│   ├── models.py               # Element dataclass
│   ├── chunking/chunker.py     # Recursive text + visual chunker
│   ├── context/builder.py      # Evidence formatter
│   ├── embedding/visual_bge.py # Visualized-BGE embedder
│   ├── ingestion/              # Docling parsing pipeline
│   ├── llm/                    # Ollama LLM client + daemon manager
│   ├── prompt/builder.py       # RAG prompt builder
│   ├── query/planner.py        # Query decomposition planner
│   ├── reranking/              # CrossEncoder + BGE rerankers
│   └── vectorstore/qdrant.py   # Qdrant wrapper
├── ui/                         # PySide6 / QML desktop app
│   ├── main.py                 # App controller
│   ├── rag_worker.py           # Async QThread worker
│   └── qml/Main.qml           # QML frontend
├── scripts/ingest.py           # CLI ingestion script
├── tests/                      # Component + integration tests
├── documents/                  # Your documents go here (gitignored)
├── models/                     # Model weights go here (gitignored)
└── data/                       # Extracted artifacts (gitignored)
```

### Key Configuration

Edit [`src/rag/config.py`](src/rag/config.py) to customize:

```python
# Switch to CUDA when GPU is available
DEVICE = "cuda"           # or "cpu"

# Change the LLM model
# (must be pulled via: ollama pull <model>)
# Set in OllamaLLM constructor or rag_worker.py

# Adjust chunking
CHUNK_MAX_TOKENS = 512    # Max tokens per text chunk
# In chunker.py:
# max_chars = 1500        # Max characters per chunk
# overlap = 200           # Overlap between consecutive chunks
```

---

## Hardware Requirements

| Component | Minimum | Recommended |
|:---|:---|:---|
| **RAM** | 8 GB | 16 GB+ |
| **Storage** | 10 GB free | 20 GB+ free |
| **CPU** | Any modern x64 | 8+ cores |
| **GPU** | Not required | NVIDIA GPU w/ CUDA for 5–10x speedup |
| **Python** | 3.11 | 3.11 or 3.12 |

> On CPU, expect ~3–6 seconds per query (1.5s query planning + 0.3s vector search + 2–4s LLM streaming).
> On GPU (CUDA), query latency drops to under 2 seconds.

---

## Troubleshooting

**Q: App crashes with `TORCH_COMPILE` errors on Windows**

This is a known Windows + PyTorch CPU issue. The codebase already includes the fix, but if you write new scripts ensure you set:
```python
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CPP"] = "1"
```
...before importing `torch`.

**Q: Ollama model not found**

Pull the model first: `ollama pull qwen3.5:4b-q4_K_M`

**Q: `Visualized_m3.pth` not found**

Download it from HuggingFace BAAI/bge-visualized and place at `models/Visualized_m3.pth`.

**Q: Qdrant collection error on first run**

The collection is created automatically on first document ingestion. No manual setup needed.

---

## Roadmap

- [ ] Wire Cross-Encoder reranker into the desktop UI worker
- [ ] Hybrid BM25 sparse + dense retrieval (Qdrant Sparse Vectors)
- [ ] Multi-turn conversation memory with sliding window buffer
- [ ] VLM integration for rich figure understanding (llava, qwen2.5-vl)
- [ ] Hash-based incremental re-indexing (skip already-indexed docs)
- [ ] Batch embedding during ingestion (2–3× throughput improvement)
- [ ] Graph RAG layer with entity extraction and knowledge graph
- [ ] FastAPI backend mode for web/API access

See [`docs/ROADMAP_AND_IMPROVEMENTS.md`](docs/ROADMAP_AND_IMPROVEMENTS.md) for detailed plans.
See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full architectural diagrams.

---

## License

MIT License — see [`LICENSE`](LICENSE) for details.

---

## Acknowledgements

- [**IBM Docling**](https://github.com/docling-project/docling) — Document parsing and layout analysis
- [**BAAI Visualized-BGE**](https://huggingface.co/BAAI/bge-visualized) — Cross-modal text + image embeddings
- [**Qdrant**](https://qdrant.tech) — Local vector search engine
- [**Ollama**](https://ollama.com) — Local LLM runtime
- [**FlagEmbedding**](https://github.com/FlagOpen/FlagEmbedding) — BGE reranker library

---

<p align="center">
  Built with ❤️ for local-first AI — no cloud, no keys, just your documents and your machine.
</p>
