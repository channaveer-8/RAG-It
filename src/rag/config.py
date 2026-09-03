from pathlib import Path


# ============================================================
# Project Root
# ============================================================

# config.py
# D:\Project-AI\multi-model-rag-pipeline\src\rag\config.py
#
# parents[0] = rag
# parents[1] = src
# parents[2] = multi-model-rag-pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# Input Documents
# ============================================================

DOCUMENTS_DIR = PROJECT_ROOT / "documents"


# ============================================================
# Data
# ============================================================

DATA_DIR = PROJECT_ROOT / "data"

PARSED_DIR = DATA_DIR / "parsed"
IMAGES_DIR = DATA_DIR / "images"
TABLES_DIR = DATA_DIR / "tables"
METADATA_DIR = DATA_DIR / "metadata"
CHUNKS_DIR = DATA_DIR / "chunks"


# ============================================================
# Models
# ============================================================

MODELS_DIR = PROJECT_ROOT / "models"

# ------------------------------------------------------------
# Visualized-BGE / BGE-M3
# ------------------------------------------------------------

BGE_MODEL_NAME = "BAAI/bge-m3"

BGE_MODEL_WEIGHT = (
    MODELS_DIR / "Visualized_m3.pth"
)

EMBEDDING_DIMENSION = 1024

DEVICE = "cpu"


# ============================================================
# Reranker
# ============================================================

RERANKER_MODEL_NAME = (
    "cross-encoder/ms-marco-MiniLM-L6-v2"
)

# Number of dense-retrieval candidates
# sent to the reranker.
RERANKER_TOP_K = 20

# Number of final candidates after reranking.
RERANKER_FINAL_K = 5

# Maximum query + passage sequence length.
RERANKER_MAX_LENGTH = 512


# ============================================================
# Vector Database
# ============================================================

QDRANT_DIR = PROJECT_ROOT / "qdrant_data"

COLLECTION_NAME = (
    "multimodal_documents"
)


# ============================================================
# Chunking
# ============================================================

CHUNK_MAX_TOKENS = 512


# ============================================================
# Supported Documents
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".html",
    ".md",
    ".txt",
}