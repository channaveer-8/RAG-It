import os
import sys
from pathlib import Path

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CPP"] = "1"

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)

from rag.config import (
    QDRANT_DIR,
    COLLECTION_NAME,
    BGE_MODEL_NAME,
    BGE_MODEL_WEIGHT,
    EMBEDDING_DIMENSION,
    DEVICE,
)

from rag.embedding.visual_bge import (
    VisualBGEEmbedder,
)

from rag.vectorstore.qdrant import (
    QdrantVectorStore,
)

from rag.ingestion.indexer import (
    DocumentIndexer,
)


DOCUMENT = (
    PROJECT_ROOT
    / "documents"
    / "THE CONSTITUTION OF INDIA.pdf"
)


def progress(message):

    print(
        f"\n>>> {message}",
        flush=True,
    )


def main():

    print("=" * 70)
    print("SINGLE DOCUMENT INDEX TEST")
    print("=" * 70)

    print(
        f"\nDocument:\n{DOCUMENT}"
    )

    if not DOCUMENT.exists():

        raise FileNotFoundError(
            DOCUMENT
        )

    print(
        "\nLoading Visualized-BGE..."
    )

    embedder = VisualBGEEmbedder(

        model_name=BGE_MODEL_NAME,

        model_weight=str(
            BGE_MODEL_WEIGHT
        ),

        device=DEVICE,
    )

    print(
        "Visualized-BGE ready."
    )

    vectorstore = (
        QdrantVectorStore(

            path=str(
                QDRANT_DIR
            ),

            collection_name=(
                COLLECTION_NAME
            ),

            vector_size=(
                EMBEDDING_DIMENSION
            ),
        )
    )

    indexer = DocumentIndexer(
        progress_callback=progress
    )

    print(
        "\nStarting indexing..."
    )

    stats = indexer.index_document(

        source=DOCUMENT,

        embedder=embedder,

        vectorstore=vectorstore,
    )

    print(
        "\nRESULT:"
    )

    print(
        stats
    )

    vectorstore.close()


if __name__ == "__main__":
    main()