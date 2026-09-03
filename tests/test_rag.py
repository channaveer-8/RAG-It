import os

# ============================================================
# CPU / Windows Torch compatibility
# ============================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CPP"] = "1"


import sys
import time
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


# ============================================================
# Configuration
# ============================================================

from rag.config import (
    QDRANT_DIR,
    COLLECTION_NAME,
    BGE_MODEL_NAME,
    BGE_MODEL_WEIGHT,
    EMBEDDING_DIMENSION,
    DEVICE,
)


# ============================================================
# Components
# ============================================================

from rag.embedding.visual_bge import (
    VisualBGEEmbedder,
)

from rag.vectorstore.qdrant import (
    QdrantVectorStore,
)

from rag.context import (
    ContextBuilder,
)

from rag.llm import (
    OllamaLLM,
)


# ============================================================
# Runtime Configuration
# ============================================================

TOP_K = 5

OLLAMA_MODEL = (
    "qwen3.5:4b-q4_K_M"
)


# ============================================================
# Dense Retrieval
# ============================================================

def retrieve(
    query: str,
    embedder,
    vectorstore,
    top_k: int,
):

    embedding = (
        embedder.encode_text(
            query
        )
    )

    vector = (
        embedder.to_numpy(
            embedding
        )
        .flatten()
        .tolist()
    )

    return vectorstore.search(
        vector=vector,
        limit=top_k,
    )


# ============================================================
# Display Retrieved Results
# ============================================================

def display_results(
    results,
):

    print(
        "\n" + "-" * 70
    )

    print(
        "RETRIEVED CHUNKS"
    )

    print(
        "-" * 70
    )


    for rank, result in enumerate(
        results,
        start=1,
    ):

        payload = (
            result.payload
            or {}
        )


        chunk_id = payload.get(
            "chunk_id",
            "N/A",
        )


        chunk_type = payload.get(
            "chunk_type",
            "N/A",
        )


        pages = payload.get(
            "pages",
            payload.get(
                "page_numbers",
                [],
            ),
        )


        print(
            f"{rank}. {chunk_id}"
        )

        print(
            f"   Type : {chunk_type}"
        )

        print(
            f"   Pages: {pages}"
        )

        print(
            f"   Score: {result.score:.4f}"
        )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "INTERACTIVE RAG"
    )

    print("=" * 70)


    # ========================================================
    # Load Visualized-BGE ONCE
    # ========================================================

    print(
        "\nLoading Visualized-BGE..."
    )

    bge_start = time.perf_counter()


    embedder = VisualBGEEmbedder(

        model_name=BGE_MODEL_NAME,

        model_weight=str(
            BGE_MODEL_WEIGHT
        ),

        device=DEVICE,
    )


    print(
        f"Visualized-BGE ready "
        f"({time.perf_counter() - bge_start:.2f}s)"
    )


    # ========================================================
    # Connect Qdrant ONCE
    # ========================================================

    print(
        "\nOpening Qdrant..."
    )


    qdrant_start = time.perf_counter()


    vectorstore = QdrantVectorStore(

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


    print(
        f"Qdrant ready "
        f"({time.perf_counter() - qdrant_start:.2f}s)"
    )


    print(
        f"Vectors: "
        f"{vectorstore.count()}"
    )


    # ========================================================
    # Connect Ollama ONCE
    # ========================================================

    print(
        "\nConnecting to Ollama..."
    )


    llm = OllamaLLM(

        model_name=(
            OLLAMA_MODEL
        ),

        base_url=(
            "http://localhost:11434"
        ),

        temperature=0.2,
    )


    print(
        f"Ollama ready"
    )

    print(
        f"Model: {OLLAMA_MODEL}"
    )


    # ========================================================
    # Context Builder
    # ========================================================

    context_builder = ContextBuilder(
        max_chunks=TOP_K,
    )


    # ========================================================
    # Runtime Ready
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RAG READY"
    )

    print(
        "=" * 70
    )

    print(
        "\nAsk questions continuously."
    )

    print(
        "Press Ctrl+C to stop."
    )


    # ========================================================
    # Interactive Loop
    # ========================================================

    try:

        while True:

            print(
                "\n" + "-" * 70
            )


            query = input(
                "\nQuestion: "
            ).strip()


            if not query:

                continue


            # =================================================
            # Stage 1: Query Embedding + Retrieval
            # =================================================

            print(
                "\nRetrieving..."
            )


            retrieval_start = (
                time.perf_counter()
            )


            results = retrieve(

                query=query,

                embedder=embedder,

                vectorstore=vectorstore,

                top_k=TOP_K,
            )


            retrieval_time = (
                time.perf_counter()
                - retrieval_start
            )


            print(
                f"Retrieved "
                f"{len(results)} chunks "
                f"in "
                f"{retrieval_time:.2f}s"
            )


            display_results(
                results
            )


            # =================================================
            # Stage 2: Context Assembly
            # =================================================

            print(
                "\nBuilding context..."
            )


            context_start = (
                time.perf_counter()
            )


            context = (
                context_builder.build(
                    results
                )
            )


            context_time = (
                time.perf_counter()
                - context_start
            )


            print(
                f"Context ready "
                f"({len(context)} characters, "
                f"{context_time:.2f}s)"
            )


            # =================================================
            # Stage 3: LLM
            # =================================================

            print(
                "\nAnswer:"
            )

            print(
                "-" * 70
            )


            llm_start = (
                time.perf_counter()
            )


            llm.generate(

                query=query,

                context=context,

                stream=True,
            )


            llm_time = (
                time.perf_counter()
                - llm_start
            )


            print(
                "-" * 70
            )


            print(
                f"Generation time: "
                f"{llm_time:.2f}s"
            )


    except KeyboardInterrupt:

        print(
            "\n\nStopping RAG..."
        )


    finally:

        # ====================================================
        # Cleanup ONCE
        # ====================================================

        print(
            "Closing Qdrant..."
        )


        vectorstore.close()


        print(
            "RAG stopped."
        )


        print(
            "=" * 70
        )


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()