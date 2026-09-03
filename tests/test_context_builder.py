import sys
from pathlib import Path


# ============================================================
# Project Root
# ============================================================

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
    EMBEDDING_DIMENSION,
    BGE_MODEL_NAME,
    BGE_MODEL_WEIGHT,
    DEVICE,
)

from rag.embedding.visual_bge import (
    VisualBGEEmbedder,
)

from rag.vectorstore.qdrant import (
    QdrantVectorStore,
)

from rag.context import (
    ContextBuilder,
)


def main():

    print("=" * 70)
    print("CONTEXT BUILDER TEST")
    print("=" * 70)


    # ========================================================
    # Load embedding model
    # ========================================================

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


    # ========================================================
    # Open Qdrant
    # ========================================================

    print(
        "\nOpening Qdrant..."
    )

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


    # ========================================================
    # Query
    # ========================================================

    query = (
        "electric vehicle battery"
    )


    print(
        f"\nQuery:"
    )

    print(
        f"  {query}"
    )


    # ========================================================
    # Retrieve
    # ========================================================

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


    results = vectorstore.search(

        vector=vector,

        limit=5,
    )


    print(
        f"\nRetrieved chunks: "
        f"{len(results)}"
    )


    # ========================================================
    # Build context
    # ========================================================

    builder = ContextBuilder(
        max_chunks=5,
    )


    context = builder.build(
        results
    )


    # ========================================================
    # Display
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ASSEMBLED CONTEXT"
    )

    print(
        "=" * 70
    )

    print()

    print(
        context
    )


    # ========================================================
    # Complete
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CONTEXT BUILDER TEST COMPLETE"
    )

    print(
        "=" * 70
    )


    vectorstore.close()


if __name__ == "__main__":

    main()