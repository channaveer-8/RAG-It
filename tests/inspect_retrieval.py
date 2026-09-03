import os

# ============================================================
# CPU / Windows Torch compatibility
# ============================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CPP"] = "1"


import sys
from pathlib import Path


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

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


# ============================================================
# Queries to inspect
# ============================================================

QUERIES = [

    "electric vehicle battery",

    "electric vehicle motor",

    "vehicle dashboard",

    "printed circuit board with components",

    "electronic differential",

    "battery voltage",
]


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)
    print("RETRIEVAL INSPECTION")
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
        f"\nVectors in collection: "
        f"{vectorstore.count()}"
    )


    # ========================================================
    # Inspect each query
    # ========================================================

    for query_index, query in enumerate(
        QUERIES,
        start=1,
    ):

        print(
            "\n" + "=" * 70
        )

        print(
            f"QUERY {query_index}: "
            f"{query}"
        )

        print(
            "=" * 70
        )


        # ----------------------------------------------------
        # Encode query
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # Retrieve top 10
        # ----------------------------------------------------

        results = vectorstore.search(

            vector=vector,

            limit=10,
        )


        # ----------------------------------------------------
        # Display results
        # ----------------------------------------------------

        for rank, result in enumerate(
            results,
            start=1,
        ):

            payload = (
                result.payload
                or {}
            )

            chunk_id = (
                payload.get(
                    "chunk_id",
                    "N/A",
                )
            )

            chunk_type = (
                payload.get(
                    "chunk_type",
                    "N/A",
                )
            )

            score = (
                result.score
            )

            page_numbers = (
                payload.get(
                    "page_numbers",
                    [],
                )
            )

            element_ids = (
                payload.get(
                    "element_ids",
                    [],
                )
            )

            asset_path = (
                payload.get(
                    "asset_path"
                )
            )

            caption = (
                payload.get(
                    "caption"
                )
            )

            content = (
                payload.get(
                    "content"
                )
            )


            print(
                "\n" + "-" * 70
            )

            print(
                f"Rank       : {rank}"
            )

            print(
                f"Chunk ID   : {chunk_id}"
            )

            print(
                f"Type       : {chunk_type}"
            )

            print(
                f"Score      : {score:.4f}"
            )

            print(
                f"Pages      : {page_numbers}"
            )

            print(
                f"Elements   : {element_ids}"
            )


            if asset_path:

                print(
                    f"Asset      : "
                    f"{asset_path}"
                )


            if caption:

                print(
                    f"Caption    : "
                    f"{caption}"
                )


            # ------------------------------------------------
            # Content preview
            # ------------------------------------------------

            if content:

                content_text = str(
                    content
                )

                content_text = (
                    content_text
                    .replace(
                        "\n",
                        " ",
                    )
                    .strip()
                )


                # Keep inspection output readable
                if len(
                    content_text
                ) > 500:

                    content_text = (
                        content_text[:500]
                        + "..."
                    )


                print(
                    f"Content    : "
                    f"{content_text}"
                )


    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVAL INSPECTION COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()