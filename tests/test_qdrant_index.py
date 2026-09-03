import os

# ============================================================
# CPU / Windows Torch compatibility
# ============================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CPP"] = "1"


import sys
import json
from pathlib import Path

from qdrant_client.models import PointStruct


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


from rag.config import (
    CHUNKS_DIR,
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


def load_chunks():

    chunks = []

    for path in sorted(
        CHUNKS_DIR.glob("*.json")
    ):

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        chunks.extend(
            data["chunks"]
        )

    return chunks


def main():

    print("=" * 70)
    print("MULTIMODAL RAG - QDRANT INDEXING")
    print("=" * 70)


    # ========================================================
    # Load chunks
    # ========================================================

    chunks = load_chunks()

    print(
        f"\nChunks found: "
        f"{len(chunks)}"
    )


    # ========================================================
    # Load embedding model
    # ========================================================

    embedder = VisualBGEEmbedder(

        model_name=(
            BGE_MODEL_NAME
        ),

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


    vectorstore.create_collection(
        recreate=True
    )


    # ========================================================
    # Generate embeddings
    # ========================================================

    points = []

    print(
        "\nEncoding chunks..."
    )


    for index, chunk in enumerate(
        chunks
    ):

        chunk_type = (
            chunk["chunk_type"]
        )

        print(
            f"  [{index + 1}/"
            f"{len(chunks)}] "
            f"{chunk_type:<6} "
            f"{chunk['chunk_id']}"
        )


        # ----------------------------------------------------
        # Text
        # ----------------------------------------------------

        if chunk_type == "text":

            content = (
                chunk.get("content")
                or ""
            )

            if not content.strip():

                print(
                    "      Skipping empty text"
                )

                continue

            embedding = (
                embedder.encode_text(
                    content
                )
            )


        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        elif chunk_type == "image":

            asset_path = (
                chunk.get("asset_path")
            )

            if not asset_path:

                print(
                    "      Skipping image "
                    "without asset"
                )

                continue

            embedding = (
                embedder.encode_image(
                    asset_path
                )
            )


        # ----------------------------------------------------
        # Table
        # ----------------------------------------------------

        elif chunk_type == "table":

            content = (
                chunk.get("content")
                or ""
            )

            if not content.strip():

                print(
                    "      Skipping empty table"
                )

                continue

            embedding = (
                embedder.encode_text(
                    content
                )
            )


        else:

            print(
                f"      Unsupported type: "
                f"{chunk_type}"
            )

            continue


        # ----------------------------------------------------
        # Convert tensor → list
        # ----------------------------------------------------

        vector = (
            embedder.to_numpy(
                embedding
            )
            .flatten()
            .tolist()
        )


        # ----------------------------------------------------
        # Payload
        # ----------------------------------------------------

        payload = {
            "document_id": (
                chunk["document_id"]
            ),

            "document_name": (
                chunk["document_name"]
            ),

            "chunk_id": (
                chunk["chunk_id"]
            ),

            "chunk_type": (
                chunk["chunk_type"]
            ),

            "content": (
                chunk.get("content")
            ),

            "element_ids": (
                chunk.get(
                    "element_ids",
                    [],
                )
            ),

            "source_refs": (
                chunk.get(
                    "source_refs",
                    [],
                )
            ),

            "page_numbers": (
                chunk.get(
                    "page_numbers",
                    [],
                )
            ),

            "asset_path": (
                chunk.get(
                    "asset_path"
                )
            ),

            "caption": (
                chunk.get(
                    "caption"
                )
            ),

            "metadata": (
                chunk.get(
                    "metadata",
                    {},
                )
            ),
        }


        points.append(
            PointStruct(

                id=index,

                vector=vector,

                payload=payload,
            )
        )


    # ========================================================
    # Upload
    # ========================================================

    print(
        "\nUploading vectors to Qdrant..."
    )

    vectorstore.upsert(
        points
    )


    # ========================================================
    # Verify
    # ========================================================

    count = (
        vectorstore.count()
    )

    print(
        f"\nVectors in collection: "
        f"{count}"
    )


    # ========================================================
    # Test query
    # ========================================================

    query = (
        "electric vehicle battery"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"QUERY: {query}"
    )

    print(
        "=" * 70
    )


    query_embedding = (
        embedder.encode_text(
            query
        )
    )

    query_vector = (
        embedder.to_numpy(
            query_embedding
        )
        .flatten()
        .tolist()
    )


    results = vectorstore.search(
        vector=query_vector,
        limit=5,
    )


    print(
        "\nResults:"
    )


    for rank, result in enumerate(
        results,
        start=1,
    ):

        payload = (
            result.payload
        )

        print(
            f"\n{rank}. "
            f"{payload['chunk_id']}"
        )

        print(
            f"   Type : "
            f"{payload['chunk_type']}"
        )

        print(
            f"   Page : "
            f"{payload['page_numbers']}"
        )

        print(
            f"   Score: "
            f"{result.score:.4f}"
        )

        if payload.get(
            "content"
        ):

            preview = (
                payload["content"]
                [:200]
                .replace(
                    "\n",
                    " ",
                )
            )

            print(
                f"   Text : "
                f"{preview}"
            )


    print(
        "\n" + "=" * 70
    )

    print(
        "QDRANT INDEXING TEST COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":
    main()