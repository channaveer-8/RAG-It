import os

# ============================================================
# CPU / Windows Torch compatibility
# ============================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CPP"] = "1"


import sys
import json
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
    RERANKER_MODEL_NAME,
    RERANKER_TOP_K,
    RERANKER_FINAL_K,
    RERANKER_MAX_LENGTH,
)


# ============================================================
# Visualized-BGE
# ============================================================

from rag.embedding.visual_bge import (
    VisualBGEEmbedder,
)


# ============================================================
# Qdrant
# ============================================================

from rag.vectorstore.qdrant import (
    QdrantVectorStore,
)


# ============================================================
# Cross-Encoder Reranker
# ============================================================

from rag.reranking.cross_encoder_reranker import (
    CrossEncoderReranker,
)


# ============================================================
# Evaluation Dataset
# ============================================================

EVALUATION_FILE = (
    PROJECT_ROOT
    / "tests"
    / "evaluation"
    / "queries.json"
)


def load_evaluation_dataset():

    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        return json.load(file)


# ============================================================
# Dense Retrieval
# ============================================================

def retrieve(
    query,
    embedder,
    vectorstore,
    top_k,
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
# Hit@K
# ============================================================

def hit_at_k(
    results,
    relevant_chunks,
    k,
):

    retrieved_ids = {

        item["result"].payload.get(
            "chunk_id"
        )

        for item in results[:k]

        if item["result"].payload
    }


    return bool(
        retrieved_ids
        & set(relevant_chunks)
    )


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 70)

    print(
        "DENSE + CROSS-ENCODER RERANKER EVALUATION"
    )

    print("=" * 70)


    evaluation_dataset = (
        load_evaluation_dataset()
    )


    print(
        f"\nEvaluation queries: "
        f"{len(evaluation_dataset)}"
    )


    # ========================================================
    # Stage 1
    # Visualized-BGE
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
    # Qdrant
    # ========================================================

    print(
        "\nOpening Qdrant Local..."
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


    print(
        "Qdrant Local ready."
    )


    print(
        f"\nVectors in collection: "
        f"{vectorstore.count()}"
    )


    # ========================================================
    # Stage 2
    # Cross-Encoder Reranker
    # ========================================================

    print(
        "\nLoading reranker:"
    )

    print(
        f"  Model : {RERANKER_MODEL_NAME}"
    )

    print(
        f"  Device: {DEVICE}"
    )

    print(
        f"  Max length: "
        f"{RERANKER_MAX_LENGTH}"
    )


    reranker = CrossEncoderReranker(

        model_name=(
            RERANKER_MODEL_NAME
        ),

        device=DEVICE,

        max_length=(
            RERANKER_MAX_LENGTH
        ),
    )


    print(
        f"{RERANKER_MODEL_NAME} ready."
    )


    # ========================================================
    # Metrics
    # ========================================================

    hit1 = 0
    hit3 = 0
    hit5 = 0

    evaluated = 0
    skipped = 0


    # ========================================================
    # Evaluation
    # ========================================================

    for index, item in enumerate(
        evaluation_dataset,
        start=1,
    ):

        query = item["query"]

        relevant_chunks = (
            item["relevant_chunks"]
        )


        # ----------------------------------------------------
        # Skip queries without ground truth
        # ----------------------------------------------------

        if not relevant_chunks:

            skipped += 1

            print(
                f"\nSkipping: {query}"
            )

            print(
                "  No expected chunk IDs defined."
            )

            continue


        evaluated += 1


        print(
            "\n" + "-" * 70
        )

        print(
            f"QUERY [{index}]: "
            f"{query}"
        )

        print(
            "-" * 70
        )


        print(
            f"Expected relevant chunks: "
            f"{len(relevant_chunks)}"
        )


        # ====================================================
        # Stage 1
        # Dense Retrieval
        # ====================================================

        candidates = retrieve(

            query=query,

            embedder=embedder,

            vectorstore=vectorstore,

            top_k=RERANKER_TOP_K,
        )


        print(
            f"\nStage 1: "
            f"{len(candidates)} candidates"
        )


        # ====================================================
        # Stage 2
        # Cross-Encoder Reranking
        # ====================================================

        print(
            f"Stage 2: "
            f"{RERANKER_MODEL_NAME} scoring..."
        )


        reranked = reranker.rerank(

            query=query,

            results=candidates,
        )


        print(
            "Stage 2 complete."
        )


        print(
            f"Final results: "
            f"Top {RERANKER_FINAL_K}"
        )


        # ====================================================
        # Display final results
        # ====================================================

        for rank, item_result in enumerate(

            reranked[
                :RERANKER_FINAL_K
            ],

            start=1,
        ):

            result = (
                item_result["result"]
            )


            reranker_score = (
                item_result[
                    "reranker_score"
                ]
            )


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


            page_numbers = (
                payload.get(
                    "page_numbers",
                    payload.get(
                        "pages",
                        [],
                    ),
                )
            )


            relevant = (
                chunk_id
                in relevant_chunks
            )


            marker = (
                "✓"
                if relevant
                else " "
            )


            print(
                f"{marker} {rank}. "
                f"{chunk_id}"
            )


            print(
                f"     Type  : "
                f"{chunk_type}"
            )


            print(
                f"     Pages : "
                f"{page_numbers}"
            )


            print(
                f"     Score : "
                f"{reranker_score:.4f}"
            )


        # ====================================================
        # Metrics
        # ====================================================

        h1 = hit_at_k(
            reranked,
            relevant_chunks,
            1,
        )


        h3 = hit_at_k(
            reranked,
            relevant_chunks,
            3,
        )


        h5 = hit_at_k(
            reranked,
            relevant_chunks,
            5,
        )


        print(
            "\nHits:"
        )


        print(
            f"  Hit@1 : "
            f"{'YES' if h1 else 'NO'}"
        )


        print(
            f"  Hit@3 : "
            f"{'YES' if h3 else 'NO'}"
        )


        print(
            f"  Hit@5 : "
            f"{'YES' if h5 else 'NO'}"
        )


        if h1:
            hit1 += 1

        if h3:
            hit3 += 1

        if h5:
            hit5 += 1


    # ========================================================
    # Final Metrics
    # ========================================================

    print(
        "\n" + "=" * 70
    )


    print(
        "DENSE + CROSS-ENCODER RERANKER EVALUATION"
    )


    print(
        "=" * 70
    )


    if evaluated == 0:

        print(
            "\nNo evaluation queries have "
            "ground-truth chunk IDs."
        )

        print(
            f"\nEvaluated : {evaluated}"
        )

        print(
            f"Skipped   : {skipped}"
        )

        print(
            "=" * 70
        )

        vectorstore.close()

        return


    print(
        f"Hit@1 : "
        f"{hit1}/{evaluated} "
        f"({hit1 / evaluated * 100:.2f}%)"
    )


    print(
        f"Hit@3 : "
        f"{hit3}/{evaluated} "
        f"({hit3 / evaluated * 100:.2f}%)"
    )


    print(
        f"Hit@5 : "
        f"{hit5}/{evaluated} "
        f"({hit5 / evaluated * 100:.2f}%)"
    )


    print()


    print(
        f"Evaluated : {evaluated}"
    )


    print(
        f"Skipped   : {skipped}"
    )


    print(
        "\n" + "=" * 70
    )


    print(
        "CROSS-ENCODER RERANKER EVALUATION COMPLETE"
    )


    print(
        "=" * 70
    )


    vectorstore.close()


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    main()