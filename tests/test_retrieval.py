import os

# ============================================================
# CPU / Windows Torch compatibility
# ============================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CPP"] = "1"


import sys
import json
from pathlib import Path


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

    if not EVALUATION_FILE.exists():

        raise FileNotFoundError(
            "Evaluation file not found:\n"
            f"{EVALUATION_FILE}"
        )

    with EVALUATION_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:

        dataset = json.load(file)


    if not isinstance(
        dataset,
        list,
    ):

        raise ValueError(
            "Evaluation dataset must "
            "be a JSON list."
        )


    for index, item in enumerate(
        dataset,
        start=1,
    ):

        if not isinstance(
            item,
            dict,
        ):

            raise ValueError(
                f"Evaluation item {index} "
                f"must be an object."
            )


        if "query" not in item:

            raise ValueError(
                f"Evaluation item {index} "
                f"is missing 'query'."
            )


        if "relevant_chunks" not in item:

            raise ValueError(
                f"Evaluation item {index} "
                f"is missing 'relevant_chunks'."
            )


        if not isinstance(
            item["relevant_chunks"],
            list,
        ):

            raise ValueError(
                f"'relevant_chunks' in "
                f"item {index} must be a list."
            )


    return dataset


# ============================================================
# Retrieval
# ============================================================

def retrieve(
    query: str,
    embedder: VisualBGEEmbedder,
    vectorstore: QdrantVectorStore,
    top_k: int = 5,
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
    k: int,
) -> bool:

    retrieved_ids = {

        result.payload.get(
            "chunk_id"
        )

        for result in results[:k]

        if result.payload
        and result.payload.get(
            "chunk_id"
        )
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
    print("DENSE RETRIEVAL EVALUATION")
    print("=" * 70)


    # ========================================================
    # Load evaluation dataset
    # ========================================================

    evaluation_dataset = (
        load_evaluation_dataset()
    )

    print(
        f"\nEvaluation queries: "
        f"{len(evaluation_dataset)}"
    )


    # ========================================================
    # Load Visualized-BGE
    # ========================================================

    print(
        "\nLoading Visualized-BGE..."
    )

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


    vector_count = (
        vectorstore.count()
    )

    print(
        f"\nVectors in collection: "
        f"{vector_count}"
    )


    if vector_count == 0:

        raise RuntimeError(
            "Qdrant collection is empty. "
            "Run the indexing test first."
        )


    # ========================================================
    # Evaluation counters
    # ========================================================

    hit1 = 0
    hit3 = 0
    hit5 = 0

    evaluated = 0
    skipped = 0


    # ========================================================
    # Evaluate queries
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
                f"\nSkipping [{index}]: "
                f"{query}"
            )

            print(
                "  No relevant chunk IDs defined."
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
            f"RELEVANT CHUNKS: "
            f"{relevant_chunks}"
        )

        print(
            "-" * 70
        )


        # ----------------------------------------------------
        # Retrieve top 5
        # ----------------------------------------------------

        results = retrieve(

            query=query,

            embedder=embedder,

            vectorstore=vectorstore,

            top_k=5,
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

            score = result.score


            is_relevant = (
                chunk_id
                in relevant_chunks
            )


            marker = (
                "✓"
                if is_relevant
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
                f"     Score : "
                f"{score:.4f}"
            )


        # ----------------------------------------------------
        # Calculate Hit@K
        # ----------------------------------------------------

        h1 = hit_at_k(
            results,
            relevant_chunks,
            1,
        )

        h3 = hit_at_k(
            results,
            relevant_chunks,
            3,
        )

        h5 = hit_at_k(
            results,
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


        # ----------------------------------------------------
        # Update counters
        # ----------------------------------------------------

        if h1:
            hit1 += 1

        if h3:
            hit3 += 1

        if h5:
            hit5 += 1


    # ========================================================
    # Final evaluation
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "RETRIEVAL EVALUATION"
    )

    print(
        "=" * 70
    )


    if evaluated == 0:

        print(
            "\nNo queries were evaluated."
        )

        print(
            "Add relevant chunk IDs to:"
        )

        print(
            f"  {EVALUATION_FILE}"
        )

        vectorstore.close()

        return


    # ========================================================
    # Metrics
    # ========================================================

    hit1_percent = (
        hit1
        / evaluated
        * 100
    )

    hit3_percent = (
        hit3
        / evaluated
        * 100
    )

    hit5_percent = (
        hit5
        / evaluated
        * 100
    )


    print(
        f"Hit@1 : "
        f"{hit1}/{evaluated} "
        f"({hit1_percent:.2f}%)"
    )

    print(
        f"Hit@3 : "
        f"{hit3}/{evaluated} "
        f"({hit3_percent:.2f}%)"
    )

    print(
        f"Hit@5 : "
        f"{hit5}/{evaluated} "
        f"({hit5_percent:.2f}%)"
    )


    print(
        f"\nEvaluated : "
        f"{evaluated}"
    )

    print(
        f"Skipped   : "
        f"{skipped}"
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "DENSE RETRIEVAL BASELINE COMPLETE"
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