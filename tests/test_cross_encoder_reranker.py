import os

# ============================================================
# CPU / Windows Torch compatibility
# ============================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CPP"] = "1"


import sys
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
    RERANKER_MODEL_NAME,
    RERANKER_MAX_LENGTH,
    DEVICE,
)


# ============================================================
# Cross-Encoder Reranker
# ============================================================

from rag.reranking.cross_encoder_reranker import (
    CrossEncoderReranker,
)


# ============================================================
# Test
# ============================================================

def main():

    print("=" * 70)
    print("CROSS-ENCODER RERANKER TEST")
    print("=" * 70)


    # ========================================================
    # Configuration
    # ========================================================

    print(
        "\nModel:"
    )

    print(
        f"  {RERANKER_MODEL_NAME}"
    )


    print(
        "\nDevice:"
    )

    print(
        f"  {DEVICE}"
    )


    print(
        "\nMax length:"
    )

    print(
        f"  {RERANKER_MAX_LENGTH}"
    )


    # ========================================================
    # Load reranker
    # ========================================================

    print(
        "\nLoading reranker..."
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
    # Test query
    # ========================================================

    query = (
        "electric vehicle battery"
    )


    # ========================================================
    # Test candidates
    # ========================================================

    candidates = [

        (
            "The battery nominal "
            "voltage is 525.4 V and "
            "capacity is 296 Ah."
        ),

        (
            "The vehicle uses four "
            "independent in-wheel "
            "motors."
        ),

        (
            "The system implements "
            "an electronic differential."
        ),

    ]


    # ========================================================
    # Display query
    # ========================================================

    print(
        "\nQuery:"
    )

    print(
        f"  {query}"
    )


    # ========================================================
    # Display candidates
    # ========================================================

    print(
        "\nCandidates:"
    )


    for index, candidate in enumerate(
        candidates,
        start=1,
    ):

        print(
            f"  {index}. "
            f"{candidate}"
        )


    # ========================================================
    # Build query-document pairs
    # ========================================================

    pairs = [

        (
            query,
            candidate,
        )

        for candidate in candidates

    ]


    # ========================================================
    # Score candidates
    # ========================================================

    print(
        "\nScoring candidates..."
    )


    scores = reranker.model.predict(

        pairs,

        show_progress_bar=True,

    )


    # ========================================================
    # Rank candidates
    # ========================================================

    ranked = sorted(

        zip(
            candidates,
            scores,
        ),

        key=lambda item:
            float(item[1]),

        reverse=True,
    )


    # ========================================================
    # Results
    # ========================================================

    print(
        "\nResults:"
    )


    for rank, (
        candidate,
        score,
    ) in enumerate(
        ranked,
        start=1,
    ):

        print(
            f"\n{rank}. "
            f"Score: "
            f"{float(score):.4f}"
        )

        print(
            f"   {candidate}"
        )


    # ========================================================
    # Complete
    # ========================================================

    print(
        "\n" + "=" * 70
    )


    print(
        f"{RERANKER_MODEL_NAME} TEST COMPLETE"
    )


    print(
        "=" * 70
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    main()