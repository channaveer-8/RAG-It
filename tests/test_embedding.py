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
    BGE_MODEL_NAME,
    BGE_MODEL_WEIGHT,
    DEVICE,
)

from rag.embedding.visual_bge import (
    VisualBGEEmbedder,
)


def main():

    print("=" * 70)
    print("VISUALIZED-BGE EMBEDDING TEST")
    print("=" * 70)

    embedder = VisualBGEEmbedder(
    model_name=BGE_MODEL_NAME,
    model_weight=str(
        BGE_MODEL_WEIGHT
    ),
    device=DEVICE,
    )

    # --------------------------------------------------------
    # Text
    # --------------------------------------------------------

    text_embedding = (
        embedder.encode_text(
            "Electric vehicle battery"
        )
    )

    print(
        "\nText embedding:"
    )

    print(
        f"  Shape: "
        f"{text_embedding.shape}"
    )

    print(
        f"  Device: "
        f"{text_embedding.device}"
    )

    # --------------------------------------------------------
    # Image
    # --------------------------------------------------------

    image_files = list(
        (
            PROJECT_ROOT
            / "data"
            / "images"
        ).glob(
            "*/*.png"
        )
    )

    if not image_files:

        raise RuntimeError(
            "No extracted images found."
        )

    image_path = image_files[0]

    image_embedding = (
        embedder.encode_image(
            str(image_path)
        )
    )

    print(
        "\nImage embedding:"
    )

    print(
        f"  Image: "
        f"{image_path.name}"
    )

    print(
        f"  Shape: "
        f"{image_embedding.shape}"
    )

    print(
        f"  Device: "
        f"{image_embedding.device}"
    )

    # --------------------------------------------------------
    # Multimodal
    # --------------------------------------------------------

    multimodal_embedding = (
        embedder.encode_multimodal(
            text="Electric vehicle",
            image_path=str(
                image_path
            ),
        )
    )

    print(
        "\nMultimodal embedding:"
    )

    print(
        f"  Shape: "
        f"{multimodal_embedding.shape}"
    )

    # --------------------------------------------------------
    # Similarity
    # --------------------------------------------------------

    similarity = torch_cosine_similarity(
        text_embedding,
        image_embedding,
    )

    print(
        "\nText <-> Image similarity:"
    )

    print(
        f"  {similarity:.6f}"
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "EMBEDDING TEST COMPLETE"
    )

    print(
        "=" * 70
    )


def torch_cosine_similarity(
    a,
    b,
):

    import torch

    return torch.nn.functional.cosine_similarity(
        a,
        b,
        dim=-1,
    ).item()


if __name__ == "__main__":
    main()