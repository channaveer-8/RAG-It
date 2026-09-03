import json
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


from rag.models import Element
from rag.chunking.chunker import (
    DocumentChunker,
)


def load_elements():

    metadata_files = list(
        (
            PROJECT_ROOT
            / "data"
            / "metadata"
        ).glob("*.json")
    )

    if not metadata_files:
        raise RuntimeError(
            "No metadata JSON files found."
        )

    path = metadata_files[0]

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        data = json.load(file)

    return [
        Element(**element)
        for element in data["elements"]
    ]


def main():

    print("=" * 70)
    print("CHUNKING TEST")
    print("=" * 70)

    elements = load_elements()

    print(
        f"\nInput elements: "
        f"{len(elements)}"
    )

    chunker = DocumentChunker(
    max_chars=1500,
    overlap=200,
    strategy="recursive",
)

    chunks = chunker.chunk(
        elements
    )

    print(
        f"Output chunks: "
        f"{len(chunks)}"
    )

    print("\nChunk types:")

    counts = {}

    for chunk in chunks:

        counts[chunk.chunk_type] = (
            counts.get(
                chunk.chunk_type,
                0,
            ) + 1
        )

    for key, value in counts.items():

        print(
            f"  {key:<10}: {value}"
        )

    print("\nFirst 10 chunks:")

    for index, chunk in enumerate(
        chunks[:10],
        start=1,
    ):

        print("\n" + "-" * 70)

        print(
            f"Chunk {index}"
        )

        print(
            f"ID      : {chunk.chunk_id}"
        )

        print(
            f"Type    : {chunk.chunk_type}"
        )

        print(
            f"Pages   : {chunk.page_numbers}"
        )

        print(
            f"Elements: {len(chunk.element_ids)}"
        )

        if chunk.content:

            preview = (
                chunk.content[:300]
                .replace("\n", " ")
            )

            print(
                f"Content : {preview}"
            )

        if chunk.asset_path:

            print(
                f"Asset   : "
                f"{chunk.asset_path}"
            )

    print("\n" + "=" * 70)
    print("CHUNKING TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()