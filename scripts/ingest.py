import os

# ============================================================
# CPU / Windows Torch compatibility
# ============================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CPP"] = "1"


import sys
import json
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
    METADATA_DIR,
    CHUNKS_DIR,
)

from rag.ingestion.scanner import (
    scan_documents,
)

from rag.ingestion.parser import (
    DoclingParser,
)

from rag.ingestion.extractor import (
    DocumentExtractor,
)

from rag.chunking.chunker import (
    DocumentChunker,
)


def main():

    print("=" * 70)
    print("MULTIMODAL RAG - DOCUMENT INGESTION")
    print("=" * 70)

    documents = scan_documents()

    print(
        f"\nDocuments found: "
        f"{len(documents)}"
    )

    if not documents:

        print(
            "\nNo documents found."
        )

        return

    parser = DoclingParser()

    extractor = DocumentExtractor()

    chunker = DocumentChunker(
        max_chars=1500,
        overlap=200,
        strategy="recursive",
    )


    # ========================================================
    # Process documents
    # ========================================================

    for source in documents:

        print("\n" + "-" * 70)

        print(
            f"Processing: {source.name}"
        )

        print("-" * 70)


        # ====================================================
        # 1. Parse with Docling
        # ====================================================

        print(
            "\n[1/4] Parsing with Docling..."
        )

        document = parser.parse(
            source
        )

        print(
            "      Done."
        )


        # ====================================================
        # 2. Extract normalized elements
        # ====================================================

        print(
            "\n[2/4] Extracting elements..."
        )

        elements = extractor.extract(
            document=document,
            source_path=source,
        )

        print(
            "      Done."
        )


        # ====================================================
        # Document ID
        # ====================================================

        document_id = (
            elements[0].document_id
            if elements
            else source.stem
        )


        # ====================================================
        # Save normalized metadata
        # ====================================================

        output_path = (
            METADATA_DIR /
            f"{document_id}.json"
        )

        metadata = {
            "document_id": document_id,

            "document_name": (
                source.name
            ),

            "parser": "docling",

            "element_count": (
                len(elements)
            ),

            "elements": [
                element.to_dict()
                for element in elements
            ],
        }


        with output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )


        print(
            f"\n      Metadata saved:"
        )

        print(
            f"      {output_path}"
        )


        # ====================================================
        # 3. Chunk document
        # ====================================================

        print(
            "\n[3/4] Creating chunks..."
        )

        chunks = chunker.chunk(
            elements
        )

        print(
            f"      Created "
            f"{len(chunks)} chunks."
        )


        # ====================================================
        # Save chunk metadata
        # ====================================================

        chunk_output_path = (
            CHUNKS_DIR /
            f"{document_id}.json"
        )

        chunk_data = {

            "document_id": (
                document_id
            ),

            "document_name": (
                source.name
            ),

            "chunk_count": (
                len(chunks)
            ),

            "chunking": {

                "strategy": (
                    "recursive"
                ),

                "max_chars": 1500,

                "overlap": 200,
            },

            "chunks": [
                chunk.to_dict()
                for chunk in chunks
            ],
        }


        with chunk_output_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                chunk_data,
                file,
                indent=2,
                ensure_ascii=False,
            )


        print(
            "\n      Chunks saved:"
        )

        print(
            f"      {chunk_output_path}"
        )


        # ====================================================
        # 4. Statistics
        # ====================================================

        print(
            "\n[4/4] Statistics"
        )


        # ----------------------------------------------------
        # Element statistics
        # ----------------------------------------------------

        element_counts = {}

        for element in elements:

            element_type = (
                element.element_type
            )

            element_counts[
                element_type
            ] = (
                element_counts.get(
                    element_type,
                    0,
                ) + 1
            )


        print(
            "\n      Elements:"
        )

        for element_type, count in (
            element_counts.items()
        ):

            print(
                f"        "
                f"{element_type:<10}: "
                f"{count}"
            )


        # ----------------------------------------------------
        # Chunk statistics
        # ----------------------------------------------------

        chunk_counts = {}

        for chunk in chunks:

            chunk_type = (
                chunk.chunk_type
            )

            chunk_counts[
                chunk_type
            ] = (
                chunk_counts.get(
                    chunk_type,
                    0,
                ) + 1
            )


        print(
            "\n      Chunks:"
        )

        for chunk_type, count in (
            chunk_counts.items()
        ):

            print(
                f"        "
                f"{chunk_type:<10}: "
                f"{count}"
            )


    # ========================================================
    # Complete
    # ========================================================

    print("\n" + "=" * 70)
    print("INGESTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()