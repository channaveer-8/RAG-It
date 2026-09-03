import os

# ============================================================
# CPU / Windows Torch compatibility
# ============================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CPP"] = "1"


import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))


from rag.config import PARSED_DIR
from rag.ingestion.scanner import scan_documents
from rag.ingestion.parser import DoclingParser


def main():

    print("=" * 70)
    print("DOCLING PARSE + RAW JSON EXPORT")
    print("=" * 70)

    documents = scan_documents()

    print(f"\nDocuments found: {len(documents)}")

    if not documents:
        print("\nNo documents found.")
        return

    parser = DoclingParser()

    for source in documents:

        print("\n" + "-" * 70)
        print(f"Document: {source.name}")
        print("-" * 70)

        print("\nParsing...")

        document = parser.parse(source)

        print("Parsing complete.")

        # ----------------------------------------------------
        # Basic information
        # ----------------------------------------------------

        print(
            f"\nDocling version : {document.version}"
        )

        print(
            f"Pages           : {document.num_pages()}"
        )

        print(
            f"Text elements   : {len(document.texts)}"
        )

        print(
            f"Pictures        : {len(document.pictures)}"
        )

        print(
            f"Tables          : {len(document.tables)}"
        )

        print(
            f"Groups          : {len(document.groups)}"
        )

        # ----------------------------------------------------
        # Save raw Docling JSON
        # ----------------------------------------------------

        output_name = (
            source.stem + ".json"
        )

        output_path = (
            PARSED_DIR / output_name
        )

        print(
            f"\nSaving raw Docling JSON:"
        )

        print(output_path)

        parser.save_json(
            document,
            output_path,
        )

        print("Saved.")


    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()