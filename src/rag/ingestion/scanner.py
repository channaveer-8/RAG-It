from pathlib import Path

from rag.config import (
    DOCUMENTS_DIR,
    SUPPORTED_EXTENSIONS,
)


def scan_documents() -> list[Path]:
    """
    Find supported documents recursively.
    """

    if not DOCUMENTS_DIR.exists():
        DOCUMENTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

    documents = []

    for path in DOCUMENTS_DIR.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        documents.append(path)

    return sorted(documents)