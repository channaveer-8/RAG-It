import json
import time
import uuid
from pathlib import Path

from qdrant_client.models import PointStruct

from rag.config import (
    METADATA_DIR,
    CHUNKS_DIR,
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


class DocumentIndexer:
    """
    Single-document ingestion and indexing.

    Pipeline:

        Document
          ↓
        Docling
          ↓
        Extraction
          ↓
        Chunking
          ↓
        Embedding
          ↓
        Qdrant
    """

    def __init__(
        self,
        progress_callback=None,
    ):

        self.progress_callback = (
            progress_callback
        )

        self.parser = DoclingParser()

        self.extractor = DocumentExtractor()

        self.chunker = DocumentChunker(
            max_chars=1500,
            overlap=200,
            strategy="recursive",
        )

    # ========================================================
    # Progress
    # ========================================================

    def _progress(
        self,
        message: str,
    ):

        print(
            f"[INDEXER] {message}",
            flush=True,
        )

        if self.progress_callback:

            self.progress_callback(
                message
            )

    # ========================================================
    # Index document
    # ========================================================

    def index_document(
        self,
        source: Path,
        embedder,
        vectorstore,
    ):

        source = Path(source)

        if not source.exists():

            raise FileNotFoundError(
                f"Document not found: {source}"
            )

        total_start = time.perf_counter()

        self._progress(
            f"START: {source.name}"
        )

        # ====================================================
        # 1. Parse
        # ====================================================

        start = time.perf_counter()

        self._progress(
            "1/5 Parsing with Docling..."
        )

        document = self.parser.parse(
            source
        )

        self._progress(
            "1/5 Parsing complete "
            f"({time.perf_counter() - start:.2f}s)"
        )

        # ====================================================
        # 2. Extract
        # ====================================================

        start = time.perf_counter()

        self._progress(
            "2/5 Extracting elements..."
        )

        elements = self.extractor.extract(
            document=document,
            source_path=source,
        )

        self._progress(
            "2/5 Extraction complete: "
            f"{len(elements)} elements "
            f"({time.perf_counter() - start:.2f}s)"
        )

        if not elements:

            raise RuntimeError(
                f"No elements extracted from "
                f"{source.name}"
            )

        document_id = (
            elements[0].document_id
        )

        # ====================================================
        # Save normalized metadata
        # ====================================================

        metadata_path = (
            METADATA_DIR
            / f"{document_id}.json"
        )

        metadata = {
            "document_id": document_id,
            "document_name": source.name,
            "parser": "docling",
            "element_count": len(elements),
            "elements": [
                element.to_dict()
                for element in elements
            ],
        }

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with metadata_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

        self._progress(
            "Metadata saved."
        )

        # ====================================================
        # 3. Chunk
        # ====================================================

        start = time.perf_counter()

        self._progress(
            "3/5 Creating chunks..."
        )

        chunks = self.chunker.chunk(
            elements
        )

        self._progress(
            "3/5 Chunking complete: "
            f"{len(chunks)} chunks "
            f"({time.perf_counter() - start:.2f}s)"
        )

        if not chunks:

            raise RuntimeError(
                f"No chunks created from "
                f"{source.name}"
            )

        # ====================================================
        # Save chunks
        # ====================================================

        chunk_path = (
            CHUNKS_DIR
            / f"{document_id}.json"
        )

        chunk_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        chunk_data = {
            "document_id": document_id,
            "document_name": source.name,
            "chunk_count": len(chunks),
            "chunking": {
                "strategy": "recursive",
                "max_chars": 1500,
                "overlap": 200,
            },
            "chunks": [
                chunk.to_dict()
                for chunk in chunks
            ],
        }

        with chunk_path.open(
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                chunk_data,
                file,
                indent=2,
                ensure_ascii=False,
            )

        self._progress(
            "Chunk metadata saved."
        )

        # ====================================================
        # 4. Embeddings
        # ====================================================

        start = time.perf_counter()

        self._progress(
            "4/5 Generating embeddings..."
        )

        points = []

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            chunk_type = (
                chunk.chunk_type
            )

            self._progress(
                f"Embedding "
                f"{index}/{len(chunks)} "
                f"[{chunk_type}] "
                f"{chunk.chunk_id}"
            )

            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            if chunk_type == "text":

                content = (
                    chunk.content
                    or ""
                )

                if not content.strip():

                    self._progress(
                        "Skipping empty text chunk."
                    )

                    continue

                embedding = (
                    embedder.encode_text(
                        content
                    )
                )

            # ------------------------------------------------
            # Image
            # ------------------------------------------------

            elif chunk_type == "image":

                asset_path = (
                    chunk.asset_path
                )

                if not asset_path:

                    self._progress(
                        "Skipping image without asset."
                    )

                    continue

                embedding = (
                    embedder.encode_image(
                        asset_path
                    )
                )

            # ------------------------------------------------
            # Table
            # ------------------------------------------------

            elif chunk_type == "table":

                content = (
                    chunk.content
                    or ""
                )

                if not content.strip():

                    self._progress(
                        "Skipping empty table chunk."
                    )

                    continue

                embedding = (
                    embedder.encode_text(
                        content
                    )
                )

            else:

                self._progress(
                    f"Skipping unsupported type: "
                    f"{chunk_type}"
                )

                continue

            vector = (
                embedder.to_numpy(
                    embedding
                )
                .flatten()
                .tolist()
            )

            payload = {
                "document_id":
                    chunk.document_id,

                "document_name":
                    chunk.document_name,

                "chunk_id":
                    chunk.chunk_id,

                "chunk_type":
                    chunk.chunk_type,

                "content":
                    chunk.content,

                "element_ids":
                    chunk.element_ids,

                "source_refs":
                    chunk.source_refs,

                "page_numbers":
                    chunk.page_numbers,

                "asset_path":
                    chunk.asset_path,

                "caption":
                    chunk.caption,

                "metadata":
                    chunk.metadata,
            }

            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    chunk.chunk_id,
                )
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload=payload,
                )
            )

        self._progress(
            "4/5 Embeddings complete: "
            f"{len(points)} vectors "
            f"({time.perf_counter() - start:.2f}s)"
        )

        if not points:

            raise RuntimeError(
                f"No embeddable chunks found in "
                f"{source.name}"
            )

        # ====================================================
        # 5. Qdrant
        # ====================================================

        start = time.perf_counter()

        self._progress(
            "5/5 Uploading vectors to Qdrant..."
        )

        vectorstore.create_collection(
            recreate=False
        )

        vectorstore.upsert(
            points
        )

        self._progress(
            "5/5 Qdrant upload complete "
            f"({time.perf_counter() - start:.2f}s)"
        )

        total_time = (
            time.perf_counter()
            - total_start
        )

        self._progress(
            f"COMPLETE: {source.name} "
            f"in {total_time:.2f}s"
        )

        return {
            "document_id":
                document_id,

            "document_name":
                source.name,

            "elements":
                len(elements),

            "chunks":
                len(chunks),

            "vectors":
                len(points),

            "time":
                total_time,
        }

    # ========================================================
    # Remove document
    # ========================================================

    def remove_document(
        self,
        document_name: str,
        vectorstore,
    ):

        metadata_file = None
        chunk_file = None
        document_id = None

        for path in METADATA_DIR.glob(
            "*.json"
        ):

            try:

                with path.open(
                    "r",
                    encoding="utf-8",
                ) as file:

                    data = json.load(file)

            except Exception:

                continue

            if (
                data.get(
                    "document_name"
                )
                == document_name
            ):

                document_id = (
                    data.get(
                        "document_id"
                    )
                )

                metadata_file = path

                break

        if document_id:

            chunk_file = (
                CHUNKS_DIR
                / f"{document_id}.json"
            )

        vectorstore.delete_document(
            document_name=document_name,
            document_id=document_id,
        )

        if metadata_file is not None:

            metadata_file.unlink(
                missing_ok=True
            )

        if chunk_file is not None:

            chunk_file.unlink(
                missing_ok=True
            )

        return document_id