from dataclasses import dataclass, field
from typing import Optional

from rag.models import Element


# ============================================================
# Retrieval Chunk
# ============================================================

@dataclass
class Chunk:

    chunk_id: str

    document_id: str
    document_name: str

    chunk_type: str

    content: Optional[str] = None

    element_ids: list[str] = field(
        default_factory=list
    )

    source_refs: list[str] = field(
        default_factory=list
    )

    page_numbers: list[int] = field(
        default_factory=list
    )

    asset_path: Optional[str] = None

    caption: Optional[str] = None

    metadata: dict = field(
        default_factory=dict
    )

    def to_dict(self) -> dict:

        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "chunk_type": self.chunk_type,
            "content": self.content,
            "element_ids": self.element_ids,
            "source_refs": self.source_refs,
            "page_numbers": self.page_numbers,
            "asset_path": self.asset_path,
            "caption": self.caption,
            "metadata": self.metadata,
        }


# ============================================================
# Document Chunker
# ============================================================

class DocumentChunker:
    """
    Generic, configurable chunker for multimodal documents.

    Supported element types:

        text
        image
        table

    Supported text strategies:

        fixed
        recursive

    Images and tables remain independent retrieval units.
    """

    VALID_STRATEGIES = {
        "fixed",
        "recursive",
    }

    def __init__(
        self,
        max_chars: int = 1500,
        overlap: int = 200,
        strategy: str = "recursive",
    ):

        if max_chars <= 0:
            raise ValueError(
                "max_chars must be greater than 0"
            )

        if overlap < 0:
            raise ValueError(
                "overlap cannot be negative"
            )

        if overlap >= max_chars:
            raise ValueError(
                "overlap must be smaller than max_chars"
            )

        if strategy not in self.VALID_STRATEGIES:
            raise ValueError(
                f"Unknown chunking strategy: {strategy}. "
                f"Supported: {self.VALID_STRATEGIES}"
            )

        self.max_chars = max_chars
        self.overlap = overlap
        self.strategy = strategy

    # ========================================================
    # PUBLIC API
    # ========================================================

    def chunk(
        self,
        elements: list[Element],
    ) -> list[Chunk]:

        if not elements:
            return []

        # ----------------------------------------------------
        # Preserve original document order
        # ----------------------------------------------------

        ordered_elements = self._sort_elements(
            elements
        )

        chunks = []

        text_buffer = []

        for element in ordered_elements:

            if element.element_type == "text":

                text_buffer.append(
                    element
                )

                continue

            # ------------------------------------------------
            # Flush text before image/table
            # ------------------------------------------------

            if text_buffer:

                chunks.extend(
                    self._chunk_text(
                        text_buffer
                    )
                )

                text_buffer = []

            # ------------------------------------------------
            # Image / Table
            # ------------------------------------------------

            chunks.append(
                self._create_asset_chunk(
                    element
                )
            )

        # ----------------------------------------------------
        # Flush remaining text
        # ----------------------------------------------------

        if text_buffer:

            chunks.extend(
                self._chunk_text(
                    text_buffer
                )
            )

        # ----------------------------------------------------
        # Assign deterministic chunk IDs
        # ----------------------------------------------------

        for index, chunk in enumerate(
            chunks,
            start=1,
        ):

            chunk.chunk_id = (
                f"{chunk.document_id}"
                f"_chunk_{index:06d}"
            )

            chunk.metadata[
                "chunk_index"
            ] = index

        return chunks

    # ========================================================
    # ORDERING
    # ========================================================

    @staticmethod
    def _sort_elements(
        elements: list[Element],
    ) -> list[Element]:

        """
        Preserve approximate document order.

        Primary:
            page number

        Secondary:
            original element order
            represented by the numeric suffix
            in element_id.
        """

        def sort_key(
            element: Element,
        ):

            page = (
                element.page_number
                if element.page_number is not None
                else 10**9
            )

            # ------------------------------------------------
            # Extract numeric element index
            # ------------------------------------------------

            try:

                index = int(
                    element.element_id
                    .rsplit("_", 1)[-1]
                )

            except (
                ValueError,
                IndexError,
            ):

                index = 10**9

            return (
                page,
                index,
            )

        return sorted(
            elements,
            key=sort_key,
        )

    # ========================================================
    # TEXT
    # ========================================================

    def _chunk_text(
        self,
        elements: list[Element],
    ) -> list[Chunk]:

        if self.strategy == "fixed":

            return self._chunk_text_fixed(
                elements
            )

        return self._chunk_text_recursive(
            elements
        )

    # ========================================================
    # RECURSIVE TEXT CHUNKING
    # ========================================================

    def _chunk_text_recursive(
        self,
        elements: list[Element],
    ) -> list[Chunk]:

        chunks = []

        current = []

        current_length = 0

        for element in elements:

            text = (
                element.content or ""
            ).strip()

            if not text:
                continue

            # ------------------------------------------------
            # If a single element is too large,
            # split the element itself.
            # ------------------------------------------------

            if len(text) > self.max_chars:

                if current:

                    chunks.append(
                        self._create_text_chunk(
                            current
                        )
                    )

                    current = []
                    current_length = 0

                chunks.extend(
                    self._split_large_element(
                        element
                    )
                )

                continue

            additional_length = (
                len(text)
                if not current
                else len(text) + 2
            )

            if (
                current
                and
                current_length
                + additional_length
                > self.max_chars
            ):

                chunks.append(
                    self._create_text_chunk(
                        current
                    )
                )

                current = (
                    self._build_overlap(
                        current
                    )
                )

                current_length = len(
                    self._join_text(
                        current
                    )
                )

            current.append(
                element
            )

            current_length = len(
                self._join_text(
                    current
                )
            )

        if current:

            chunks.append(
                self._create_text_chunk(
                    current
                )
            )

        return chunks

    # ========================================================
    # FIXED TEXT CHUNKING
    # ========================================================

    def _chunk_text_fixed(
        self,
        elements: list[Element],
    ) -> list[Chunk]:

        text = self._join_text(
            elements
        )

        if not text:

            return []

        chunks = []

        start = 0

        while start < len(text):

            end = min(
                start + self.max_chars,
                len(text),
            )

            chunk_text = text[
                start:end
            ]

            chunks.append(
                self._create_text_chunk_from_raw(
                    chunk_text,
                    elements,
                )
            )

            if end >= len(text):

                break

            start = (
                end - self.overlap
            )

        return chunks

    # ========================================================
    # LARGE ELEMENT
    # ========================================================

    def _split_large_element(
        self,
        element: Element,
    ) -> list[Chunk]:

        text = (
            element.content or ""
        ).strip()

        chunks = []

        start = 0

        while start < len(text):

            end = min(
                start + self.max_chars,
                len(text),
            )

            piece = text[
                start:end
            ]

            chunks.append(
                self._create_text_chunk_from_raw(
                    piece,
                    [element],
                )
            )

            if end >= len(text):

                break

            start = (
                end - self.overlap
            )

        return chunks

    # ========================================================
    # TEXT CHUNK CREATION
    # ========================================================

    def _create_text_chunk(
        self,
        elements: list[Element],
    ) -> Chunk:

        content = self._join_text(
            elements
        )

        return self._create_text_chunk_from_raw(
            content,
            elements,
        )

    def _create_text_chunk_from_raw(
        self,
        content: str,
        elements: list[Element],
    ) -> Chunk:

        first = elements[0]

        pages = sorted(
            {
                e.page_number
                for e in elements
                if e.page_number is not None
            }
        )

        return Chunk(

            chunk_id="",

            document_id=(
                first.document_id
            ),

            document_name=(
                first.document_name
            ),

            chunk_type="text",

            content=content,

            element_ids=[
                e.element_id
                for e in elements
            ],

            source_refs=[
                e.source_ref
                for e in elements
                if e.source_ref
            ],

            page_numbers=pages,

            metadata={
                "strategy": self.strategy,
                "max_chars": self.max_chars,
                "overlap": self.overlap,
                "element_count": len(
                    elements
                ),
                "parser": "docling",
            },
        )

    # ========================================================
    # IMAGE / TABLE
    # ========================================================

    @staticmethod
    def _create_asset_chunk(
        element: Element,
    ) -> Chunk:

        # ----------------------------------------------------
        # Image
        # ----------------------------------------------------

        if element.element_type == "image":

            # Caption is the initial textual representation
            # used later for multimodal retrieval.
            content = (
                element.caption
                or ""
            )

        # ----------------------------------------------------
        # Table
        # ----------------------------------------------------

        elif element.element_type == "table":

            content = (
                element.content
                or ""
            )

        else:

            raise ValueError(
                f"Unsupported asset type: "
                f"{element.element_type}"
            )

        return Chunk(

            chunk_id="",

            document_id=(
                element.document_id
            ),

            document_name=(
                element.document_name
            ),

            chunk_type=(
                element.element_type
            ),

            content=content,

            element_ids=[
                element.element_id
            ],

            source_refs=(
                [element.source_ref]
                if element.source_ref
                else []
            ),

            page_numbers=(
                [element.page_number]
                if element.page_number
                is not None
                else []
            ),

            asset_path=(
                element.asset_path
            ),

            caption=(
                element.caption
            ),

            metadata={
                "element_subtype": (
                    element.element_subtype
                ),
                "parser": "docling",
            },
        )

    # ========================================================
    # OVERLAP
    # ========================================================

    def _build_overlap(
        self,
        elements: list[Element],
    ) -> list[Element]:

        result = []

        total = 0

        for element in reversed(
            elements
        ):

            text = (
                element.content
                or ""
            )

            if (
                total + len(text)
                > self.overlap
            ):

                break

            result.insert(
                0,
                element,
            )

            total += len(text)

        return result

    # ========================================================
    # JOIN TEXT
    # ========================================================

    @staticmethod
    def _join_text(
        elements: list[Element],
    ) -> str:

        return "\n\n".join(
            element.content.strip()
            for element in elements
            if element.content
        )