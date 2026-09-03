import hashlib
import time
from pathlib import Path

from docling_core.types.doc import (
    PictureItem,
    TableItem,
    TextItem,
)

from rag.config import (
    IMAGES_DIR,
    TABLES_DIR,
)

from rag.models import Element


class DocumentExtractor:
    """
    Convert a DoclingDocument into our normalized
    multimodal RAG representation.

    Primary element types:

        text
        image
        table
    """

    # ========================================================
    # EXTRACT
    # ========================================================

    def extract(
        self,
        document,
        source_path: Path,
    ) -> list[Element]:

        document_hash = self._calculate_hash(
            source_path
        )

        document_id = document_hash[:16]

        elements = []

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        elements.extend(
            self._extract_text(
                document=document,
                document_id=document_id,
                document_hash=document_hash,
                source_path=source_path,
            )
        )

        # ----------------------------------------------------
        # IMAGES
        # ----------------------------------------------------

        elements.extend(
            self._extract_images(
                document=document,
                document_id=document_id,
                document_hash=document_hash,
                source_path=source_path,
            )
        )

        # ----------------------------------------------------
        # TABLES
        # ----------------------------------------------------

        elements.extend(
            self._extract_tables(
                document=document,
                document_id=document_id,
                document_hash=document_hash,
                source_path=source_path,
            )
        )

        return elements

    # ========================================================
    # TEXT
    # ========================================================

    def _extract_text(
        self,
        document,
        document_id,
        document_hash,
        source_path,
    ):

        elements = []

        for index, item in enumerate(
            document.texts,
            start=1,
        ):

            if not isinstance(
                item,
                TextItem,
            ):
                continue

            text = getattr(
                item,
                "text",
                None,
            )

            if not text:
                continue

            text = text.strip()

            if not text:
                continue

            element_id = (
                f"{document_id}_text_"
                f"{index:06d}"
            )

            elements.append(
                Element(
                    document_id=document_id,
                    document_name=source_path.name,
                    document_hash=document_hash,

                    element_id=element_id,
                    element_type="text",
                    source_ref=item.self_ref,

                    element_subtype=(
                        self._get_label(item)
                    ),

                    content=text,

                    page_number=(
                        self._get_page_number(item)
                    ),

                    parent_element_id=(
                        self._get_parent_id(item)
                    ),

                    related_element_ids=[],
                )
            )

        return elements

    # ========================================================
    # IMAGES
    # ========================================================

    def _extract_images(
        self,
        document,
        document_id,
        document_hash,
        source_path,
    ):

        elements = []

        output_dir = (
            IMAGES_DIR / document_id
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        image_count = len(
            document.pictures
        )

        print(
            f"\nImages detected by Docling: "
            f"{image_count}",
            flush=True,
        )

        total_start = time.perf_counter()

        for index, item in enumerate(
            document.pictures,
            start=1,
        ):

            if not isinstance(
                item,
                PictureItem,
            ):

                print(
                    "WARNING: Unexpected picture type: "
                    f"{type(item)}",
                    flush=True,
                )

                continue

            element_id = (
                f"{document_id}_image_"
                f"{index:06d}"
            )

            output_path = (
                output_dir
                / f"image_{index:06d}.png"
            )

            image_start = (
                time.perf_counter()
            )

            print(
                f"Extracting image "
                f"{index}/{image_count}...",
                flush=True,
            )

            # ------------------------------------------------
            # Get rendered image
            # ------------------------------------------------

            try:

                image = item.get_image(
                    document
                )

            except Exception as exc:

                print(
                    f"  ERROR extracting image "
                    f"{index}: {exc}",
                    flush=True,
                )

                continue

            if image is None:

                print(
                    f"  WARNING: image {index} "
                    "returned None",
                    flush=True,
                )

                continue

            # ------------------------------------------------
            # Save image
            # ------------------------------------------------

            try:

                image.save(
                    output_path,
                    format="PNG",
                )

                print(
                    f"  Saved: {output_path}",
                    flush=True,
                )

            except Exception as exc:

                print(
                    f"  ERROR saving image "
                    f"{index}: {exc}",
                    flush=True,
                )

                continue

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            caption = self._get_caption(
                document,
                item,
            )

            elements.append(
                Element(
                    document_id=document_id,
                    document_name=source_path.name,
                    document_hash=document_hash,

                    element_id=element_id,
                    element_type="image",
                    source_ref=item.self_ref,

                    element_subtype=(
                        self._get_image_subtype(item)
                    ),

                    content=None,

                    caption=caption,

                    page_number=(
                        self._get_page_number(item)
                    ),

                    parent_element_id=(
                        self._get_parent_id(item)
                    ),

                    related_element_ids=[],

                    asset_path=str(
                        output_path
                    ),
                )
            )

            image_time = (
                time.perf_counter()
                - image_start
            )

            print(
                f"  Image {index} complete: "
                f"{image_time:.2f}s",
                flush=True,
            )

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            f"IMAGE EXTRACTION COMPLETE "
            f"({image_count} images, "
            f"{total_time:.2f}s)",
            flush=True,
        )

        return elements

    # ========================================================
    # TABLES
    # ========================================================

    def _extract_tables(
        self,
        document,
        document_id,
        document_hash,
        source_path,
    ):

        elements = []

        output_dir = (
            TABLES_DIR / document_id
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        table_count = len(
            document.tables
        )

        print(
            f"\nTables detected by Docling: "
            f"{table_count}",
            flush=True,
        )

        total_start = (
            time.perf_counter()
        )

        for index, item in enumerate(
            document.tables,
            start=1,
        ):

            if not isinstance(
                item,
                TableItem,
            ):
                continue

            table_start = (
                time.perf_counter()
            )

            element_id = (
                f"{document_id}_table_"
                f"{index:06d}"
            )

            table_path = (
                output_dir
                / f"table_{index:06d}.json"
            )

            print(
                f"\n[Table {index}/{table_count}] "
                "Starting...",
                flush=True,
            )

            # ------------------------------------------------
            # Structured table
            # ------------------------------------------------

            dataframe_start = (
                time.perf_counter()
            )

            try:

                print(
                    "  Exporting dataframe...",
                    flush=True,
                )

                dataframe = (
                    item.export_to_dataframe(
                        document
                    )
                )

                dataframe.to_json(
                    table_path,
                    orient="records",
                    indent=2,
                    force_ascii=False,
                )

                dataframe_time = (
                    time.perf_counter()
                    - dataframe_start
                )

                print(
                    f"  Dataframe export: "
                    f"{dataframe_time:.2f}s",
                    flush=True,
                )

            except Exception as exc:

                print(
                    f"  WARNING: table {index} "
                    f"could not be exported: "
                    f"{exc}",
                    flush=True,
                )

                table_path = None

            # ------------------------------------------------
            # Markdown
            # ------------------------------------------------

            markdown_start = (
                time.perf_counter()
            )

            try:

                print(
                    "  Exporting markdown...",
                    flush=True,
                )

                markdown = (
                    item.export_to_markdown(
                        document
                    )
                )

                markdown_time = (
                    time.perf_counter()
                    - markdown_start
                )

                print(
                    f"  Markdown export: "
                    f"{markdown_time:.2f}s",
                    flush=True,
                )

            except Exception as exc:

                print(
                    f"  WARNING: markdown export "
                    f"failed: {exc}",
                    flush=True,
                )

                markdown = None

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

            caption = self._get_caption(
                document,
                item,
            )

            page_number = (
                self._get_page_number(
                    item
                )
            )

            elements.append(
                Element(
                    document_id=document_id,
                    document_name=source_path.name,
                    document_hash=document_hash,

                    element_id=element_id,
                    element_type="table",
                    source_ref=item.self_ref,

                    content=markdown,

                    caption=caption,

                    page_number=page_number,

                    parent_element_id=(
                        self._get_parent_id(
                            item
                        )
                    ),

                    related_element_ids=[],

                    asset_path=(
                        str(table_path)
                        if table_path
                        else None
                    ),
                )
            )

            table_time = (
                time.perf_counter()
                - table_start
            )

            print(
                f"  Table {index} complete: "
                f"{table_time:.2f}s",
                flush=True,
            )

        total_time = (
            time.perf_counter()
            - total_start
        )

        print(
            "\nTABLE EXTRACTION COMPLETE "
            f"({table_count} tables, "
            f"{total_time:.2f}s)",
            flush=True,
        )

        return elements

    # ========================================================
    # FILE HASH
    # ========================================================

    @staticmethod
    def _calculate_hash(
        path: Path,
    ) -> str:

        sha256 = hashlib.sha256()

        with path.open(
            "rb"
        ) as file:

            while chunk := file.read(
                1024 * 1024
            ):

                sha256.update(
                    chunk
                )

        return sha256.hexdigest()

    # ========================================================
    # PAGE
    # ========================================================

    @staticmethod
    def _get_page_number(
        item,
    ):

        try:

            if item.prov:

                return item.prov[0].page_no

        except (
            AttributeError,
            IndexError,
            TypeError,
        ):

            pass

        return None

    # ========================================================
    # PARENT
    # ========================================================

    @staticmethod
    def _get_parent_id(
        item,
    ):

        try:

            if item.parent is None:

                return None

            return item.parent.cref

        except AttributeError:

            return None

    # ========================================================
    # CHILDREN
    # ========================================================

    @staticmethod
    def _get_child_ids(
        item,
    ):

        result = []

        try:

            for child in item.children:

                if hasattr(
                    child,
                    "cref",
                ):

                    result.append(
                        child.cref
                    )

        except (
            AttributeError,
            TypeError,
        ):

            pass

        return result

    # ========================================================
    # LABEL
    # ========================================================

    @staticmethod
    def _get_label(
        item,
    ):

        try:

            label = item.label

            if hasattr(
                label,
                "value",
            ):

                return label.value

            return str(label)

        except AttributeError:

            return None

    # ========================================================
    # IMAGE SUBTYPE
    # ========================================================

    @staticmethod
    def _get_image_subtype(
        item,
    ):

        label = (
            DocumentExtractor._get_label(
                item
            )
        )

        if not label:

            return None

        label = label.lower()

        if label == "picture":
            return "figure"

        if label == "chart":
            return "chart"

        if label == "diagram":
            return "diagram"

        return label

    # ========================================================
    # CAPTION
    # ========================================================

    @staticmethod
    def _get_caption(
        document,
        item,
    ):

        try:

            caption = item.caption_text(
                document
            )

            if caption:

                return caption.strip()

        except Exception:

            pass

        return None