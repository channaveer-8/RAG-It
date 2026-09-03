from typing import Any


class ContextBuilder:
    """
    Build LLM-ready context from retrieved RAG chunks.

    This class is intentionally independent of:

        - embedding models
        - vector databases
        - rerankers
        - LLMs

    Input:
        Retrieved Qdrant results.

    Output:
        A single formatted evidence/context string.

    Design goals:

        1. Give the LLM useful document evidence.
        2. Preserve document and page information.
        3. Support text, image, and table chunks.
        4. Do not expose internal retrieval labels such as
           "Context 1", "Context 2", etc.
        5. Keep chunk IDs available for traceability.
        6. Avoid unnecessary metadata in the LLM context.
    """

    def __init__(
        self,
        max_chunks: int = 5,
    ):

        self.max_chunks = max_chunks


    # ========================================================
    # Build Context
    # ========================================================

    def build(
        self,
        results,
    ) -> str:
        """
        Convert retrieved results into a single context string.
        """

        if not results:

            return ""


        context_parts = []


        # ----------------------------------------------------
        # Only use the configured number of chunks.
        # ----------------------------------------------------

        for item in results[:self.max_chunks]:

            result = self._get_result(
                item
            )


            if result is None:

                continue


            payload = (
                getattr(
                    result,
                    "payload",
                    None,
                )
                or {}
            )


            if not isinstance(
                payload,
                dict,
            ):

                continue


            chunk_context = (
                self._build_chunk(
                    payload
                )
            )


            if chunk_context:

                context_parts.append(
                    chunk_context
                )


        return "\n\n".join(
            context_parts
        ).strip()


    # ========================================================
    # Build Individual Chunk
    # ========================================================

    @staticmethod
    def _build_chunk(
        payload: dict[str, Any],
    ) -> str:
        """
        Build the evidence representation of one chunk.
        """

        lines = []


        # ====================================================
        # Metadata
        # ====================================================

        document_name = (
            payload.get(
                "document_name"
            )
        )


        document_id = (
            payload.get(
                "document_id"
            )
        )


        chunk_id = (
            payload.get(
                "chunk_id"
            )
        )


        chunk_type = (
            payload.get(
                "chunk_type"
            )
        )


        pages = (
            payload.get(
                "pages",
                payload.get(
                    "page_numbers",
                    [],
                ),
            )
        )


        content = (
            payload.get(
                "content"
            )
        )


        caption = (
            payload.get(
                "caption"
            )
        )


        asset_path = (
            payload.get(
                "asset_path"
            )
        )


        source_refs = (
            payload.get(
                "source_refs",
                [],
            )
        )


        # ====================================================
        # Source information
        #
        # Do NOT call this "Context 1", etc.
        # ====================================================

        lines.append(
            "DOCUMENT EVIDENCE"
        )


        # ----------------------------------------------------
        # Document
        # ----------------------------------------------------

        if document_name:

            lines.append(
                f"Document: "
                f"{document_name}"
            )

        elif document_id:

            lines.append(
                f"Document ID: "
                f"{document_id}"
            )


        # ----------------------------------------------------
        # Pages
        # ----------------------------------------------------

        formatted_pages = (
            ContextBuilder._format_pages(
                pages
            )
        )


        if formatted_pages:

            lines.append(
                f"Pages: "
                f"{formatted_pages}"
            )


        # ----------------------------------------------------
        # Type
        # ----------------------------------------------------

        if chunk_type:

            lines.append(
                f"Type: "
                f"{chunk_type}"
            )


        # ----------------------------------------------------
        # Chunk ID
        #
        # This is useful for traceability but the prompt
        # will tell the LLM not to mention internal IDs.
        # ----------------------------------------------------

        if chunk_id:

            lines.append(
                f"Chunk ID: "
                f"{chunk_id}"
            )


        # ====================================================
        # Source references
        # ====================================================

        if source_refs:

            formatted_refs = (
                ContextBuilder._format_source_refs(
                    source_refs
                )
            )


            if formatted_refs:

                lines.append(
                    f"Source reference: "
                    f"{formatted_refs}"
                )


        # ====================================================
        # Caption
        # ====================================================

        if caption:

            caption_text = str(
                caption
            ).strip()


            if caption_text:

                lines.append(
                    ""
                )

                lines.append(
                    "CAPTION"
                )

                lines.append(
                    caption_text
                )


        # ====================================================
        # Content
        # ====================================================

        if content is not None:

            content_text = str(
                content
            ).strip()


            if content_text:

                lines.append(
                    ""
                )

                lines.append(
                    "CONTENT"
                )

                lines.append(
                    content_text
                )


        # ====================================================
        # Asset
        # ====================================================

        if asset_path:

            asset_text = str(
                asset_path
            ).strip()


            if asset_text:

                lines.append(
                    ""
                )

                lines.append(
                    "ASSET"
                )

                lines.append(
                    asset_text
                )


        # ====================================================
        # Final validation
        # ====================================================

        result = "\n".join(
            lines
        ).strip()


        # Don't return metadata-only chunks.
        #
        # A chunk should contain some actual evidence,
        # caption, or asset information.
        # ====================================================

        has_content = bool(
            content
            and str(
                content
            ).strip()
        )


        has_caption = bool(
            caption
            and str(
                caption
            ).strip()
        )


        has_asset = bool(
            asset_path
            and str(
                asset_path
            ).strip()
        )


        if not (
            has_content
            or has_caption
            or has_asset
        ):

            return ""


        return result


    # ========================================================
    # Page formatting
    # ========================================================

    @staticmethod
    def _format_pages(
        pages,
    ) -> str:
        """
        Format page numbers.

        Examples:

            [12]
                -> "12"

            [12, 13, 14]
                -> "12-14"

            [12, 14, 15]
                -> "12, 14-15"

        This makes source information easier for the LLM
        to understand and easier for the UI to display.
        """

        if pages is None:

            return ""


        # ----------------------------------------------------
        # Normalize a single page.
        # ----------------------------------------------------

        if isinstance(
            pages,
            (int, float, str),
        ):

            return str(
                pages
            )


        if not isinstance(
            pages,
            (list, tuple, set),
        ):

            return str(
                pages
            )


        if not pages:

            return ""


        # ----------------------------------------------------
        # Convert to integers when possible.
        # ----------------------------------------------------

        normalized = []


        for page in pages:

            try:

                page_number = int(
                    page
                )

            except (
                ValueError,
                TypeError,
            ):

                continue


            if page_number not in normalized:

                normalized.append(
                    page_number
                )


        if not normalized:

            return ""


        normalized.sort()


        # ----------------------------------------------------
        # Build ranges.
        # ----------------------------------------------------

        ranges = []


        start = normalized[0]

        previous = normalized[0]


        for page in normalized[1:]:

            if page == previous + 1:

                previous = page

                continue


            # Close previous range.

            if start == previous:

                ranges.append(
                    str(start)
                )

            else:

                ranges.append(
                    f"{start}-{previous}"
                )


            start = page

            previous = page


        # ----------------------------------------------------
        # Close final range.
        # ----------------------------------------------------

        if start == previous:

            ranges.append(
                str(start)
            )

        else:

            ranges.append(
                f"{start}-{previous}"
            )


        return ", ".join(
            ranges
        )


    # ========================================================
    # Source references
    # ========================================================

    @staticmethod
    def _format_source_refs(
        source_refs,
    ) -> str:
        """
        Convert source references into a compact string.
        """

        if not source_refs:

            return ""


        if isinstance(
            source_refs,
            str,
        ):

            return source_refs.strip()


        if not isinstance(
            source_refs,
            (list, tuple),
        ):

            return str(
                source_refs
            )


        formatted = []


        for reference in source_refs:

            if reference is None:

                continue


            if isinstance(
                reference,
                dict,
            ):

                # --------------------------------------------
                # Common source reference fields.
                # --------------------------------------------

                value = (
                    reference.get(
                        "ref"
                    )
                    or reference.get(
                        "source_ref"
                    )
                    or reference.get(
                        "id"
                    )
                    or reference.get(
                        "cref"
                    )
                )


                if value is not None:

                    formatted.append(
                        str(value)
                    )

                else:

                    formatted.append(
                        str(reference)
                    )

            else:

                formatted.append(
                    str(reference)
                )


        return ", ".join(
            formatted
        )


    # ========================================================
    # Normalize Retrieval Result
    # ========================================================

    @staticmethod
    def _get_result(
        item,
    ):
        """
        Normalize different retrieval result formats.

        Supported:

        1. Direct Qdrant result

            ScoredPoint(...)

        2. Reranker result

            {
                "result": qdrant_result,
                "reranker_score": ...
            }

        3. Dictionary containing a Qdrant result under
           another common key.
        """

        if item is None:

            return None


        # ----------------------------------------------------
        # Reranker format
        # ----------------------------------------------------

        if isinstance(
            item,
            dict,
        ):

            if "result" in item:

                return item[
                    "result"
                ]


            if "qdrant_result" in item:

                return item[
                    "qdrant_result"
                ]


            if "point" in item:

                return item[
                    "point"
                ]


        # ----------------------------------------------------
        # Direct Qdrant result
        # ----------------------------------------------------

        return item