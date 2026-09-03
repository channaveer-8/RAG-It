import os

# ============================================================
# CPU / Windows Torch compatibility
# ============================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CPP"] = "1"


from FlagEmbedding import FlagReranker


class BGEReranker:
    """
    BGE cross-encoder reranker.

    Stage 1:
        Qdrant retrieves candidates.

    Stage 2:
        BGE reranker scores query/candidate pairs.
    """

    def __init__(
        self,
        model_name: str = (
            "BAAI/bge-reranker-v2-m3"
        ),
        use_fp16: bool = False,
    ):

        print(
            "Loading BGE reranker..."
        )

        self.model = FlagReranker(
            model_name,
            use_fp16=use_fp16,
        )

        print(
            "BGE reranker loaded."
        )


    # ========================================================
    # Rerank
    # ========================================================

    def rerank(
        self,
        query: str,
        results,
    ):

        if not results:
            return []


        pairs = []

        valid_results = []


        for result in results:

            payload = (
                result.payload
                or {}
            )

            content = self._get_text(
                payload
            )

            if not content:
                continue


            pairs.append(
                [
                    query,
                    content,
                ]
            )

            valid_results.append(
                result
            )


        if not pairs:
            return []


        scores = (
            self.model.compute_score(
                pairs,
                normalize=True,
            )
        )


        # Single result returns float
        if isinstance(
            scores,
            float,
        ):

            scores = [scores]


        reranked = []


        for result, score in zip(
            valid_results,
            scores,
        ):

            reranked.append(
                {
                    "result": result,
                    "reranker_score": float(
                        score
                    ),
                }
            )


        reranked.sort(
            key=lambda item:
                item["reranker_score"],
            reverse=True,
        )


        return reranked


    # ========================================================
    # Candidate text
    # ========================================================

    @staticmethod
    def _get_text(
        payload,
    ):

        content = (
            payload.get("content")
        )

        caption = (
            payload.get("caption")
        )


        parts = []


        if caption:
            parts.append(
                str(caption)
            )


        if content:
            parts.append(
                str(content)
            )


        return "\n".join(
            parts
        ).strip()