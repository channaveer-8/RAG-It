import os

# ============================================================
# CPU / Windows Torch compatibility
# ============================================================

os.environ["TORCH_COMPILE_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CPP"] = "1"


from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    """
    Generic Stage-2 cross-encoder reranker.

    Stage 1:
        Visualized-BGE -> Qdrant -> Top-N candidates

    Stage 2:
        CrossEncoder -> reranked candidates

    The actual model is supplied through configuration.
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        max_length: int = 512,
    ):

        self.model_name = model_name
        self.device = device
        self.max_length = max_length

        print(
            f"Loading {model_name}..."
        )

        self.model = CrossEncoder(
            model_name,
            max_length=max_length,
            device=device,
        )

        print(
            f"{model_name} loaded."
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

            candidate_text = (
                self._build_candidate_text(
                    payload
                )
            )

            if not candidate_text:
                continue


            pairs.append(
                (
                    query,
                    candidate_text,
                )
            )

            valid_results.append(
                result
            )


        if not pairs:
            return []


        scores = self.model.predict(
            pairs,
            show_progress_bar=False,
            batch_size=8,
        )


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
    # Convert chunk payload to text
    # ========================================================

    @staticmethod
    def _build_candidate_text(
        payload,
    ):

        parts = []


        chunk_type = payload.get(
            "chunk_type"
        )

        caption = payload.get(
            "caption"
        )

        content = payload.get(
            "content"
        )


        if chunk_type:

            parts.append(
                f"Type: {chunk_type}"
            )


        if caption:

            parts.append(
                f"Caption: {caption}"
            )


        if content:

            parts.append(
                f"Content:\n{content}"
            )


        return "\n".join(
            parts
        ).strip()