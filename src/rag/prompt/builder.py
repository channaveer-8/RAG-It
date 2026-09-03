class PromptBuilder:
    """
    Builds user-configurable RAG-specific instructions.

    The final LLM prompt is assembled by OllamaLLM.

    This class does not know about:
        - Qdrant
        - embeddings
        - rerankers
        - Ollama
        - retrieved context
        - user queries
    """

    DEFAULT_RAG_PROMPT = (
        "Answer the user's question using the provided "
        "document context.\n\n"

        "Use the retrieved documents as the primary source "
        "of factual information.\n"

        "If the retrieved context contains the answer, "
        "provide a complete and useful answer based on it.\n"

        "If the answer is not present in the retrieved "
        "context, clearly say that the information could "
        "not be found in the provided documents.\n\n"

        "Do not invent facts, commands, values, procedures, "
        "or technical details that are not supported by the "
        "provided documents.\n\n"

        "Do not refer to the retrieved material as "
        "\"Context 1\", \"Context 2\", etc.\n"

        "Do not expose internal chunk IDs, retrieval scores, "
        "or other internal RAG implementation details unless "
        "the user explicitly asks for them.\n\n"

        "When useful, identify the source naturally using "
        "the document name and page number.\n\n"

        "For step-by-step questions, organize the answer "
        "as numbered steps and include commands in code blocks "
        "when appropriate.\n"

        "For comparison questions, use a clear structure "
        "or table when appropriate.\n"

        "For questions about figures, diagrams, or tables, "
        "use the corresponding retrieved information when "
        "available."
    )


    def __init__(
        self,
        rag_prompt: str | None = None,
    ):

        self.rag_prompt = (
            rag_prompt.strip()
            if rag_prompt
            else self.DEFAULT_RAG_PROMPT
        )


    # ========================================================
    # Build RAG Instructions
    # ========================================================

    def build(self) -> str:

        return self.rag_prompt