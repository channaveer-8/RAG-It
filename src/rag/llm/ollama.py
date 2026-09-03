import json
import requests


class OllamaLLM:
    """
    Local Ollama LLM adapter.

    The final prompt is assembled here.
    Streaming tokens can optionally be sent
    to a callback for UI applications.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are a helpful RAG assistant."
    )

    DEFAULT_RAG_PROMPT = (
        "Answer the user's question using only "
        "the provided context.\n"
        "If the answer cannot be found in the context, "
        "say that there is not enough information "
        "in the provided documents.\n"
        "Do not invent facts."
    )

    def __init__(
        self,
        model_name: str = "qwen3.5:4b-q4_K_M",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.2,
        system_prompt: str | None = None,
    ):

        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self.temperature = temperature

        self.system_prompt = (
            system_prompt.strip()
            if system_prompt
            else self.DEFAULT_SYSTEM_PROMPT
        )

    # ========================================================
    # Generate
    # ========================================================

    def generate(
        self,
        query: str,
        context: str,
        rag_prompt: str | None = None,
        stream: bool = True,
        on_token=None,
    ) -> str:

        prompt = self._build_prompt(
            query=query,
            context=context,
            rag_prompt=rag_prompt,
        )

        response = requests.post(
            f"{self.base_url}/api/generate",

            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": self.temperature,
                },
            },

            stream=stream,

            timeout=300,
        )

        response.raise_for_status()

        # ====================================================
        # Streaming
        # ====================================================

        if stream:

            full_response = []

            for line in response.iter_lines():

                if not line:
                    continue

                data = json.loads(
                    line.decode("utf-8")
                )

                token = data.get(
                    "response",
                    "",
                )

                if token:

                    full_response.append(
                        token
                    )

                    if on_token is not None:

                        on_token(token)

                    else:

                        print(
                            token,
                            end="",
                            flush=True,
                        )

                if data.get(
                    "done",
                    False,
                ):

                    break

            if on_token is None:
                print()

            return "".join(
                full_response
            ).strip()

        # ====================================================
        # Non-streaming
        # ====================================================

        data = response.json()

        return data.get(
            "response",
            "",
        ).strip()

    # ========================================================
    # Final Prompt
    # ========================================================

    def _build_prompt(
        self,
        query: str,
        context: str,
        rag_prompt: str | None = None,
    ) -> str:

        query = query.strip()
        context = context.strip()

        if not query:

            raise ValueError(
                "Query cannot be empty."
            )

        rag_instructions = (
            rag_prompt.strip()
            if rag_prompt
            else self.DEFAULT_RAG_PROMPT
        )

        return f"""System Instructions:
{self.system_prompt}

RAG Instructions:
{rag_instructions}

Retrieved Context:
--------------------
{context}
--------------------

User Question:
{query}

Answer:
"""