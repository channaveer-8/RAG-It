import json
from typing import List

import requests


class QueryPlanner:
    """
    Convert a user's question into multiple retrieval queries
    using Ollama.

    This component does NOT:
        - search Qdrant
        - generate embeddings
        - generate the final answer

    Its only responsibility is query decomposition.
    """

    DEFAULT_MODEL = "qwen3.5:4b-q4_K_M"
    DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"

    SYSTEM_INSTRUCTION = """
You are a retrieval query planner for a document-based RAG system.

Your task is to break the user's question into multiple
independent search queries that can be used to retrieve
relevant information from documents.

Do NOT answer the user's question.

Return ONLY valid JSON using exactly this format:

{
  "queries": [
    "query 1",
    "query 2",
    "query 3"
  ]
}

Rules:

1. Create multiple focused retrieval queries when the question
   contains multiple information requirements.

2. Preserve important entities and terminology from the user.

3. Include important information such as applicable:
   - people
   - organizations
   - hardware
   - software
   - tools
   - protocols
   - commands
   - procedures
   - concepts
   - actions
   - error messages
   - technical terms
   - dates
   - locations
   - definitions
   - relationships

4. Do not invent entities, commands, facts, or terminology.

5. For technical questions, separate different technical
   requirements into different queries.

6. For non-technical questions, identify the important concepts
   appropriate to that domain.

7. Each query should be useful as an independent document search.

8. Avoid duplicate or nearly identical queries.

9. Do not include explanations outside the JSON.

10. Usually generate between 3 and 8 queries.

11. If the question is already very simple, generate 2 to 3
    focused queries rather than unnecessarily expanding it.
"""

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        ollama_url: str = DEFAULT_OLLAMA_URL,
        timeout: int = 120,
    ):
        self.model = model
        self.ollama_url = ollama_url
        self.timeout = timeout

    # ========================================================
    # PLAN
    # ========================================================

    def plan(
        self,
        question: str,
    ) -> List[str]:

        question = question.strip()

        if not question:
            return []

        prompt = (
            self.SYSTEM_INSTRUCTION
            + "\n\n"
            + "USER QUESTION:\n"
            + question
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,

            # Ask Ollama to use its thinking capability.
            "think": True,
        }

        print(
            "\n[QUERY PLANNER] Sending question to Ollama...",
            flush=True,
        )

        response = requests.post(
            self.ollama_url,
            json=payload,
            timeout=self.timeout,
        )

        response.raise_for_status()

        data = response.json()

        # ----------------------------------------------------
        # Ollama's normal generated response
        # ----------------------------------------------------

        raw_response = (
            data.get("response")
            or ""
        ).strip()

        if not raw_response:

            raise RuntimeError(
                "Ollama returned an empty response "
                "from the query planner."
            )

        print(
            "[QUERY PLANNER] Ollama response received.",
            flush=True,
        )

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        queries = self._parse_queries(
            raw_response
        )

        if not queries:

            raise RuntimeError(
                "Query planner returned no retrieval queries.\n"
                f"Raw response:\n{raw_response}"
            )

        return queries

    # ========================================================
    # PARSE
    # ========================================================

    @staticmethod
    def _parse_queries(
        response: str,
    ) -> List[str]:

        text = response.strip()

        # ----------------------------------------------------
        # Remove markdown code fences if the model adds them.
        # ----------------------------------------------------

        if text.startswith("```"):

            lines = text.splitlines()

            lines = [
                line
                for line in lines
                if not line.strip().startswith("```")
            ]

            text = "\n".join(lines).strip()

        # ----------------------------------------------------
        # Parse JSON
        # ----------------------------------------------------

        try:

            data = json.loads(
                text
            )

        except json.JSONDecodeError as exc:

            raise RuntimeError(
                "Query planner did not return valid JSON.\n"
                f"Response:\n{text}"
            ) from exc

        queries = data.get(
            "queries"
        )

        if not isinstance(
            queries,
            list,
        ):

            raise RuntimeError(
                "Query planner JSON does not contain "
                "a 'queries' list."
            )

        # ----------------------------------------------------
        # Clean queries
        # ----------------------------------------------------

        cleaned = []

        for query in queries:

            if not isinstance(
                query,
                str,
            ):
                continue

            query = query.strip()

            if not query:
                continue

            if query not in cleaned:

                cleaned.append(
                    query
                )

        return cleaned