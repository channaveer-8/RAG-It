import json
import re

import requests


class RetrievalPlanner:
    """
    Convert a user question into retrieval-oriented
    search queries.

    The planner does NOT answer the user's question.

    Its only responsibility is to identify what information
    should be retrieved from the document collection.

    It is intentionally domain-agnostic and can work with:

        - technical documents
        - legal documents
        - financial documents
        - research papers
        - books
        - reports
        - manuals
        - general documents

    Pipeline:

        User Query
            ↓
        RetrievalPlanner
            ↓
        Search Queries
            ↓
        Embedding
            ↓
        Vector Search
    """

    DEFAULT_MAX_QUERIES = 5

    def __init__(
        self,
        model_name: str = "qwen3.5:4b-q4_K_M",
        base_url: str = "http://localhost:11434",
        max_queries: int = DEFAULT_MAX_QUERIES,
        timeout: int = 120,
    ):

        self.model_name = model_name

        self.base_url = (
            base_url.rstrip("/")
        )

        self.max_queries = max(
            1,
            int(max_queries),
        )

        self.timeout = timeout


    # ========================================================
    # PLAN
    # ========================================================

    def plan(
        self,
        query: str,
    ) -> list[str]:
        """
        Generate retrieval queries for a user question.

        The original user query is always retained as the
        first retrieval query.

        Returns:

            [
                original_query,
                generated_query_1,
                generated_query_2,
                ...
            ]
        """

        query = (
            query.strip()
            if query
            else ""
        )


        if not query:

            return []


        prompt = self._build_prompt(
            query
        )


        payload = {

            "model": (
                self.model_name
            ),

            "prompt": prompt,

            "stream": False,

            # We intentionally do NOT use long-form
            # thinking for query planning.
            "think": False,

            "options": {

                # Deterministic planning.
                "temperature": 0,
            },
        }


        response = requests.post(

            f"{self.base_url}/api/generate",

            json=payload,

            timeout=self.timeout,
        )


        response.raise_for_status()


        data = response.json()


        raw_response = (
            data.get(
                "response",
                "",
            )
            or ""
        ).strip()


        generated_queries = (
            self._parse_response(
                raw_response
            )
        )


        # ----------------------------------------------------
        # Always preserve the original query.
        #
        # The planner is an aid to retrieval, not the
        # authority. This protects against planner mistakes.
        # ----------------------------------------------------

        queries = [

            query,

            *generated_queries,
        ]


        queries = self._deduplicate(
            queries
        )


        # max_queries means the total number of searches,
        # including the original query.
        return queries[
            :self.max_queries
        ]


    # ========================================================
    # PROMPT
    # ========================================================

    def _build_prompt(
        self,
        query: str,
    ) -> str:
        """
        Build a domain-agnostic planning prompt.
        """

        return f"""
You are a retrieval query planner for a document
question-answering system.

Your task is NOT to answer the user's question.

Your task is to identify the information that should
be retrieved from the available documents before another
model answers the question.

Break complex questions into their important retrieval
requirements.

Adapt the retrieval queries to the subject matter of
the user's question.

Consider, when relevant:

- important entities and subjects
- people, organizations, products, places, or objects
- concepts and topics
- events, dates, and time periods
- properties, attributes, and values
- relationships between entities
- actions or procedures
- comparisons or distinctions
- causes and effects
- definitions
- requirements, constraints, or conditions
- examples
- domain-specific terminology

For procedural questions, retrieve the information
needed for the complete procedure.

For comparison questions, retrieve the information
about each side of the comparison.

For definition questions, retrieve the relevant
definitions and terminology.

For questions involving multiple entities, retrieve
information about each important entity and their
relationship.

Do not assume that the documents are technical.

Do not assume a particular document type or domain.

Do not invent facts.

Do not answer the question.

Generate up to {self.max_queries} concise retrieval
queries.

Return ONLY valid JSON in exactly this format:

{{
  "queries": [
    "search query 1",
    "search query 2",
    "search query 3"
  ]
}}

User question:

{query}
""".strip()


    # ========================================================
    # PARSE RESPONSE
    # ========================================================

    @staticmethod
    def _parse_response(
        response: str,
    ) -> list[str]:
        """
        Parse the planner's response.

        Preferred format:

            {
                "queries": [
                    "...",
                    "..."
                ]
            }

        Includes fallbacks because local models may sometimes
        wrap JSON in markdown or produce simple numbered lists.
        """

        if not response:

            return []


        # ----------------------------------------------------
        # 1. Direct JSON
        # ----------------------------------------------------

        try:

            data = json.loads(
                response
            )

            queries = (
                data.get(
                    "queries",
                    [],
                )
            )

            if isinstance(
                queries,
                list,
            ):

                return RetrievalPlanner._clean_queries(
                    queries
                )

        except (
            json.JSONDecodeError,
            TypeError,
            AttributeError,
        ):

            pass


        # ----------------------------------------------------
        # 2. JSON inside markdown / extra text
        # ----------------------------------------------------

        match = re.search(
            r"\{[\s\S]*\}",
            response,
        )


        if match:

            json_text = (
                match.group(0)
            )


            try:

                data = json.loads(
                    json_text
                )

                queries = (
                    data.get(
                        "queries",
                        [],
                    )
                )


                if isinstance(
                    queries,
                    list,
                ):

                    return (
                        RetrievalPlanner._clean_queries(
                            queries
                        )
                    )

            except (
                json.JSONDecodeError,
                TypeError,
                AttributeError,
            ):

                pass


        # ----------------------------------------------------
        # 3. Line-based fallback
        # ----------------------------------------------------

        queries = []


        for line in response.splitlines():

            line = line.strip()


            if not line:

                continue


            # Remove common numbering:

            line = re.sub(
                r"^\s*(?:[-*]|\d+[.)])\s*",
                "",
                line,
            )


            line = line.strip(
                "\"' "
            )


            if line:

                queries.append(
                    line
                )


        return RetrievalPlanner._clean_queries(
            queries
        )


    # ========================================================
    # CLEAN QUERIES
    # ========================================================

    @staticmethod
    def _clean_queries(
        queries,
    ) -> list[str]:
        """
        Normalize query values returned by the model.
        """

        cleaned = []


        for query in queries:

            if query is None:

                continue


            query = str(
                query
            ).strip()


            if not query:

                continue


            # Remove accidental surrounding quotes.
            query = query.strip(
                "\"' "
            )


            if query:

                cleaned.append(
                    query
                )


        return cleaned


    # ========================================================
    # DEDUPLICATE
    # ========================================================

    @staticmethod
    def _deduplicate(
        queries: list[str],
    ) -> list[str]:
        """
        Remove duplicate queries while preserving order.
        """

        result = []

        seen = set()


        for query in queries:

            query = (
                query.strip()
                if query
                else ""
            )


            if not query:

                continue


            normalized = " ".join(
                query.lower().split()
            )


            if normalized in seen:

                continue


            seen.add(
                normalized
            )


            result.append(
                query
            )


        return result