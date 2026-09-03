import json
import sys
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    Signal,
    Slot,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parents[1]
)

SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(
    0,
    str(SRC_DIR)
)


# ============================================================
# RAG components
# ============================================================

from rag.config import (
    DOCUMENTS_DIR,
    QDRANT_DIR,
    COLLECTION_NAME,
    BGE_MODEL_NAME,
    BGE_MODEL_WEIGHT,
    EMBEDDING_DIMENSION,
    DEVICE,
)

from rag.embedding.visual_bge import (
    VisualBGEEmbedder,
)

from rag.vectorstore.qdrant import (
    QdrantVectorStore,
)

from rag.context import (
    ContextBuilder,
)

from rag.prompt import (
    PromptBuilder,
)

from rag.llm import (
    OllamaLLM,
)

from rag.llm.ollama_manager import (
    OllamaManager,
)

from rag.ingestion.indexer import (
    DocumentIndexer,
)

from rag.query.planner import (
    RetrievalPlanner,
)


class RAGWorker(QObject):

    # ========================================================
    # Signals
    # ========================================================

    ready = Signal()

    statusChanged = Signal(str)

    errorOccurred = Signal(str)

    answerStarted = Signal()

    answerToken = Signal(str)

    answerFinished = Signal()

    sourcesChanged = Signal(str)

    imagesChanged = Signal(str)

    tablesChanged = Signal(str)

    documentsChanged = Signal(str)


    # ========================================================
    # Initialization
    # ========================================================

    def __init__(self):

        super().__init__()

        self.embedder = None
        self.vectorstore = None
        self.context_builder = None
        self.llm = None
        self.ollama_manager = None
        self.indexer = None
        self.retrieval_planner = None

        self.busy = False
        self.initialized = False


    # ========================================================
    # Indexer progress
    # ========================================================

    def _index_progress(
        self,
        message: str,
    ):

        self.statusChanged.emit(
            message
        )


    # ========================================================
    # Initialize
    # ========================================================

    @Slot()
    def initialize(self):

        if self.initialized:
            return


        try:

            # ====================================================
            # Ollama
            # ====================================================

            self.statusChanged.emit(
                "Checking Ollama..."
            )


            self.ollama_manager = (
                OllamaManager(

                    model_name=(
                        "qwen3.5:4b-q4_K_M"
                    ),

                    base_url=(
                        "http://localhost:11434"
                    ),
                )
            )


            self.ollama_manager.ensure_ready()


            # ====================================================
            # Visualized-BGE
            # ====================================================

            self.statusChanged.emit(
                "Loading Visualized-BGE..."
            )


            self.embedder = (
                VisualBGEEmbedder(

                    model_name=(
                        BGE_MODEL_NAME
                    ),

                    model_weight=str(
                        BGE_MODEL_WEIGHT
                    ),

                    device=DEVICE,
                )
            )


            # ====================================================
            # Qdrant
            # ====================================================

            self.statusChanged.emit(
                "Connecting to Qdrant..."
            )


            self.vectorstore = (
                QdrantVectorStore(

                    path=str(
                        QDRANT_DIR
                    ),

                    collection_name=(
                        COLLECTION_NAME
                    ),

                    vector_size=(
                        EMBEDDING_DIMENSION
                    ),
                )
            )


            self.vectorstore.create_collection(
                recreate=False
            )


            # ====================================================
            # Indexer
            # ====================================================

            self.statusChanged.emit(
                "Initializing ingestion..."
            )


            self.indexer = (
                DocumentIndexer(
                    progress_callback=(
                        self._index_progress
                    )
                )
            )


            # ====================================================
            # Retrieval planner
            # ====================================================

            self.statusChanged.emit(
                "Initializing retrieval planner..."
            )


            self.retrieval_planner = (
                RetrievalPlanner(

                    model_name=(
                        "qwen3.5:4b-q4_K_M"
                    ),

                    base_url=(
                        "http://localhost:11434"
                    ),

                    max_queries=5,

                    timeout=120,
                )
            )


            # ====================================================
            # Context builder
            # ====================================================

            self.context_builder = (
                ContextBuilder(
                    max_chunks=10
                )
            )


            # ====================================================
            # Final LLM
            # ====================================================

            self.statusChanged.emit(
                "Connecting to Ollama..."
            )


            self.llm = (
                OllamaLLM(

                    model_name=(
                        "qwen3.5:4b-q4_K_M"
                    ),

                    base_url=(
                        "http://localhost:11434"
                    ),

                    temperature=0.2,
                )
            )


            # ====================================================
            # Documents
            # ====================================================

            self._load_documents()


            self.initialized = True


            self.statusChanged.emit(
                "Ready"
            )

            self.ready.emit()


        except Exception as exc:

            print(
                f"[RAG] Initialization error: "
                f"{exc}",
                flush=True,
            )


            self.errorOccurred.emit(
                str(exc)
            )


            self.statusChanged.emit(
                "Error"
            )


    # ========================================================
    # Documents
    # ========================================================

    def _load_documents(self):

        extensions = {
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".html",
            ".md",
            ".txt",
        }


        documents = []


        DOCUMENTS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )


        for path in sorted(
            DOCUMENTS_DIR.iterdir(),
            key=lambda item:
                item.name.lower(),
        ):

            if (
                path.is_file()
                and path.suffix.lower()
                in extensions
            ):

                documents.append(
                    path.name
                )


        self.documentsChanged.emit(
            json.dumps(
                documents,
                ensure_ascii=False,
            )
        )


    # ========================================================
    # Add document
    # ========================================================

    @Slot(str)
    def add_document(
        self,
        source_path: str,
    ):

        if self.busy:
            return


        if not self.initialized:
            return


        source = Path(
            source_path
        )


        if not source.exists():

            self.errorOccurred.emit(
                f"File not found: {source}"
            )

            return


        self.busy = True


        try:

            self.statusChanged.emit(
                "Adding document..."
            )


            destination = (
                DOCUMENTS_DIR
                / source.name
            )


            # ------------------------------------------------
            # Replace existing document
            # ------------------------------------------------

            if destination.exists():

                self.statusChanged.emit(
                    "Replacing existing document..."
                )


                self.indexer.remove_document(

                    document_name=(
                        destination.name
                    ),

                    vectorstore=(
                        self.vectorstore
                    ),
                )


                destination.unlink(
                    missing_ok=True
                )


            destination.parent.mkdir(
                parents=True,
                exist_ok=True,
            )


            # ------------------------------------------------
            # Copy
            # ------------------------------------------------

            import shutil

            shutil.copy2(
                source,
                destination,
            )


            # ------------------------------------------------
            # Index
            # ------------------------------------------------

            self.statusChanged.emit(
                "Parsing and indexing..."
            )


            stats = (
                self.indexer.index_document(

                    source=destination,

                    embedder=self.embedder,

                    vectorstore=self.vectorstore,
                )
            )


            self._load_documents()


            self.statusChanged.emit(
                (
                    f"Indexed "
                    f"{stats['document_name']} "
                    f"({stats['vectors']} vectors)"
                )
            )


            self.statusChanged.emit(
                "Ready"
            )


        except Exception as exc:

            print(
                f"[RAG] Add document error: "
                f"{exc}",
                flush=True,
            )


            self.errorOccurred.emit(
                str(exc)
            )


            self.statusChanged.emit(
                "Error"
            )


        finally:

            self.busy = False


    # ========================================================
    # Remove document
    # ========================================================

    @Slot(str)
    def remove_document(
        self,
        document_name: str,
    ):

        if self.busy:
            return


        if not self.initialized:
            return


        if not document_name:
            return


        self.busy = True


        try:

            self.statusChanged.emit(
                "Removing document..."
            )


            path = (
                DOCUMENTS_DIR
                / document_name
            )


            self.indexer.remove_document(

                document_name=(
                    document_name
                ),

                vectorstore=(
                    self.vectorstore
                ),
            )


            path.unlink(
                missing_ok=True
            )


            self._load_documents()


            self.statusChanged.emit(
                "Ready"
            )


        except Exception as exc:

            print(
                f"[RAG] Remove document error: "
                f"{exc}",
                flush=True,
            )


            self.errorOccurred.emit(
                str(exc)
            )


            self.statusChanged.emit(
                "Error"
            )


        finally:

            self.busy = False


    # ========================================================
    # Ask
    # ========================================================

    @Slot(str, str)
    def ask(
        self,
        query: str,
        rag_prompt: str,
    ):

        if self.busy:
            return


        if not self.initialized:
            return


        query = query.strip()


        if not query:
            return


        self.busy = True


        try:

            # ====================================================
            # Clear previous result panels
            # ====================================================

            self.imagesChanged.emit(
                json.dumps([])
            )


            self.tablesChanged.emit(
                json.dumps([])
            )


            self.sourcesChanged.emit(
                json.dumps([])
            )


            # ====================================================
            # Query complexity
            # ====================================================

            complexity = (
                self._estimate_query_complexity(
                    query
                )
            )


            if complexity == "complex":

                max_context_chunks = 10
                retrieval_limit = 10

            else:

                max_context_chunks = 5
                retrieval_limit = 6


            self.context_builder.max_chunks = (
                max_context_chunks
            )


            print(
                "\n" + "=" * 80,
                flush=True,
            )


            print(
                "QUERY ANALYSIS",
                flush=True,
            )


            print(
                f"Complexity: {complexity}",
                flush=True,
            )


            print(
                f"Context chunks: "
                f"{max_context_chunks}",
                flush=True,
            )


            print(
                f"Retrieval limit/query: "
                f"{retrieval_limit}",
                flush=True,
            )


            print(
                "=" * 80,
                flush=True,
            )


            # ====================================================
            # Retrieval planning
            # ====================================================

            self.statusChanged.emit(
                "Planning retrieval..."
            )


            retrieval_queries = (
                self.retrieval_planner.plan(
                    query
                )
            )


            if not retrieval_queries:

                retrieval_queries = [
                    query
                ]


            print(
                "\n" + "=" * 80,
                flush=True,
            )


            print(
                "RETRIEVAL PLANNER",
                flush=True,
            )


            print(
                "=" * 80,
                flush=True,
            )


            for index, search_query in enumerate(
                retrieval_queries,
                start=1,
            ):

                print(
                    f"{index}. "
                    f"{search_query}",
                    flush=True,
                )


            print(
                "=" * 80,
                flush=True,
            )


            # ====================================================
            # Multi-query retrieval
            # ====================================================

            self.statusChanged.emit(
                "Retrieving..."
            )


            # Each query gets its own ranked list.
            retrieval_lists = []


            for query_index, search_query in enumerate(
                retrieval_queries,
                start=1,
            ):

                print(
                    "\n" + "-" * 70,
                    flush=True,
                )


                print(
                    (
                        f"[RETRIEVAL] Query "
                        f"{query_index}/"
                        f"{len(retrieval_queries)}"
                    ),
                    flush=True,
                )


                print(
                    search_query,
                    flush=True,
                )


                # ------------------------------------------------
                # Embed query
                # ------------------------------------------------

                embedding = (
                    self.embedder.encode_text(
                        search_query
                    )
                )


                vector = (
                    self.embedder.to_numpy(
                        embedding
                    )
                    .flatten()
                    .tolist()
                )


                # ------------------------------------------------
                # Search
                # ------------------------------------------------

                query_results = (
                    self.vectorstore.search(

                        vector=vector,

                        limit=retrieval_limit,
                    )
                )


                print(
                    (
                        f"[RETRIEVAL] "
                        f"{len(query_results)} "
                        f"results"
                    ),
                    flush=True,
                )


                retrieval_lists.append(
                    query_results
                )


                for rank, result in enumerate(
                    query_results,
                    start=1,
                ):

                    payload = (
                        result.payload
                        or {}
                    )


                    print(
                        (
                            f"    {rank}. "
                            f"{payload.get('document_name')} "
                            f"| "
                            f"{payload.get('chunk_id')} "
                            f"| page="
                            f"{payload.get('page_numbers')} "
                            f"| type="
                            f"{payload.get('chunk_type')} "
                            f"| score="
                            f"{float(result.score):.4f}"
                        ),
                        flush=True,
                    )


            # ====================================================
            # Reciprocal Rank Fusion
            # ====================================================

            self.statusChanged.emit(
                "Combining retrieval results..."
            )


            fused_results = (
                self._reciprocal_rank_fusion(
                    retrieval_lists
                )
            )


            print(
                "\n" + "=" * 80,
                flush=True,
            )


            print(
                "RRF FUSED RESULTS",
                flush=True,
            )


            print(
                "=" * 80,
                flush=True,
            )


            print(
                f"Unique candidates: "
                f"{len(fused_results)}",
                flush=True,
            )


            # ----------------------------------------------------
            # Final candidate count
            # ----------------------------------------------------

            final_limit = (
                max_context_chunks * 2
            )


            results = fused_results[
                :final_limit
            ]


            for rank, result_info in enumerate(
                results,
                start=1,
            ):

                result = (
                    result_info["result"]
                )


                payload = (
                    result.payload
                    or {}
                )


                print(
                    (
                        f"{rank}. "
                        f"{payload.get('document_name')} "
                        f"| "
                        f"{payload.get('chunk_id')} "
                        f"| page="
                        f"{payload.get('page_numbers')} "
                        f"| type="
                        f"{payload.get('chunk_type')} "
                        f"| "
                        f"RRF="
                        f"{result_info['rrf_score']:.6f}"
                    ),
                    flush=True,
                )


            print(
                "=" * 80,
                flush=True,
            )


            # ====================================================
            # Convert fused results back to Qdrant results
            # ====================================================

            qdrant_results = [
                item["result"]
                for item in results
            ]


            # ====================================================
            # Sources / images / tables
            # ====================================================

            sources = []

            images = []

            tables = []


            for result_info in results:

                result = (
                    result_info["result"]
                )


                payload = (
                    result.payload
                    or {}
                )


                document_name = (
                    payload.get(
                        "document_name",
                        "Unknown",
                    )
                )


                pages = (
                    payload.get(
                        "page_numbers",
                        payload.get(
                            "pages",
                            [],
                        ),
                    )
                )


                chunk_type = (
                    payload.get(
                        "chunk_type",
                        "unknown",
                    )
                )


                content = (
                    payload.get(
                        "content",
                        "",
                    )
                    or ""
                )


                asset_path = (
                    payload.get(
                        "asset_path"
                    )
                )


                # ------------------------------------------------
                # Source
                # ------------------------------------------------

                sources.append(
                    {
                        "document":
                            document_name,

                        "page":
                            pages,

                        "type":
                            chunk_type,

                        "score":
                            round(
                                float(
                                    result.score
                                ),
                                4,
                            ),

                        "rrf_score":
                            round(
                                result_info[
                                    "rrf_score"
                                ],
                                6,
                            ),

                        "chunk_id":
                            payload.get(
                                "chunk_id",
                                "",
                            ),

                        "content":
                            content,

                        "asset_path":
                            asset_path or "",
                    }
                )


                # ------------------------------------------------
                # Image
                # ------------------------------------------------

                if (
                    chunk_type == "image"
                    and asset_path
                ):

                    image_path = Path(
                        asset_path
                    )


                    if image_path.exists():

                        images.append(
                            {
                                "path":
                                    image_path.as_uri(),

                                "document":
                                    document_name,

                                "page":
                                    pages,

                                "score":
                                    round(
                                        result_info[
                                            "rrf_score"
                                        ],
                                        6,
                                    ),
                            }
                        )


                # ------------------------------------------------
                # Table
                # ------------------------------------------------

                if (
                    chunk_type == "table"
                ):

                    tables.append(
                        {
                            "document":
                                document_name,

                            "page":
                                pages,

                            "content":
                                content,

                            "score":
                                round(
                                    result_info[
                                        "rrf_score"
                                    ],
                                    6,
                                ),
                        }
                    )


            self.sourcesChanged.emit(
                json.dumps(
                    sources,
                    ensure_ascii=False,
                )
            )


            self.imagesChanged.emit(
                json.dumps(
                    images,
                    ensure_ascii=False,
                )
            )


            self.tablesChanged.emit(
                json.dumps(
                    tables,
                    ensure_ascii=False,
                )
            )


            # ====================================================
            # Context
            # ====================================================

            self.statusChanged.emit(
                "Building context..."
            )


            context = (
                self.context_builder.build(
                    qdrant_results
                )
            )


            # ====================================================
            # Context diagnostic
            # ====================================================

            print(
                "\n" + "=" * 80,
                flush=True,
            )


            print(
                "FINAL CONTEXT",
                flush=True,
            )


            print(
                "=" * 80,
                flush=True,
            )


            print(
                context,
                flush=True,
            )


            print(
                "=" * 80,
                flush=True,
            )


            # ====================================================
            # No retrieval
            # ====================================================

            if not context.strip():

                self.answerToken.emit(
                    "I don't have enough "
                    "information in the provided "
                    "documents."
                )


                self.answerFinished.emit()


                self.statusChanged.emit(
                    "Ready"
                )


                return


            # ====================================================
            # RAG prompt
            # ====================================================

            prompt_builder = (
                PromptBuilder(
                    rag_prompt=rag_prompt
                )
            )


            rag_instructions = (
                prompt_builder.build()
            )


            # ====================================================
            # Generate
            # ====================================================

            self.statusChanged.emit(
                "Generating..."
            )


            self.answerStarted.emit()


            def on_token(
                token: str,
            ):

                if token:

                    self.answerToken.emit(
                        token
                    )


            print(
                "\n" + "=" * 80,
                flush=True,
            )


            print(
                "FINAL GENERATION",
                flush=True,
            )


            print(
                "=" * 80,
                flush=True,
            )


            answer = (
                self.llm.generate(

                    query=query,

                    context=context,

                    rag_prompt=(
                        rag_instructions
                    ),

                    stream=True,

                    on_token=on_token,
                )
            )


            # ====================================================
            # Empty answer protection
            # ====================================================

            if (
                not answer
                or not answer.strip()
            ):

                self.answerToken.emit(
                    "I don't have enough "
                    "information in the provided "
                    "documents."
                )


            self.answerFinished.emit()


            self.statusChanged.emit(
                "Ready"
            )


        except Exception as exc:

            print(
                f"[RAG] Ask error: "
                f"{exc}",
                flush=True,
            )


            self.errorOccurred.emit(
                str(exc)
            )


            self.statusChanged.emit(
                "Error"
            )


        finally:

            self.busy = False


    # ========================================================
    # Query Complexity
    # ========================================================

    @staticmethod
    def _estimate_query_complexity(
        query: str,
    ) -> str:
        """
        Lightweight heuristic.

        This avoids another LLM call just to classify
        query complexity.
        """

        query_lower = (
            query.lower()
        )


        words = query_lower.split()


        # ----------------------------------------------------
        # Strong indicators of a multi-part question.
        # ----------------------------------------------------

        indicators = [

            "step by step",

            "step-by-step",

            "steps",

            "how do i",

            "how to",

            "explain",

            "compare",

            "difference between",

            "distinguish",

            "why",

            "advantages",

            "disadvantages",

            "pros and cons",

            "procedure",

            "process",

            "configure",

            "debug",

            "troubleshoot",

            "with examples",

            "give examples",

            "and ",
        ]


        indicator_count = sum(
            1
            for indicator in indicators
            if indicator in query_lower
        )


        if (
            len(words) >= 20
            or indicator_count >= 2
        ):

            return "complex"


        return "simple"


    # ========================================================
    # Reciprocal Rank Fusion
    # ========================================================

    @staticmethod
    def _reciprocal_rank_fusion(
        retrieval_lists,
        k: int = 60,
    ):
        """
        Combine multiple ranked retrieval lists using
        Reciprocal Rank Fusion.

        RRF score:

            score = sum(
                1 / (k + rank)
            )

        Rank starts at 1.

        This is preferable to directly comparing Qdrant
        similarity scores produced by different query
        embeddings.
        """

        fused = {}


        for result_list in retrieval_lists:

            for rank, result in enumerate(
                result_list,
                start=1,
            ):

                payload = (
                    result.payload
                    or {}
                )


                chunk_id = (
                    payload.get(
                        "chunk_id"
                    )
                )


                if chunk_id:

                    key = str(
                        chunk_id
                    )

                else:

                    key = (
                        f"{payload.get('document_name')}|"
                        f"{payload.get('page_numbers')}|"
                        f"{payload.get('content', '')}"
                    )


                if key not in fused:

                    fused[key] = {

                        "result":
                            result,

                        "rrf_score":
                            0.0,

                        "appearances":
                            0,

                    }


                fused[key][
                    "rrf_score"
                ] += (
                    1.0
                    / (
                        k + rank
                    )
                )


                fused[key][
                    "appearances"
                ] += 1


                # Keep the highest original similarity
                # for diagnostics/source display.
                existing_result = (
                    fused[key]["result"]
                )


                if (
                    float(
                        result.score
                    )
                    >
                    float(
                        existing_result.score
                    )
                ):

                    fused[key][
                        "result"
                    ] = result


        ordered = sorted(

            fused.values(),

            key=lambda item: (
                item["rrf_score"],
                item["appearances"],
            ),

            reverse=True,
        )


        return ordered


    # ========================================================
    # Cleanup
    # ========================================================

    @Slot()
    def shutdown(self):

        self.initialized = False

        self.busy = False


        if self.vectorstore is not None:

            self.vectorstore.close()

            self.vectorstore = None


        self.embedder = None

        self.context_builder = None

        self.llm = None

        self.indexer = None

        self.ollama_manager = None

        self.retrieval_planner = None