from typing import Optional

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)


class QdrantVectorStore:
    """
    Local Qdrant vector store.
    """

    def __init__(
        self,
        path: str,
        collection_name: str,
        vector_size: int = 1024,
    ):

        self.collection_name = (
            collection_name
        )

        self.vector_size = vector_size

        print(
            "Opening Qdrant Local..."
        )

        self.client = QdrantClient(
            path=path
        )

        print(
            "Qdrant Local ready."
        )

    # ========================================================
    # Collection
    # ========================================================

    def collection_exists(self) -> bool:

        collections = (
            self.client.get_collections()
        )

        return any(
            collection.name
            == self.collection_name
            for collection
            in collections.collections
        )

    def create_collection(
        self,
        recreate: bool = False,
    ):

        if (
            recreate
            and self.collection_exists()
        ):

            self.client.delete_collection(
                self.collection_name
            )

        if not self.collection_exists():

            self.client.create_collection(

                collection_name=(
                    self.collection_name
                ),

                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            )

    # ========================================================
    # Insert / Update
    # ========================================================

    def upsert(
        self,
        points: list[PointStruct],
    ):

        if not points:
            return

        self.create_collection(
            recreate=False
        )

        self.client.upsert(

            collection_name=(
                self.collection_name
            ),

            points=points,
        )

    # ========================================================
    # Delete document
    # ========================================================

    def delete_document(
        self,
        document_name: Optional[str] = None,
        document_id: Optional[str] = None,
    ):

        conditions = []

        if document_id:

            conditions.append(
                FieldCondition(
                    key="document_id",
                    match=MatchValue(
                        value=document_id
                    ),
                )
            )

        elif document_name:

            conditions.append(
                FieldCondition(
                    key="document_name",
                    match=MatchValue(
                        value=document_name
                    ),
                )
            )

        if not conditions:
            return

        self.client.delete(

            collection_name=(
                self.collection_name
            ),

            points_selector=Filter(
                must=conditions
            ),
        )

    # ========================================================
    # Count
    # ========================================================

    def count(self) -> int:

        if not self.collection_exists():
            return 0

        result = self.client.count(
            collection_name=(
                self.collection_name
            ),
            exact=True,
        )

        return result.count

    # ========================================================
    # Search
    # ========================================================

    def search(
        self,
        vector: list[float],
        limit: int = 5,
    ):

        if not self.collection_exists():
            return []

        return self.client.query_points(

            collection_name=(
                self.collection_name
            ),

            query=vector,

            limit=limit,

            with_payload=True,

        ).points

    # ========================================================
    # Close
    # ========================================================

    def close(self):

        self.client.close()