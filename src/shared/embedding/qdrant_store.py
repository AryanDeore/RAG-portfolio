"""
Qdrant client wrapper to manage collection lifecycle, upserts, and ANN search.
"""

import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    Modifier,
    PointStruct,
    Filter,
    NamedVector,
    NamedSparseVector,
    SparseVector,
    Prefetch,
    FusionQuery,
    Fusion,
)
from configs.settings import settings

logger = logging.getLogger(__name__)
DIST = {"Cosine": Distance.COSINE, "Dot": Distance.DOT, "Euclid": Distance.EUCLID}

class QdrantStore:
    def __init__(self):
        # Convert qdrant_url to string (handles both AnyHttpUrl field and str property)
        url = str(settings.qdrant_url)
        logger.debug(f"Connecting to Qdrant at: {url}")
        logger.debug(f"Collection name: {settings.embed_collection}")
        self.client = QdrantClient(
            url=url,
            api_key=settings.qdrant_api_key or None
        )

    def create_or_recreate(self, recreate: bool = False) -> None:
        """
        Create the target collection with named dense + sparse vector config.

        Args:
            recreate (bool): If True, forcibly recreates the collection (drops existing data).
        """
        vectors_config = {
            "dense": VectorParams(size=settings.embed_dim, distance=DIST[settings.embed_metric])
        }
        sparse_vectors_config = {
            "sparse": SparseVectorParams(modifier=Modifier.IDF)
        }

        if recreate:
            if self.client.collection_exists(collection_name=settings.embed_collection):
                logger.info(f"Deleting existing collection: {settings.embed_collection}")
                self.client.delete_collection(collection_name=settings.embed_collection)
            self.client.create_collection(
                collection_name=settings.embed_collection,
                vectors_config=vectors_config,
                sparse_vectors_config=sparse_vectors_config,
            )
        else:
            try:
                self.client.create_collection(
                    collection_name=settings.embed_collection,
                    vectors_config=vectors_config,
                    sparse_vectors_config=sparse_vectors_config,
                )
            except Exception:
                pass

    def upsert_points(self, points: list[PointStruct]) -> None:
        """
        Upsert (insert or replace) a batch of points into the collection.

        Args:
            points (list[PointStruct]): Vector+payload objects with stable IDs.
        """
        self.client.upsert(collection_name=settings.embed_collection, points=points)

    def search(
        self,
        query_vec: list[float],
        k: int,
        sparse_vec: tuple[list[int], list[float]] | None = None,
        filt: Filter | None = None,
    ):
        """
        Execute search against the collection — hybrid (dense+BM25+RRF) when
        sparse_vec is provided, dense-only fallback otherwise.

        Args:
            query_vec (list[float]): Dense query vector.
            k (int): Maximum number of results to return.
            sparse_vec: Optional (indices, values) tuple for BM25 sparse query.
            filt (Filter | None): Optional Qdrant filter.

        Returns:
            list[ScoredPoint]: Scored hits with payloads and IDs.
        """
        collection = settings.embed_collection_read or settings.embed_collection
        logger.debug(f"Searching collection '{collection}' with k={k}, vector_dim={len(query_vec)}")

        try:
            exists = self.client.collection_exists(collection)
            if not exists:
                error_msg = f"Collection '{collection}' does not exist"
                logger.error(error_msg)
                raise ValueError(error_msg)

            if sparse_vec is None:
                # Dense-only fallback using named vector
                result = self.client.query_points(
                    collection_name=collection,
                    query=query_vec,
                    using="dense",
                    limit=k,
                    query_filter=filt,
                )
                logger.debug(f"Dense-only search returned {len(result.points)} results")
                return result.points

            # Hybrid search: dense + BM25 with RRF fusion
            sparse_qvec = SparseVector(indices=sparse_vec[0], values=sparse_vec[1])
            result = self.client.query_points(
                collection_name=collection,
                prefetch=[
                    Prefetch(query=query_vec, using="dense", limit=k * 2, filter=filt),
                    Prefetch(query=sparse_qvec, using="sparse", limit=k * 2, filter=filt),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=k,
            )
            logger.debug(f"Hybrid search returned {len(result.points)} results")
            return result.points

        except Exception as e:
            logger.error(f"Search error: {type(e).__name__}: {e}")
            raise
