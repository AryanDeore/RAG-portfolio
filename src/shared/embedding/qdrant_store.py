"""
Qdrant client wrapper to manage collection lifecycle, upserts, and ANN search.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter
from configs.settings import settings

DIST = {"Cosine": Distance.COSINE, "Dot": Distance.DOT, "Euclid": Distance.EUCLID}

class QdrantStore:
    def __init__(self):
        # Convert qdrant_url to string (handles both AnyHttpUrl field and str property)
        url = str(settings.qdrant_url)
        self.client = QdrantClient(
            url=url,
            api_key=settings.qdrant_api_key
        )

    def create_or_recreate(self, recreate: bool = False) -> None:
        """
        Create the target collection if it does not exist, or recreate it when requested.

        Args:
            recreate (bool): If True, forcibly recreates the collection (drops existing data).
                             If False, attempts creation and ignores 'already exists' errors.

        Returns:
            None
        """
        params = VectorParams(size=settings.embed_dim, distance=DIST[settings.embed_metric])
        if recreate:
            self.client.recreate_collection(collection_name=settings.embed_collection, vectors_config=params)
        else:
            try:
                self.client.create_collection(collection_name=settings.embed_collection, vectors_config=params)
            except Exception:
                # Likely already exists; proceed without failing.
                pass

    def upsert_points(self, points: list[PointStruct]) -> None:
        """
        Upsert (insert or replace) a batch of points into the collection.

        Args:
            points (list[PointStruct]): Vector+payload objects with stable IDs.

        Returns:
            None
        """
        self.client.upsert(collection_name=settings.embed_collection, points=points)

    def search(self, query_vec: list[float], k: int, filt: Filter | None = None):
        """
        Execute an ANN search against the collection.

        Args:
            query_vec (list[float]): Dense query vector (must match collection dimension).
            k (int): Maximum number of nearest neighbors to return.
            filt (Filter | None): Optional Qdrant filter to constrain payload-based subsets.

        Returns:
            list[qdrant_client.http.models.ScoredPoint]: Scored hits with payloads and IDs.
        """
        return self.client.search(
            collection_name=settings.embed_collection,
            query_vector=query_vec,
            limit=k,
            query_filter=filt
        )
