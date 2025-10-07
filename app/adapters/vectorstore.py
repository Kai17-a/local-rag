from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

QDRANT_COLLECTION_NAME = Config.get("qdrant", "collection_name")


class QdrantVectorStore:
    def __init__(self, host=None, port=None, collection_name=None):
        self.client = QdrantClient(
            host=host or Config.get("qdrant", "host"),
            port=port or Config.get("qdrant", "port"),
        )
        self.collection = collection_name or QDRANT_COLLECTION_NAME

    def init_collection(self):
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            )
            logger.info(f"Qdrantコレクション '{self.collection}' を作成しました。")

    def search(self, query_embed: list[float], top_k=3):
        if not query_embed or len(query_embed) != 768:
            logger.warning(f"Invalid vector for search: {query_embed}")
            return []
        hits = self.client.query_points(
            collection_name=self.collection, query=query_embed, limit=top_k
        )
        return hits.points if hits and hasattr(hits, "points") else []

    def upsert_points(self, points: list):
        if points:
            self.client.upsert(collection_name=self.collection, points=points)
            logger.info(f"Qdrantに{len(points)}件のポイントを登録しました。")
