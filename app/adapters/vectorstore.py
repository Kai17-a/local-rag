from __future__ import annotations

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from ..core.exceptions import VectorStoreError
from ..core.models import SearchResult
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class QdrantVectorStore:
    """Qdrantベクターストアのアダプター"""
    
    def __init__(
        self, 
        host: str | None = None, 
        port: int | None = None, 
        collection_name: str | None = None
    ) -> None:
        self.host = host or Config.get("qdrant", "host")
        self.port = port or Config.get("qdrant", "port")
        self.collection = collection_name or Config.get("qdrant", "collection_name")
        
        try:
            self.client = QdrantClient(host=self.host, port=self.port)
        except Exception as e:
            raise VectorStoreError(f"Qdrantクライアントの初期化に失敗: {e}") from e

    def init_collection(self) -> None:
        """コレクションを初期化"""
        try:
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=768, distance=Distance.COSINE),
                )
                logger.info(f"Qdrantコレクション '{self.collection}' を作成しました")
        except Exception as e:
            raise VectorStoreError(f"コレクション初期化に失敗: {e}") from e

    def search(self, query_embed: list[float], top_k: int = 3) -> list[SearchResult]:
        """ベクトル検索を実行"""
        if not query_embed:
            raise VectorStoreError("検索ベクトルが空です")
        if len(query_embed) != 768:
            raise VectorStoreError("無効な検索ベクトルです（768次元である必要があります）")
        return self._perform_search(query_embed, top_k)
    
    def _perform_search(self, query_embed: list[float], top_k: int) -> list[SearchResult]:
        """実際の検索を実行"""
        try:
            hits = self.client.query_points(
                collection_name=self.collection, 
                query=query_embed, 
                limit=top_k
            )
            
            results = []
            if hits and hasattr(hits, "points") and hits.points:
                for point in hits.points:
                    payload = point.payload or {}
                    results.append(SearchResult(
                        text=payload.get("text", ""),
                        source=payload.get("source", ""),
                        score=point.score or 0.0,
                        chunk_id=payload.get("chunk_id")
                    ))
            return results
                    
        except Exception as e:
            logger.error(f"ベクトル検索エラー: {e}")
            raise VectorStoreError(f"検索に失敗しました: {e}") from e

    def upsert_points(self, points: list[PointStruct]) -> None:
        """ポイントを挿入・更新"""
        if not points:
            return
        self._upsert_to_qdrant(points)
    
    def _upsert_to_qdrant(self, points: list[PointStruct]) -> None:
        """Qdrantにポイントを登録"""
        try:
            self.client.upsert(collection_name=self.collection, points=points)
            logger.info(f"Qdrantに{len(points)}件のポイントを登録しました")
        except Exception as e:
            logger.error(f"ポイント登録エラー: {e}")
            raise VectorStoreError(f"ポイント登録に失敗しました: {e}") from e
