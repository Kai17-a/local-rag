from __future__ import annotations

import requests

from ..core.exceptions import EmbeddingError
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DockerEmbedder:
    """Docker埋め込みモデルのアダプター（llama.cpp互換API使用）"""
    
    def __init__(self) -> None:
        self.base_url = Config.get("docker", "base_url")
        self.embed_endpoint = Config.get("docker", "embed_endpoint")
        self.embed_model = Config.get("docker", "embed_model")
        self.headers = {"Content-Type": "application/json"}
        
    def embed(self, text: str) -> list[float]:
        """テキストを埋め込みベクトルに変換"""
        clean_text = text.strip()
        if not clean_text:
            raise EmbeddingError("空のテキストは埋め込みできません")
        return self._generate_embedding(clean_text)
    
    def _generate_embedding(self, text: str) -> list[float]:
        """埋め込みベクトルを生成"""
        data = {
            "model": self.embed_model,
            "input": text,
            "encoding_format": "float"
        }
        
        try:
            url = f"{self.base_url}{self.embed_endpoint}"
            response = requests.post(url, json=data, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            data_list = result.get("data", [])
            
            if not data_list or not data_list[0].get("embedding"):
                raise EmbeddingError("埋め込みベクトルが取得できませんでした")
            
            embedding = data_list[0]["embedding"]
            return [float(x) for x in embedding]
                    
        except requests.RequestException as e:
            logger.error(f"Docker埋め込み生成リクエストエラー: {e}")
            raise EmbeddingError(f"埋め込み生成に失敗しました: {e}") from e
        except Exception as e:
            logger.error(f"Docker埋め込み生成エラー: {e}")
            raise EmbeddingError(f"予期しないエラー: {e}") from e