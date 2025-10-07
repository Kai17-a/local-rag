import requests
from utils.config import Config
from utils.logger import get_logger

logger = get_logger(__name__)

EMBED_URL = Config.get("ollama", "embed_url")
EMBED_MODEL = Config.get("ollama", "embed_model")


class OllamaEmbedder:
    def embed(self, text: str) -> list[float]:
        try:
            r = requests.post(EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
            r.raise_for_status()
            return [float(x) for x in r.json().get("embedding", [])]
        except Exception as e:
            logger.error(f"Embedding生成エラー: {e}")
            return []
