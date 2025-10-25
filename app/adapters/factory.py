from __future__ import annotations

from ..utils.config import Config
from ..utils.logger import get_logger
from .docker_embedder import DockerEmbedder
from .docker_llm import DockerLLMClient
from .embedder import OllamaEmbedder
from .llm import OllamaOpenAIClient

logger = get_logger(__name__)


def create_llm_client():
    """設定に基づいてLLMクライアントを作成"""
    model_type = Config.get("model_type", default="ollama")
    
    if model_type.lower() == "docker":
        logger.info("Docker LLMクライアントを使用します")
        return DockerLLMClient()
    else:
        logger.info("Ollama LLMクライアントを使用します")
        return OllamaOpenAIClient()


def create_embedder():
    """設定に基づいて埋め込みモデルを作成"""
    model_type = Config.get("model_type", default="ollama")
    
    if model_type.lower() == "docker":
        logger.info("Docker埋め込みモデルを使用します")
        return DockerEmbedder()
    else:
        logger.info("Ollama埋め込みモデルを使用します")
        return OllamaEmbedder()