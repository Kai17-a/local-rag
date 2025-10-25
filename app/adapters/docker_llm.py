from __future__ import annotations

import requests

from ..core.exceptions import LLMError
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DockerLLMClient:
    """Docker LLMクライアント（llama.cpp互換API使用）"""
    
    def __init__(self) -> None:
        self.base_url = Config.get("docker", "base_url")
        self.chat_endpoint = Config.get("docker", "chat_endpoint")
        self.model = Config.get("docker", "model")
        self.system_prompt = Config.get("docker", "system_prompt")
        self.headers = {"Content-Type": "application/json"}

    def chat(self, query: str, context: str) -> str:
        """質問と文脈を使って回答を生成"""
        clean_query = query.strip()
        if not clean_query:
            raise LLMError("質問が空です")
        return self._generate_response(clean_query, context)
    
    def _generate_response(self, query: str, context: str) -> str:
        """LLMレスポンスを生成"""
        prompt = f"{self.system_prompt}\n\n質問:\n{query}\n参考文書:\n{context}"
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": f"質問:\n{query}\n参考文書:\n{context}"}
            ],
        }
        
        try:
            url = f"{self.base_url}{self.chat_endpoint}"
            response = requests.post(url, json=data, headers=self.headers, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            choices = result.get("choices", [])
            
            if not choices or not choices[0].get("message", {}).get("content"):
                raise LLMError("LLMから回答が得られませんでした")
            
            return choices[0]["message"]["content"].strip()
                    
        except requests.RequestException as e:
            logger.error(f"Docker LLM呼び出しエラー: {e}")
            raise LLMError(f"回答生成に失敗しました: {e}") from e
        except Exception as e:
            logger.error(f"Docker LLM予期しないエラー: {e}")
            raise LLMError(f"予期しないエラー: {e}") from e