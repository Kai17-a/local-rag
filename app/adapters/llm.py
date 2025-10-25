from __future__ import annotations

from openai import OpenAI

from ..core.exceptions import LLMError
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class OllamaOpenAIClient:
    """Ollama LLMクライアント（OpenAI互換API使用）"""
    
    def __init__(self) -> None:
        self.base_url = Config.get("ollama", "base_url")
        self.model = Config.get("ollama", "model")
        self.system_prompt = Config.get("ollama", "system_prompt")
        self.client = OpenAI(api_key="dummy", base_url=self.base_url)

    def chat(self, query: str, context: str) -> str:
        """質問と文脈を使って回答を生成"""
        clean_query = query.strip()
        if not clean_query:
            raise LLMError("質問が空です")
        return self._generate_response(clean_query, context)
    
    def _generate_response(self, query: str, context: str) -> str:
        """LLMレスポンスを生成"""
        prompt = f"{self.system_prompt}\n\n質問:\n{query}\n参考文書:\n{context}"
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                timeout=60
            )
            
            if not response.choices or not response.choices[0].message.content:
                raise LLMError("LLMから回答が得られませんでした")
            
            return response.choices[0].message.content.strip()
                    
        except Exception as e:
            logger.error(f"LLM呼び出しエラー: {e}")
            raise LLMError(f"回答生成に失敗しました: {e}") from e
