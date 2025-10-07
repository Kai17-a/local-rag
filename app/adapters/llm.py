from openai import OpenAI
from utils.config import Config

OLLAMA_URL = Config.get("ollama", "base_url")
DEFAULT_MODEL = Config.get("ollama", "model")
DEFAULT_SYSTEM_PROMPT = Config.get("ollama", "system_prompt")


class OllamaOpenAIClient:
    def __init__(
        self,
        base_url=OLLAMA_URL,
        model=DEFAULT_MODEL,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
    ):
        self.client = OpenAI(api_key="dummy", base_url=base_url)
        self.model = model
        self.system_prompt = system_prompt

    def chat(self, query: str, context: str):
        prompt = f"{self.system_prompt}\n\n質問:\n{query}\n参考文書:\n{context}"
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
