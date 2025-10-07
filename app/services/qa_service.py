from utils.logger import get_logger

logger = get_logger(__name__)


class QAService:
    def __init__(self, llm_client, embedder, vector_store):
        self.llm_client = llm_client
        self.embedder = embedder
        self.vector_store = vector_store

    def answer(self, query: str):
        query_embed = self.embedder.embed(query)
        if not query_embed or len(query_embed) != 768:
            logger.warning("Failed to generate valid embedding for query.")
            return ""

        points = self.vector_store.search(query_embed)
        if not points:
            logger.info("Related documents is not found.")
            return "関連する資料がありませんでした"

        texts = []
        sources = []
        for point in points:
            payload = point.payload or {}
            text = payload.get("text")
            source = payload.get("source")
            if text:
                texts.append(text)
            if source:
                sources.append(source)

        context_text = "\n".join(texts)
        sources = list(set(sources))

        answer = self.llm_client.chat(query, context_text)

        if sources:
            answer += "\n\n---\n参考資料:\n" + "\n".join(f"- {src}" for src in sources)
        return answer
