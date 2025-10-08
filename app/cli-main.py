from adapters.llm import OllamaOpenAIClient
from adapters.vectorstore import QdrantVectorStore
from adapters.embedder import OllamaEmbedder
from services.qa_service import QAService
from app.services.document_ingest_service import DocumentIngestService
from utils.io import multiline_input, save_log
from utils.logger import get_logger

logger = get_logger(__name__)


def main():
    embedder = OllamaEmbedder()
    vector_store = QdrantVectorStore()
    vector_store.init_collection()  # ← ここで必ず初期化

    llm_client = OllamaOpenAIClient()
    qa_service = QAService(llm_client, embedder, vector_store)
    pdf_ingest_service = DocumentIngestService(embedder, vector_store)

    print("1: 質問・検索\n2: PDF登録")
    choice = input("番号を選択してください: ").strip()
    if choice == "2":
        target_dir = input("documents directory path(ex. /home/documents): ").strip()
        pdf_ingest_service.ingest(target_dir)
    else:
        query = multiline_input()
        answer = qa_service.answer(query)
        print(f"Answer:\n{answer}")
        save_log(query, answer)


if __name__ == "__main__":
    main()
