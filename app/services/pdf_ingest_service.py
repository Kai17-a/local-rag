import os
import uuid
import pdfplumber
from pathlib import Path
from qdrant_client.models import PointStruct
from utils.logger import get_logger
from utils.config import Config

logger = get_logger(__name__)
DEBUG_CHUNK_OUTPUT = Config.get("debug", "chunk_output", default=False)


class PDFIngestService:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def get_pdfs(self, directory: str) -> list[str]:
        pdf_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(".pdf"):
                    pdf_files.append(os.path.join(root, file))
        return pdf_files

    def load_pdf_document(self, path: str) -> str:
        if not os.path.exists(path):
            logger.error(f"PDFファイルが存在しません: {path}")
            return ""
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        return text

    def split_text_per_page(self, text: str) -> list[str]:
        pages = [page.strip() for page in text.split("\n\n") if page.strip()]
        return pages

    def store_qdrant(self, pdfs: list[str]):
        debug_dir = Path("./debug_chunks")

        for pdf in pdfs:
            points = []
            logger.info(f"{pdf} をqdrantに保存中")
            pdf_name = os.path.basename(pdf)
            pdf_txt = self.load_pdf_document(pdf)
            chunks = self.split_text_per_page(pdf_txt)

            if DEBUG_CHUNK_OUTPUT:
                pdf_debug_dir = debug_dir / pdf_name.replace(".pdf", "")
                pdf_debug_dir.mkdir(parents=True, exist_ok=True)

            for idx, chunk in enumerate(chunks):
                if DEBUG_CHUNK_OUTPUT:
                    with open(
                        pdf_debug_dir / f"chunk_{idx:03}.txt", "w", encoding="utf-8"
                    ) as f:
                        f.write(chunk)

                embed = self.embedder.embed(chunk)
                if embed:
                    points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embed,
                            payload={
                                "text": chunk,
                                "source": pdf_name,
                                "chunk_id": idx,
                            },
                        )
                    )

            self.vector_store.upsert_points(points)

        logger.info("インデックス作成完了")
        if DEBUG_CHUNK_OUTPUT:
            logger.info("チャンク内容を ./debug_chunks に出力しました。")

    def ingest(self, target_dir: str):
        if not os.path.exists(target_dir):
            logger.error(f"{target_dir} is not found.")
            return
        self.vector_store.init_collection()
        pdfs = self.get_pdfs(target_dir)
        if len(pdfs) == 0:
            logger.warning("PDFファイルが見つかりません。")
            return
        self.store_qdrant(pdfs)
