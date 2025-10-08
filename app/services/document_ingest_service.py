import os
import uuid
import pdfplumber
from pathlib import Path
from qdrant_client.models import PointStruct
from utils.logger import get_logger
from utils.config import Config

logger = get_logger(__name__)
DEBUG_CHUNK_OUTPUT = Config.get("debug", "chunk_output", default=False)


class DocumentIngestService:
    def __init__(self, embedder, vector_store):
        self.embedder = embedder
        self.vector_store = vector_store

    def get_registerable_files(self, directory: str) -> list[str]:
        files = []
        for root, _, file_list in os.walk(directory):
            for file in file_list:
                if file.lower().endswith(".pdf") or file.lower().endswith(".txt"):
                    files.append(os.path.join(root, file))
        return files

    def load_pdf_document(self, path: str) -> list[str]:
        text = ""
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n\n"
        # ページごとチャンク
        return [page.strip() for page in text.split("\n\n") if page.strip()]

    def load_txt_document(self, path: str) -> list[str]:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # 1ファイル＝1チャンク（必要なら分割方法を調整可）
        return [content.strip()] if content.strip() else []

    def store_qdrant(self, files: list[str]):
        debug_dir = Path("./debug_chunks")
        for file in files:
            ext = os.path.splitext(file)[1].lower()
            points = []
            logger.info(f"{file} をqdrantに保存中")
            base_name = os.path.basename(file)
            if ext == ".pdf":
                chunks = self.load_pdf_document(file)
            elif ext == ".txt":
                chunks = self.load_txt_document(file)
            else:
                logger.warning(f"未対応ファイル形式: {file}")
                continue

            if DEBUG_CHUNK_OUTPUT:
                debug_subdir = debug_dir / base_name.replace(ext, "")
                debug_subdir.mkdir(parents=True, exist_ok=True)

            for idx, chunk in enumerate(chunks):
                if DEBUG_CHUNK_OUTPUT:
                    with open(
                        debug_subdir / f"chunk_{idx:03}.txt", "w", encoding="utf-8"
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
                                "source": base_name,
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
        files = self.get_registerable_files(target_dir)
        if len(files) == 0:
            logger.warning("登録可能なファイルが見つかりません。")
            return
        self.store_qdrant(files)
