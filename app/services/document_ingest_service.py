from __future__ import annotations

import os
import uuid
from pathlib import Path

import pdfplumber
from qdrant_client.models import PointStruct

from ..core.exceptions import DocumentProcessingError
from ..utils.config import Config
from ..utils.logger import get_logger

logger = get_logger(__name__)


class DocumentIngestService:
    """文書取り込みサービス"""
    
    def __init__(self, embedder, vector_store) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.debug_chunk_output = Config.get("debug", "chunk_output", default=False)

    def get_registerable_files(self, directory: str) -> list[str]:
        """登録可能なファイル一覧を取得"""
        directory_path = Path(directory)
        
        if not directory_path.exists():
            raise DocumentProcessingError(f"ディレクトリが存在しません: {directory}")
        return self._scan_supported_files(directory_path)
    
    def _scan_supported_files(self, directory_path: Path) -> list[str]:
        """サポートされているファイルをスキャン"""
        supported_extensions = {".pdf", ".txt"}
        files = []
        
        try:
            for root, _, file_list in os.walk(directory_path):
                for file in file_list:
                    file_path = Path(file)
                    if file_path.suffix.lower() in supported_extensions:
                        files.append(os.path.join(root, file))
            return files
        except Exception as e:
            raise DocumentProcessingError(f"ファイル検索エラー: {e}") from e

    def load_pdf_document(self, path: str) -> list[str]:
        """PDFファイルを読み込んでチャンクに分割"""
        try:
            chunks = []
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        chunks.append(page_text.strip())
            return chunks
        except Exception as e:
            raise DocumentProcessingError(f"PDF読み込みエラー ({path}): {e}") from e

    def load_txt_document(self, path: str) -> list[str]:
        """テキストファイルを読み込んでチャンクに分割"""
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            
            if not content:
                return []
            return self._split_text_into_chunks(content)
                    
        except Exception as e:
            raise DocumentProcessingError(f"テキストファイル読み込みエラー ({path}): {e}") from e
    
    def _split_text_into_chunks(
        self, 
        text: str, 
        chunk_size: int = 1000, 
        overlap: int = 100
    ) -> list[str]:
        """テキストをチャンクに分割"""
        if len(text) <= chunk_size:
            return [text]
        return self._perform_chunking(text, chunk_size, overlap)
    
    def _perform_chunking(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """実際のチャンク分割を実行"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # 文の途中で切れないように調整
            if end < len(text) and not text[end].isspace():
                last_period = chunk.rfind('。')
                last_newline = chunk.rfind('\n')
                cut_point = max(last_period, last_newline)
                
                if cut_point > start + chunk_size // 2:
                    chunk = text[start:cut_point + 1]
                    end = cut_point + 1
            
            if stripped_chunk := chunk.strip():
                chunks.append(stripped_chunk)
                
            start = end - overlap
            
        return chunks

    def store_qdrant(self, files: list[str]) -> None:
        """ファイルをQdrantに保存"""
        debug_dir = Path("./debug_chunks")
        
        for file_path in files:
            try:
                self._process_single_file(file_path, debug_dir)
            except Exception as e:
                logger.error(f"ファイル処理エラー ({file_path}): {e}")
                continue
        
        logger.info("インデックス作成完了")
        if self.debug_chunk_output:
            logger.info("チャンク内容を ./debug_chunks に出力しました")
    
    def _process_single_file(self, file_path: str, debug_dir: Path) -> None:
        """単一ファイルを処理"""
        ext = os.path.splitext(file_path)[1].lower()
        base_name = os.path.basename(file_path)
        logger.info(f"{file_path} をQdrantに保存中")
        
        # ファイル形式に応じて読み込み
        if ext == ".pdf":
            chunks = self.load_pdf_document(file_path)
        elif ext == ".txt":
            chunks = self.load_txt_document(file_path)
        else:
            logger.warning(f"未対応ファイル形式: {file_path}")
            return

        if not chunks:
            logger.warning(f"チャンクが生成されませんでした: {file_path}")
            return
        
        self._create_and_store_points(chunks, base_name, ext, debug_dir)
    
    def _create_and_store_points(
        self, 
        chunks: list[str], 
        base_name: str, 
        ext: str, 
        debug_dir: Path
    ) -> None:
        """チャンクからポイントを作成してストア"""
        # デバッグ出力準備
        debug_subdir = None
        if self.debug_chunk_output:
            debug_subdir = debug_dir / base_name.replace(ext, "")
            debug_subdir.mkdir(parents=True, exist_ok=True)

        # ポイント作成
        points = []
        for idx, chunk in enumerate(chunks):
            if debug_subdir:
                debug_file = debug_subdir / f"chunk_{idx:03}.txt"
                debug_file.write_text(chunk, encoding="utf-8")
            
            try:
                embed = self.embedder.embed(chunk)
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
            except Exception as e:
                logger.warning(f"チャンク埋め込み生成失敗 ({base_name}, chunk {idx}): {e}")
                continue

        if not points:
            logger.warning(f"有効なポイントが生成されませんでした: {base_name}")
        else:
            self.vector_store.upsert_points(points)
            logger.info(f"{base_name}: {len(points)}チャンクを登録")

    def ingest(self, target_dir: str) -> None:
        """ディレクトリ内の文書を一括取り込み"""
        try:
            self.vector_store.init_collection()
            files = self.get_registerable_files(target_dir)
            
            if not files:
                logger.warning("登録可能なファイルが見つかりません")
                return
            
            logger.info(f"{len(files)}個のファイルを処理します")
            self.store_qdrant(files)
            
        except Exception as e:
            logger.error(f"文書取り込みエラー: {e}")
            raise DocumentProcessingError(f"文書取り込みに失敗しました: {e}") from e

    def register_text(self, text: str, source: str = "input_text") -> int:
        """テキストをチャンク分割してベクターストアに登録"""
        clean_text = text.strip()
        if not clean_text:
            raise DocumentProcessingError("空のテキストは登録できません")
        return self._process_text_registration(clean_text, source)
    
    def _process_text_registration(self, text: str, source: str) -> int:
        """テキスト登録処理を実行"""
        try:
            chunks = self._split_text_into_chunks(text, chunk_size=300, overlap=50)
            
            points = []
            for idx, chunk in enumerate(chunks):
                try:
                    embed = self.embedder.embed(chunk)
                    points.append(
                        PointStruct(
                            id=str(uuid.uuid4()),
                            vector=embed,
                            payload={
                                "text": chunk,
                                "source": source,
                                "chunk_id": idx,
                            },
                        )
                    )
                except Exception as e:
                    logger.warning(f"テキストチャンク埋め込み生成失敗 (chunk {idx}): {e}")
                    continue
            
            if not points:
                logger.warning("有効なポイントが生成されませんでした")
                return 0
            
            self.vector_store.upsert_points(points)
            logger.info(f"テキスト登録完了: {len(points)}チャンク")
            return len(points)
            
        except Exception as e:
            logger.error(f"テキスト登録エラー: {e}")
            raise DocumentProcessingError(f"テキスト登録に失敗しました: {e}") from e
