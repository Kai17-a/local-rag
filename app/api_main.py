from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from .adapters.factory import create_embedder, create_llm_client
from .adapters.vectorstore import QdrantVectorStore
from .core.exceptions import RAGException
from .core.models import (
    DirectoryRequest,
    DocumentIngestResponse,
    ErrorResponse,
    FileUploadResponse,
    QAResponse,
    RegisterTextRequest,
    TextRegisterResponse,
)
from .services.document_ingest_service import DocumentIngestService
from .services.qa_service import QAService
from .utils.logger import get_logger

logger = get_logger("fastapi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """アプリケーションライフサイクル管理"""
    logger.info("FastAPI起動: リソースを初期化中...")
    
    try:
        # 各コンポーネントを初期化
        app.state.embedder = create_embedder()
        app.state.vector_store = QdrantVectorStore()
        await run_in_threadpool(app.state.vector_store.init_collection)
        app.state.llm_client = create_llm_client()
        
        # サービスを初期化
        app.state.qa_service = QAService(
            app.state.llm_client, 
            app.state.embedder, 
            app.state.vector_store
        )
        app.state.document_ingest_service = DocumentIngestService(
            app.state.embedder, 
            app.state.vector_store
        )
        
        logger.info("FastAPI起動完了")
        yield
        
    except Exception as e:
        logger.error(f"FastAPI初期化エラー: {e}")
        raise
    finally:
        logger.info("FastAPI終了")


app = FastAPI(lifespan=lifespan)


@app.get("/", response_model=QAResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def ask_question(q: str = None):
    """質問応答エンドポイント"""
    if not q or not q.strip():
        return JSONResponse(
            ErrorResponse(error="クエリパラメータ 'q' が必要です").model_dump(), 
            status_code=400
        )
    
    try:
        answer = await run_in_threadpool(app.state.qa_service.answer, q.strip())
        return QAResponse(
            question=q.strip(),
            answer=answer
        )
    except RAGException as e:
        logger.error(f"質問応答エラー: {e}")
        return JSONResponse(
            ErrorResponse(error=f"回答生成に失敗しました: {str(e)}").model_dump(), 
            status_code=500
        )
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        return JSONResponse(
            ErrorResponse(error="内部サーバーエラーが発生しました").model_dump(), 
            status_code=500
        )


@app.post("/documents/", response_model=DocumentIngestResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def ingest_documents(request: DirectoryRequest):
    """ディレクトリ内の文書を一括登録"""
    if not os.path.exists(request.directory):
        return JSONResponse(
            ErrorResponse(error=f"ディレクトリが見つかりません: {request.directory}").model_dump(), 
            status_code=400
        )
    
    try:
        await run_in_threadpool(
            app.state.document_ingest_service.ingest, 
            request.directory
        )
        return DocumentIngestResponse(
            message="文書の登録が完了しました",
            directory=request.directory
        )
        
    except RAGException as e:
        logger.error(f"文書登録エラー: {e}")
        return JSONResponse(
            ErrorResponse(error=f"文書登録に失敗しました: {str(e)}").model_dump(), 
            status_code=500
        )
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        return JSONResponse(
            ErrorResponse(error="内部サーバーエラーが発生しました").model_dump(), 
            status_code=500
        )


@app.post("/upload/", response_model=FileUploadResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def upload_files(files: list[UploadFile] = File(...)):
    """複数ファイルをアップロードして登録"""
    if not files:
        return JSONResponse(
            ErrorResponse(error="ファイルが選択されていません").model_dump(), 
            status_code=400
        )
    
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            saved_files = []
            skipped_files = []
            
            for file in files:
                if not file.filename:
                    continue
                    
                ext = Path(file.filename).suffix.lower()
                if ext not in {".pdf", ".txt"}:
                    skipped_files.append(file.filename)
                    logger.warning(f"未対応ファイル形式: {file.filename}")
                    continue
                
                dest_file = temp_path / file.filename
                await run_in_threadpool(_save_upload_file, file, dest_file)
                saved_files.append(str(dest_file))
            
            if not saved_files:
                return JSONResponse(
                    ErrorResponse(error="有効なPDFまたはTXTファイルがありません").model_dump(), 
                    status_code=400
                )
            
            await run_in_threadpool(
                app.state.document_ingest_service.store_qdrant, 
                saved_files
            )
            
            message = f"{len(saved_files)}個のファイルを登録しました"
            if skipped_files:
                message += f" ({len(skipped_files)}個のファイルをスキップ)"
            
            return FileUploadResponse(
                message=message,
                processed_files=len(saved_files),
                skipped_files=skipped_files
            )
            
    except RAGException as e:
        logger.error(f"ファイルアップロードエラー: {e}")
        return JSONResponse(
            ErrorResponse(error=f"ファイル登録に失敗しました: {str(e)}").model_dump(), 
            status_code=500
        )
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        return JSONResponse(
            ErrorResponse(error="内部サーバーエラーが発生しました").model_dump(), 
            status_code=500
        )


@app.post("/text/", response_model=TextRegisterResponse, responses={400: {"model": ErrorResponse}, 500: {"model": ErrorResponse}})
async def register_text(request: RegisterTextRequest):
    """テキストを直接登録"""
    try:
        n_chunks = await run_in_threadpool(
            app.state.document_ingest_service.register_text, 
            request.text, 
            request.source
        )
        
        return TextRegisterResponse(
            message=f"{n_chunks}個のチャンクを登録しました",
            chunks=n_chunks,
            source=request.source
        )
        
    except RAGException as e:
        logger.error(f"テキスト登録エラー: {e}")
        return JSONResponse(
            ErrorResponse(error=f"テキスト登録に失敗しました: {str(e)}").model_dump(), 
            status_code=500
        )
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        return JSONResponse(
            ErrorResponse(error="内部サーバーエラーが発生しました").model_dump(), 
            status_code=500
        )


def _save_upload_file(upload_file: UploadFile, destination: Path) -> None:
    """アップロードファイルを保存"""
    with destination.open("wb") as out_file:
        shutil.copyfileobj(upload_file.file, out_file)
