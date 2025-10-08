from fastapi import FastAPI, status, UploadFile, File, Request
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from adapters.llm import OllamaOpenAIClient
from adapters.vectorstore import QdrantVectorStore
from adapters.embedder import OllamaEmbedder
from services.qa_service import QAService
from services.document_ingest_service import DocumentIngestService  # ← ここに変更
from utils.logger import get_logger
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

import tempfile
import shutil
import os
from pathlib import Path

logger = get_logger("fastapi")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("FastAPI lifespan: initializing resources.")
    app.state.embedder = OllamaEmbedder()
    app.state.vector_store = QdrantVectorStore()
    await run_in_threadpool(app.state.vector_store.init_collection)
    app.state.llm_client = OllamaOpenAIClient()
    app.state.qa_service = QAService(
        app.state.llm_client, app.state.embedder, app.state.vector_store
    )
    app.state.document_ingest_service = DocumentIngestService(
        app.state.embedder, app.state.vector_store
    )
    yield
    logger.info("FastAPI lifespan: shutdown.")


app = FastAPI(lifespan=lifespan)


@app.get("/", status_code=status.HTTP_200_OK)
async def read_root(q: str = None):
    """
    質問応答エンドポイント
    クエリパラメータqに質問文を指定
    """
    if q is None:
        return JSONResponse(
            {"error": "Query parameter 'q' is required."}, status_code=400
        )
    answer = await run_in_threadpool(app.state.qa_service.answer, q)
    return JSONResponse({"question": q, "answer": answer})


@app.post("/", status_code=status.HTTP_201_CREATED)
async def store_document(request: Request):
    """
    ドキュメント登録エンドポイント
    JSON bodyで{"directory": "..."}を受け取る
    """
    try:
        # JSONがない・不正な場合は400
        try:
            body = await request.json()
        except Exception:
            logger.warning("リクエストボディがJSON形式ではありません")
            return JSONResponse(
                {"error": "Request body must be JSON."}, status_code=400
            )

        directory = body.get("directory") if isinstance(body, dict) else None
        if not directory or not isinstance(directory, str) or not directory.strip():
            logger.warning("リクエストボディに'directory'がありません")
            return JSONResponse(
                {"error": "Valid 'directory' is required in JSON body."},
                status_code=400,
            )
        if not os.path.exists(directory):
            logger.warning(f"指定ディレクトリが存在しません: {directory}")
            return JSONResponse(
                {"error": f"Directory '{directory}' not found."}, status_code=400
            )
        await run_in_threadpool(app.state.document_ingest_service.ingest, directory)
        return JSONResponse({"message": "Text/PDF documents indexed successfully."})
    except Exception as e:
        logger.error(f"Error in store_document: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/upload/", status_code=status.HTTP_201_CREATED)
async def upload_file(files: list[UploadFile] = File(...)):
    """
    複数PDF/テキストファイルをアップロードし、そのままベクトル登録
    """
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            saved_files = []
            for file in files:
                ext = file.filename.lower().split(".")[-1]
                if ext not in ("pdf", "txt"):
                    logger.warning(
                        f"アップロードファイルがPDFまたはTXTではありません: {file.filename}"
                    )
                    continue
                dest_file = temp_path / file.filename
                await run_in_threadpool(_save_upload_file, file, dest_file)
                saved_files.append(str(dest_file))
            if not saved_files:
                return JSONResponse(
                    {"error": "No valid PDF or TXT files uploaded."}, status_code=400
                )
            await run_in_threadpool(
                app.state.document_ingest_service.store_qdrant, saved_files
            )
            return JSONResponse(
                {"message": f"{len(saved_files)} file(s) indexed successfully."}
            )
    except Exception as e:
        logger.error(f"Error in upload_file: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


class RegisterTextRequest(BaseModel):
    text: str
    source: str = "input_text"


@app.post("/register_text/", status_code=status.HTTP_201_CREATED)
async def register_text_endpoint(req: RegisterTextRequest):
    """
    テキスト文字列をベクターストアに登録
    """
    if not req.text or not req.text.strip():
        return JSONResponse({"error": "Text is required."}, status_code=400)
    # sourceは省略可
    n_points = await run_in_threadpool(
        app.state.document_ingest_service.register_text, req.text, req.source
    )
    return JSONResponse(
        {"message": f"{n_points} chunk(s) registered.", "source": req.source}
    )


def _save_upload_file(upload_file: UploadFile, destination: Path):
    with destination.open("wb") as out_file:
        shutil.copyfileobj(upload_file.file, out_file)
