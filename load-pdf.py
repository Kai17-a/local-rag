import os
import sys
import uuid
import pdfplumber
import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from pathlib import Path

EMBED_URL = "http://localhost:12000/api/embeddings"

QDRANT_COLLECTION_NAME = "local_docs"
QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)
DEBUG_CHUNK_OUTPUT = True  # ← デバッグ出力を有効にするフラグ


def init():
    if not QDRANT_CLIENT.collection_exists(QDRANT_COLLECTION_NAME):
        QDRANT_CLIENT.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )


def ollama_embed(text: str) -> list[float]:
    """Ollama APIを使って埋め込みベクトルを生成"""
    try:
        r = requests.post(EMBED_URL, json={"model": "nomic-embed-text", "prompt": text})
        r.raise_for_status()
        return r.json().get("embedding", [])
    except Exception as e:
        print(f"Embedding生成エラー: {e}")
        return []


def get_pdfs(directory: str) -> list[str]:
    pdf_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                pdf_files.append(os.path.join(root, file))
    return pdf_files


def load_pdf_document(path: str) -> str:
    if not os.path.exists(path):
        print(f"error: pdfファイルが存在しません。\n{path}")
        return ""

    text = ""
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"  # 改ページで区切りを明示
    return text


def split_text(text: str, chunk_size: int = 300, overlap: int = 50) -> list[str]:
    """
    スライドPDFを考慮したチャンク分割処理。
    改ページや空行を優先的に区切る。
    """
    slides = [s.strip() for s in text.split("\n\n") if s.strip()]
    chunks = []
    current_chunk = ""

    for slide in slides:
        if len(current_chunk) + len(slide) > chunk_size:
            chunks.append(current_chunk.strip())
            current_chunk = current_chunk[-overlap:] + "\n" + slide
        else:
            current_chunk += "\n" + slide

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks


def split_text_per_page(text: str) -> list[str]:
    """
    PDFの1ページごとにチャンク化する
    改ページ区切り（'\n\n'）は無視して、ページ単位で1チャンクとする
    """
    # 改ページで区切られたテキストを1ページごとのリストにする
    pages = [page.strip() for page in text.split("\n\n") if page.strip()]
    return pages


def store_qdrant(pdfs: list[str]):
    debug_dir = Path("./debug_chunks")

    for pdf in pdfs:
        points = []
        print(f"{pdf} をqdrantに保存中")
        pdf_name = os.path.basename(pdf)
        pdf_txt = load_pdf_document(pdf)
        # chunks = split_text(pdf_txt)
        chunks = split_text_per_page(pdf_txt)

        # 🔍 デバッグ出力用ディレクトリを作成
        if DEBUG_CHUNK_OUTPUT:
            pdf_debug_dir = debug_dir / pdf_name.replace(".pdf", "")
            pdf_debug_dir.mkdir(parents=True, exist_ok=True)

        for idx, chunk in enumerate(chunks):
            # チャンク内容をファイルとして保存
            if DEBUG_CHUNK_OUTPUT:
                with open(
                    pdf_debug_dir / f"chunk_{idx:03}.txt", "w", encoding="utf-8"
                ) as f:
                    f.write(chunk)

            # ベクトル生成
            embed = ollama_embed(chunk)
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

        # Qdrantに登録
        if points:
            QDRANT_CLIENT.upsert(collection_name=QDRANT_COLLECTION_NAME, points=points)

    print("インデックス作成完了")
    if DEBUG_CHUNK_OUTPUT:
        print("チャンク内容を ./debug_chunks に出力しました。")


def main():
    if len(sys.argv) < 2:
        target_dir = str(input("documents directory path(ex. /home/documents): "))
    else:
        target_dir = sys.argv[1]

    if not os.path.exists(target_dir):
        print(f"{target_dir} is not found.")
        exit()

    init()
    pdfs = get_pdfs(target_dir)

    if len(pdfs) == 0:
        print(f"PDF file is not found.")
        exit()

    store_qdrant(pdfs)


if __name__ == "__main__":
    main()
