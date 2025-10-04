from pathlib import Path
import re
from qdrant_client import QdrantClient
import requests
from openai import OpenAI

OLLAMA_URL = "http://localhost:12000/v1"
EMBED_URL = "http://localhost:12000/api/embeddings"
DEFAULT_MODEL = "llama3:latest"
DEFAULT_SYSTEM_PROMPT = (
    "あなたは有能なアシスタントです。日本語で回答してください\n"
    "質問内容が英語でも日本語で回答してください\n"
    "資料に書かれていない内容は絶対に答えないこと\n"
    "資料に書かれていない内容の推測の回答はしないこと\n"
    "例外的に資料に記載されている内容を組み合わせた推測はしていいものとする"
)

OPEN_AI_CLIENT = OpenAI(api_key="dummy", base_url=OLLAMA_URL)
QDRANT_COLLECTION_NAME = "local_docs"
QDRANT_CLIENT = QdrantClient(host="localhost", port=6333)


def ollama_embed(text: str) -> list[float]:
    """Ollama APIを使って埋め込みベクトルを生成"""
    try:
        r = requests.post(EMBED_URL, json={"model": "nomic-embed-text", "prompt": text})
        r.raise_for_status()
        return r.json().get("embedding", [])
    except Exception as e:
        print(f"Embedding生成エラー: {e}")
        return []


def search_documents(query: str, top_k: int = 3) -> str:
    """RAG検索を実行し、関連文書を返す"""
    query_embed = ollama_embed(query)
    if not query_embed:
        print("Failed to generate embed query.")
        return {"text": "", "sources": []}

    if len(query_embed) != 768:
        print(f"Invalid vector length.: {len(query_embed)}")
        return {"text": "", "sources": []}

    query_embed = [float(x) for x in query_embed]

    hits = QDRANT_CLIENT.query_points(
        collection_name=QDRANT_COLLECTION_NAME, query=query_embed, limit=top_k
    )

    if not hits.points:
        return {"text": "", "sources": []}

    texts = []
    sources = []
    for point in hits.points:
        payload = point.payload or {}
        if "text" in payload:
            texts.append(payload["text"])
        if "source" in payload:
            sources.append(payload["source"])

    return {"text": "\n".join(texts), "sources": list(set(sources))}


def generate_answer(query: str) -> str:
    result = search_documents(query)
    context_text = result["text"]
    sources = result["sources"]

    if not context_text:
        print("Related documents is not found.")
        return

    prompt = f"{DEFAULT_SYSTEM_PROMPT}\n\n質問:\n{query}\n参考文書:\n{context_text}"

    response = OPEN_AI_CLIENT.chat.completions.create(
        model="llama3:latest",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    answer = response.choices[0].message.content.strip()

    # 回答の最後に参照ファイル名を追記
    if sources:
        answer += "\n\n---\n参考資料:\n" + "\n".join(f"- {src}" for src in sources)

    return answer


def sanitize_filename(name: str) -> str:
    """Windowsで使えない記号を除去して安全なファイル名にする"""
    # 禁止文字を _ に置き換え
    sanitized = re.sub(r'[\\/:*?"<>|]', "_", name)
    # ファイル名が長すぎる場合は短縮（255文字制限対策）
    return sanitized[:255].strip()


def save_log(query: str, answer: str):
    log_path = Path("log")
    log_path.mkdir(parents=True, exist_ok=True)

    # 改行・タブを空白に変換
    sanitized = re.sub(r"[\r\n\t]+", "", query)

    # Windowsで使えない記号を除去
    sanitized = re.sub(r'[\\/:*?"<>|]', "_", sanitized)

    # 長すぎる場合はカット（255文字制限）
    safe_query = sanitized[:255].strip()

    # ファイル名が空になった場合のフォールバック
    if not safe_query:
        safe_query = "logfile"

    log_file = f"{log_path}/{safe_query}.txt"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Question:\n{query}\n\n")
        f.write(f"Answer:\n{answer}")

    print(f"ログを保存しました: {log_file}")


def multiline_input(prompt="Question (空行で終了): "):
    print(prompt)
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)


def main():
    query = multiline_input()
    answer = generate_answer(query)
    print(f"Answer:\n{answer}")
    save_log(query, answer)


if __name__ == "__main__":
    main()
