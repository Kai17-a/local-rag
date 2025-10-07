from pathlib import Path
import re
from utils.logger import get_logger

logger = get_logger(__name__)


def sanitize_filename(name: str) -> str:
    sanitized = re.sub(r'[\\/:*?"<>|]', "_", name)
    return sanitized[:255].strip()


def save_log(query: str, answer: str):
    log_path = Path("log")
    log_path.mkdir(parents=True, exist_ok=True)
    sanitized = re.sub(r"[\r\n\t]+", "", query)
    safe_query = sanitize_filename(sanitized)
    if not safe_query:
        safe_query = "logfile"
    log_file = log_path / f"{safe_query}.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Question:\n{query}\n\n")
        f.write(f"Answer:\n{answer}")
    logger.info(f"ログを保存しました: {log_file}")


def multiline_input(prompt="Question (空行で終了): "):
    print(prompt)
    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)
    return "\n".join(lines)
