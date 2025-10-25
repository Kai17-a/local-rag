from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)


def sanitize_filename(name: str) -> str:
    """ファイル名として安全な文字列に変換"""
    sanitized = re.sub(r'[\\/:*?"<>|]', "_", name)
    return sanitized[:255].strip()


def save_log(query: str, answer: str, log_dir: str = "log") -> Path:
    """質問と回答をログファイルに保存"""
    try:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # ファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sanitized_query = re.sub(r"[\r\n\t]+", " ", query)
        
        safe_query = sanitize_filename(sanitized_query)
        if not safe_query:
            safe_query = "query"
            
        log_file = log_path / f"{timestamp}_{safe_query}.txt"
        
        # ログ内容を作成
        log_content = f"""質問: {query}

回答:
{answer}

---
保存日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        log_file.write_text(log_content, encoding="utf-8")
        logger.info(f"ログを保存しました: {log_file}")
        return log_file
        
    except Exception as e:
        logger.error(f"ログ保存エラー: {e}")
        raise


def multiline_input(prompt: str = "質問を入力してください (空行で終了): ") -> str:
    """複数行入力を受け取る"""
    print(prompt)
    lines = []
    
    try:
        while True:
            line = input()
            if line == "":
                break
            lines.append(line)
    except KeyboardInterrupt:
        print("\n入力がキャンセルされました")
        return ""
        
    return "\n".join(lines)


def read_file_safe(file_path: str, encoding: str = "utf-8") -> str | None:
    """ファイルを安全に読み込む"""
    try:
        return Path(file_path).read_text(encoding=encoding)
    except FileNotFoundError:
        logger.error(f"ファイルが見つかりません: {file_path}")
        return None
    except Exception as e:
        logger.error(f"ファイル読み込みエラー ({file_path}): {e}")
        return None


def write_file_safe(file_path: str, content: str, encoding: str = "utf-8") -> bool:
    """ファイルを安全に書き込む"""
    try:
        file_obj = Path(file_path)
        file_obj.parent.mkdir(parents=True, exist_ok=True)
        file_obj.write_text(content, encoding=encoding)
        return True
    except Exception as e:
        logger.error(f"ファイル書き込みエラー ({file_path}): {e}")
        return False
