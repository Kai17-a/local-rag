from __future__ import annotations

import logging
import sys


def get_logger(name: str | None = None, level: int = logging.INFO) -> logging.Logger:
    """ロガーを取得・設定"""
    logger = logging.getLogger(name)
    
    # 既にハンドラーが設定されている場合はそのまま返す
    if logger.hasHandlers():
        return logger
    else:
        return _setup_logger_handlers(logger, level)


def _setup_logger_handlers(logger: logging.Logger, level: int) -> logging.Logger:
    """ロガーのハンドラーを設定"""
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s %(name)s: %(message)s", 
        "%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # ログレベルを設定
    logger.setLevel(level)
    logger.propagate = False
    
    return logger


def setup_logging(level: int = logging.INFO) -> None:
    """アプリケーション全体のログ設定"""
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )
