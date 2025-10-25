from __future__ import annotations

from pathlib import Path
from typing import Any

import toml


class Config:
    """設定管理クラス"""
    _config: dict[str, Any] | None = None
    _config_path: str | None = None

    @classmethod
    def load(cls, path: str = "app/config.toml") -> dict[str, Any]:
        """設定ファイルを読み込み"""
        if cls._config is None or cls._config_path != path:
            return cls._load_config_file(path)
        return cls._config
    
    @classmethod
    def _load_config_file(cls, path: str) -> dict[str, Any]:
        """設定ファイルを実際に読み込む"""
        config_path = Path(path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
            
        try:
            cls._config = toml.load(config_path.open(encoding="utf-8"))
            cls._config_path = path
            return cls._config
        except Exception as e:
            raise ValueError(f"設定ファイルの読み込みに失敗しました: {e}") from e

    @classmethod
    def get(cls, *keys: str, default: Any = None) -> Any:
        """設定値を取得"""
        config = cls.load()
        
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return default
                    
        return config
    
    @classmethod
    def reload(cls) -> None:
        """設定を再読み込み"""
        cls._config = None
        cls._config_path = None
    
    @classmethod
    def get_ollama_config(cls) -> dict[str, Any]:
        """Ollama設定を取得"""
        return {
            "base_url": cls.get("ollama", "base_url"),
            "embed_url": cls.get("ollama", "embed_url"),
            "model": cls.get("ollama", "model"),
            "embed_model": cls.get("ollama", "embed_model"),
            "system_prompt": cls.get("ollama", "system_prompt"),
        }
    
    @classmethod
    def get_qdrant_config(cls) -> dict[str, Any]:
        """Qdrant設定を取得"""
        return {
            "host": cls.get("qdrant", "host"),
            "port": cls.get("qdrant", "port"),
            "collection_name": cls.get("qdrant", "collection_name"),
        }
