import toml
from pathlib import Path


class Config:
    _config = None

    @classmethod
    def load(cls, path="config.toml"):
        if cls._config is None:
            config_path = Path(path)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {path}")
            cls._config = toml.load(config_path.open(encoding="utf-8"))
        return cls._config

    @classmethod
    def get(cls, *keys, default=None):
        config = cls.load()
        for key in keys:
            if isinstance(config, dict) and key in config:
                config = config[key]
            else:
                return default
        return config
