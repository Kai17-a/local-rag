#!/usr/bin/env python3
"""
RAGアプリケーションのエントリーポイント
"""
import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.cli_main import main

if __name__ == "__main__":
    main()