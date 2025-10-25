#!/usr/bin/env python3
"""
FastAPI サーバー起動スクリプト
"""
import uvicorn
from app.api_main import app

if __name__ == "__main__":
    uvicorn.run(
        "app.api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )