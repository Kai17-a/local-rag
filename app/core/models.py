"""
ドメインモデル定義
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Document(BaseModel):
    """文書を表すモデル"""
    content: str = Field(..., description="文書の内容")
    source: str = Field(..., description="文書のソース")
    chunk_id: int | None = Field(None, description="チャンクID")
    
    model_config = {"frozen": True}


class SearchResult(BaseModel):
    """検索結果を表すモデル"""
    text: str = Field(..., description="検索結果のテキスト")
    source: str = Field(..., description="ソース名")
    score: float = Field(..., ge=0.0, le=1.0, description="類似度スコア")
    chunk_id: int | None = Field(None, description="チャンクID")
    
    model_config = {"frozen": True}


class QAResult(BaseModel):
    """質問応答結果を表すモデル"""
    question: str = Field(..., description="質問内容")
    answer: str = Field(..., description="回答内容")
    sources: list[str] = Field(default_factory=list, description="参考資料のリスト")
    
    model_config = {"frozen": True}

# APIレスポンスモデル
class QAResponse(BaseModel):
    """質問応答APIのレスポンス"""
    question: str = Field(..., description="質問内容")
    answer: str = Field(..., description="回答内容")
    status: str = Field(default="success", description="処理ステータス")


class DocumentIngestResponse(BaseModel):
    """文書登録APIのレスポンス"""
    message: str = Field(..., description="処理結果メッセージ")
    directory: str | None = Field(None, description="処理したディレクトリ")
    status: str = Field(default="success", description="処理ステータス")


class FileUploadResponse(BaseModel):
    """ファイルアップロードAPIのレスポンス"""
    message: str = Field(..., description="処理結果メッセージ")
    processed_files: int = Field(..., ge=0, description="処理されたファイル数")
    skipped_files: list[str] = Field(default_factory=list, description="スキップされたファイル")
    status: str = Field(default="success", description="処理ステータス")


class TextRegisterResponse(BaseModel):
    """テキスト登録APIのレスポンス"""
    message: str = Field(..., description="処理結果メッセージ")
    chunks: int = Field(..., ge=0, description="登録されたチャンク数")
    source: str = Field(..., description="ソース名")
    status: str = Field(default="success", description="処理ステータス")


class ErrorResponse(BaseModel):
    """エラーレスポンス"""
    error: str = Field(..., description="エラーメッセージ")
    status: str = Field(default="error", description="処理ステータス")


# APIリクエストモデル
class DirectoryRequest(BaseModel):
    """ディレクトリ指定リクエスト"""
    directory: str = Field(..., min_length=1, description="処理対象のディレクトリパス")


class RegisterTextRequest(BaseModel):
    """テキスト登録リクエスト"""
    text: str = Field(..., min_length=1, description="登録するテキスト")
    source: str = Field(default="input_text", description="ソース名")