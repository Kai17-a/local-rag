"""
カスタム例外クラス
"""


class RAGException(Exception):
    """RAGアプリケーションの基底例外"""
    pass


class EmbeddingError(RAGException):
    """埋め込み生成エラー"""
    pass


class VectorStoreError(RAGException):
    """ベクターストアエラー"""
    pass


class DocumentProcessingError(RAGException):
    """文書処理エラー"""
    pass


class LLMError(RAGException):
    """LLMエラー"""
    pass