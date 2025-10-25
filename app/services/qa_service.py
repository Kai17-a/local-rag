from __future__ import annotations

from ..core.exceptions import RAGException
from ..core.models import QAResult
from ..utils.logger import get_logger

logger = get_logger(__name__)


class QAService:
    """質問応答サービス"""
    
    def __init__(self, llm_client, embedder, vector_store) -> None:
        self.llm_client = llm_client
        self.embedder = embedder
        self.vector_store = vector_store

    def answer(self, query: str) -> str:
        """質問に対する回答を生成"""
        try:
            # 質問を埋め込みベクトルに変換
            query_embed = self.embedder.embed(query)
            
            # 関連文書を検索
            search_results = self.vector_store.search(query_embed)
            
            if not search_results:
                logger.info("関連する文書が見つかりませんでした")
                return "関連する資料がありませんでした"
            else:
                return self._generate_answer_with_sources(query, search_results)
                    
        except Exception as e:
            logger.error(f"質問応答処理エラー: {e}")
            raise RAGException(f"回答生成に失敗しました: {e}") from e
    
    def _generate_answer_with_sources(self, query: str, search_results) -> str:
        """検索結果を使って回答を生成"""
        # 検索結果から文脈とソースを抽出
        context_texts = [result.text for result in search_results if result.text]
        sources = {result.source for result in search_results if result.source}

        context_text = "\n".join(context_texts)
        
        # LLMで回答を生成
        answer = self.llm_client.chat(query, context_text)

        # ソース情報を追加
        source_list = sorted(sources)
        if not source_list:
            return answer
        else:
            source_text = "\n".join(f"- {src}" for src in source_list)
            return f"{answer}\n\n---\n参考資料:\n{source_text}"
    
    def get_qa_result(self, query: str) -> QAResult:
        """構造化された質問応答結果を取得"""
        try:
            query_embed = self.embedder.embed(query)
            search_results = self.vector_store.search(query_embed)
            
            if not search_results:
                return QAResult(
                    question=query,
                    answer="関連する資料がありませんでした",
                    sources=[]
                )
            else:
                context_texts = [r.text for r in search_results if r.text]
                sources = list({r.source for r in search_results if r.source})
                
                context_text = "\n".join(context_texts)
                answer = self.llm_client.chat(query, context_text)
                
                return QAResult(
                    question=query,
                    answer=answer,
                    sources=sources
                )
            
        except Exception as e:
            logger.error(f"QA結果取得エラー: {e}")
            raise RAGException(f"QA結果の取得に失敗しました: {e}") from e
