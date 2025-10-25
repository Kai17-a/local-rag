#!/usr/bin/env python3
"""
RAGアプリケーション CLI版
"""
from __future__ import annotations

import sys
from pathlib import Path

from .adapters.factory import create_embedder, create_llm_client
from .adapters.vectorstore import QdrantVectorStore
from .core.exceptions import RAGException
from .services.document_ingest_service import DocumentIngestService
from .services.qa_service import QAService
from .utils.io import multiline_input, save_log
from .utils.logger import get_logger, setup_logging

logger = get_logger(__name__)


def initialize_services():
    """サービスを初期化"""
    try:
        logger.info("サービスを初期化中...")
        
        embedder = create_embedder()
        vector_store = QdrantVectorStore()
        vector_store.init_collection()
        
        llm_client = create_llm_client()
        qa_service = QAService(llm_client, embedder, vector_store)
        document_service = DocumentIngestService(embedder, vector_store)
        
        logger.info("初期化完了")
        return qa_service, document_service
        
    except Exception as e:
        logger.error(f"初期化エラー: {e}")
        raise RAGException(f"サービス初期化に失敗しました: {e}")


def handle_document_ingest(document_service):
    """文書登録処理"""
    while True:
        target_dir = input("文書ディレクトリのパス (戻る: q): ").strip()
        
        if target_dir.lower() == 'q':
            return
            
        if not target_dir:
            print("パスを入力してください")
            continue
            
        if not Path(target_dir).exists():
            print(f"ディレクトリが見つかりません: {target_dir}")
            continue
            
        try:
            document_service.ingest(target_dir)
            print("文書の登録が完了しました")
            break
        except RAGException as e:
            print(f"エラー: {e}")
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            print("予期しないエラーが発生しました")


def handle_qa(qa_service):
    """質問応答処理"""
    while True:
        print("\n" + "="*50)
        query = multiline_input("質問を入力してください (戻る: q): ")
        
        if query.strip().lower() == 'q':
            return
            
        if not query.strip():
            print("質問を入力してください")
            continue
            
        try:
            print("\n回答を生成中...")
            answer = qa_service.answer(query)
            
            print(f"\n【回答】\n{answer}")
            
            # ログ保存
            try:
                log_file = save_log(query, answer)
                print(f"\nログを保存しました: {log_file}")
            except Exception as e:
                logger.warning(f"ログ保存に失敗: {e}")
                
        except RAGException as e:
            print(f"エラー: {e}")
        except Exception as e:
            logger.error(f"予期しないエラー: {e}")
            print("予期しないエラーが発生しました")


def main():
    """メイン処理"""
    setup_logging()
    
    try:
        print("RAGアプリケーション CLI版")
        print("="*50)
        
        qa_service, document_service = initialize_services()
        
        while True:
            print("\n【メニュー】")
            print("1: 質問・検索")
            print("2: 文書登録")
            print("q: 終了")
            
            choice = input("選択してください: ").strip()
            
            if choice.lower() == 'q':
                print("アプリケーションを終了します")
                break
            elif choice == "1":
                handle_qa(qa_service)
            elif choice == "2":
                handle_document_ingest(document_service)
            else:
                print("1, 2, または q を入力してください")
                
    except KeyboardInterrupt:
        print("\n\nアプリケーションが中断されました")
        sys.exit(0)
    except RAGException as e:
        logger.error(f"アプリケーションエラー: {e}")
        print(f"エラー: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"予期しないエラー: {e}")
        print("予期しないエラーが発生しました")
        sys.exit(1)


if __name__ == "__main__":
    main()
