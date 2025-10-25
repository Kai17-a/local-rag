# Local RAG

## 機能

- PDF・テキストファイルの自動処理とベクトル化
- 自然言語での質問に対する文書ベースの回答生成
- FastAPI WebサーバーとCLI対応
- Qdrantによる高速ベクトル検索

## 使用方法

### CLI実行

```bash
python run_cli.py
```

### API実行

```bash
python run_api.py
```

サーバー: `http://localhost:8000`

### APIエンドポイント

- `GET /?q=質問内容` - 質問応答
- `POST /documents/` - ディレクトリ内文書一括登録
- `POST /upload/` - ファイルアップロード
- `POST /text/` - テキスト直接登録

## 設定

`app/config.toml`でモデルタイプを選択:

```toml
model_type = "ollama"  # または "docker"
```