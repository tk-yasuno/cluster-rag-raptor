# RAPTOR クイックスタートガイド

⚡ 5分でRAPTORを使い始めるための簡易ガイド

## 📦 セットアップ（3分）

### ステップ1: Ollamaのインストール

```bash
# Windows/Mac/Linuxで以下からダウンロード
# https://ollama.ai/

# インストール後、ターミナルで確認
ollama --version
```

### ステップ2: モデルのダウンロード

```bash
# LLMモデル（要約生成用）
ollama pull granite-code:8b

# Embeddingモデル（ベクトル化用）
ollama pull mxbai-embed-large
```

### ステップ3: Pythonパッケージのインストール

```bash
pip install -U langchain langchain-community langchain-ollama scikit-learn numpy
```

## 🚀 基本的な使い方（2分）

### 最小限のコード例

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from raptor import RAPTORRetriever

# 1. モデルの初期化
llm = ChatOllama(
    model="granite-code:8b",
    base_url="http://localhost:11434",
    temperature=0
)

embeddings = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://localhost:11434"
)

# 2. RAPTORの作成
raptor = RAPTORRetriever(
    embeddings_model=embeddings,
    llm=llm,
    max_clusters=3,  # シンプルに3クラスタ
    max_depth=2      # 2階層まで
)

# 3. ドキュメントのインデックス化
raptor.index("your_document.txt")

# 4. 検索実行
results = raptor.retrieve("検索したい内容", top_k=3)

# 5. 結果の表示
for i, doc in enumerate(results, 1):
    print(f"\n=== 結果 {i} ===")
    print(doc.page_content)
```

## 📝 完全な実行例

### 例1: ローカルファイルを使った検索 (example.py)

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from raptor import RAPTORRetriever

# モデル初期化
llm = ChatOllama(
    model="granite-code:8b",
    base_url="http://localhost:11434",
    temperature=0
)

embeddings_model = OllamaEmbeddings(
    model="mxbai-embed-large",
    base_url="http://localhost:11434"
)

# RAPTORレトリーバー作成
raptor = RAPTORRetriever(
    embeddings_model=embeddings_model,
    llm=llm,
    max_clusters=3,
    max_depth=2,
    chunk_size=1000,
    chunk_overlap=200
)

print("📚 ドキュメントをインデックス化中...")
raptor.index("example_document.txt")

print("\n🔍 検索を実行中...")

# クエリ1
query1 = "主要なトピックは何ですか？"
results1 = raptor.retrieve(query1, top_k=3)

print(f"\n=== '{query1}' の検索結果 ===")
for i, doc in enumerate(results1, 1):
    print(f"\n結果 {i}:")
    print(doc.page_content[:300])
    print("...")

# クエリ2
query2 = "具体的な事例を教えてください"
results2 = raptor.retrieve(query2, top_k=3)

print(f"\n=== '{query2}' の検索結果 ===")
for i, doc in enumerate(results2, 1):
    print(f"\n結果 {i}:")
    print(doc.page_content[:300])
    print("...")
```

**実行方法**:
```bash
python example.py
```

### 例2: Wikipedia から動的に取得 (example2-wiki.py)

Wikipedia APIを使ってリアルタイムでコンテンツを取得し、RAG検索を実行する例：

```python
import requests
import tempfile
import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from raptor import RAPTORRetriever

def get_wikipedia_page(title: str) -> str:
    """Wikipedia APIからページコンテンツを取得"""
    URL = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }
    headers = {"User-Agent": "RAPTOR_RAG_Example/1.0"}
    response = requests.get(URL, params=params, headers=headers)
    data = response.json()
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None

# モデル初期化
llm = ChatOllama(model="granite-code:8b", temperature=0)
embeddings = OllamaEmbeddings(model="mxbai-embed-large")

# RAPTORレトリーバー作成
raptor = RAPTORRetriever(
    embeddings_model=embeddings,
    llm=llm,
    max_clusters=3,
    max_depth=2
)

# Wikipedia から宮崎駿のページを取得
print("🌐 Fetching Wikipedia page...")
wiki_content = get_wikipedia_page("Hayao_Miyazaki")
print(f"✅ Fetched {len(wiki_content):,} characters")

# 一時ファイルに保存してインデックス化
with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp:
    tmp.write(wiki_content)
    tmp_path = tmp.name

try:
    print("📊 Indexing Wikipedia content...")
    raptor.index(tmp_path)
    
    # 複数クエリで検索
    queries = [
        "What animation studio did Miyazaki found?",
        "What awards has Miyazaki received?",
        "What are Miyazaki's most famous films?"
    ]
    
    for query in queries:
        print(f"\n🔍 Query: '{query}'")
        results = raptor.retrieve(query, top_k=3)
        
        for i, doc in enumerate(results, 1):
            print(f"\n--- Result {i} ---")
            print(doc.page_content[:200])
            
finally:
    # 一時ファイルを削除
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

print("\n✅ Wikipedia RAG completed!")
```

**実行方法**:
```bash
python example2-wiki.py
```

**出力例**:
```
🌐 Fetching Wikipedia page...
✅ Fetched 70,159 characters
📊 Indexing Wikipedia content...
Split into 118 chunks

🔍 Query: 'What animation studio did Miyazaki found?'
Selected cluster 0 at depth 0 (similarity: 0.7885)
Selected cluster 1 at depth 1 (similarity: 0.7720)

--- Result 1 ---
=== Studio Ghibli ===
==== Foundation and Laputa (1985–1987) ====...
```

**主な特徴**:
- 📥 Wikipedia APIから動的にコンテンツ取得
- 🌳 70,159文字 → 118チャンク → 階層化
- 🔍 高精度検索（類似度 0.73-0.78）
- 🌍 任意のWikipediaページで利用可能

## 🎯 ユースケース別の設定

### ケース1: 小さな文書（<10万文字）

```python
raptor = RAPTORRetriever(
    embeddings_model=embeddings,
    llm=llm,
    max_clusters=2,   # 少ないクラスタ
    max_depth=2,      # 浅い階層
    chunk_size=500    # 小さいチャンク
)
```

**適用例**: ブログ記事、短い論文、ドキュメンテーション

### ケース2: 中規模文書（10-50万文字）

```python
raptor = RAPTORRetriever(
    embeddings_model=embeddings,
    llm=llm,
    max_clusters=3,   # バランスの良いクラスタ数
    max_depth=2,      # 標準的な階層
    chunk_size=1000   # 標準的なチャンクサイズ
)
```

**適用例**: 技術書、長編記事、研究論文

### ケース3: 大規模文書（>50万文字）

```python
raptor = RAPTORRetriever(
    embeddings_model=embeddings,
    llm=llm,
    max_clusters=5,   # 多くのクラスタ
    max_depth=3,      # 深い階層
    chunk_size=1500   # 大きいチャンク
)
```

**適用例**: 書籍全体、大規模データセット、複数文書の結合

## 💡 よくある質問

### Q1: どのくらいの時間がかかりますか？

**インデックス化**: 
- 小規模（<10万文字）: ~1分
- 中規模（10-50万文字）: ~5分
- 大規模（>50万文字）: ~15分

**検索**: 常に1秒未満

### Q2: メモリ使用量は？

- **最小要件**: 8GB RAM
- **推奨**: 16GB RAM
- **GPU**: オプション（Ollamaが自動的に使用）

### Q3: 複数の文書をインデックス化できますか？

はい！複数ファイルを結合してください：

```python
# 複数ファイルを読み込み
documents = []
for file in ["doc1.txt", "doc2.txt", "doc3.txt"]:
    with open(file, 'r', encoding='utf-8') as f:
        documents.append(f.read())

# 結合して一時ファイルに保存
combined = "\n\n".join(documents)
with open("combined.txt", 'w', encoding='utf-8') as f:
    f.write(combined)

# インデックス化
raptor.index("combined.txt")
```

### Q4: 検索結果の品質を向上させるには？

1. **より深い階層を試す**: `max_depth=3`
2. **クラスタ数を増やす**: `max_clusters=5`
3. **チャンクサイズを調整**: 大きいテキストなら `chunk_size=1500`
4. **オーバーラップを増やす**: `chunk_overlap=300`

### Q5: エラーが出た場合は？

**"Connection refused"**:
```bash
# Ollamaが起動していない
ollama serve
```

**"Model not found"**:
```bash
# モデルをダウンロード
ollama pull granite-code:8b
ollama pull mxbai-embed-large
```

**メモリ不足**:
```python
# チャンクサイズを小さく
raptor = RAPTORRetriever(chunk_size=500)
```

## 📊 期待される出力例

```
=== Starting RAPTOR Indexing ===
Loaded document length: 624212 characters
Split into 864 chunks

=== Building tree at depth 0 with 864 documents ===
Cluster 0: 344 documents
Cluster 1: 219 documents
Cluster 2: 301 documents

=== Building tree at depth 1 with 344 documents ===
...

=== RAPTOR Tree Construction Complete ===

=== Searching for: 'philosophy' ===
Selected cluster 0 at depth 0 (similarity: 0.6691)
Selected cluster 2 at depth 1 (similarity: 0.6587)

=== Top 3 Results ===
Result 1: プラトンの哲学的信条について...
Result 2: 理想的知識論に関する議論...
Result 3: プラトン的愛の概念...
```

## 🎓 次のステップ

### 学習リソース

1. **詳細ドキュメント**: [README.md](README.md)を参照
2. **カスタマイズ**: 要約プロンプトやLLMを変更
3. **最適化**: パラメータチューニングで精度向上

### 発展的な使い方

```python
# カスタム要約プロンプト
class MyRAPTOR(RAPTORRetriever):
    def summarize_cluster(self, documents):
        # 独自の要約ロジック
        pass

# 複数クエリの並列実行
queries = ["query1", "query2", "query3"]
results = [raptor.retrieve(q, top_k=3) for q in queries]
```

## 🔧 トラブルシューティングチェックリスト

- [ ] Ollamaがインストールされている
- [ ] `ollama serve` が実行中
- [ ] モデルがダウンロード済み（`ollama list`で確認）
- [ ] Pythonパッケージがインストール済み
- [ ] 文書ファイルが存在し、UTF-8エンコーディング
- [ ] 十分なメモリ（推奨16GB）

## 📞 サポート

問題が解決しない場合:
1. GitHubのIssueを確認
2. 新しいIssueを作成
3. README.mdのトラブルシューティングセクションを参照

---

🎉 これでRAPTORを使い始める準備ができました！
