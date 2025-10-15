# RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval

🌳 階層的文書検索システムの完全なオープンソース実装

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://python.langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 概要

RAPTORは、大規模文書から効率的に情報を検索するための革新的なRAG（Retrieval-Augmented Generation）手法です。従来のベクトル検索とは異なり、文書を階層的にクラスタリングし、ツリー構造で管理することで、より高速かつ文脈を保持した検索を実現します。

### 🎯 主な特徴

- **階層的ツリー構造**: 文書を再帰的にクラスタリングし、多階層のツリーを構築
- **効率的な検索**: O(log n)の検索複雑度で大規模文書にも対応
- **コンテキスト保持**: 各レベルで要約を生成し、大局的な理解を維持
- **100%オープンソース**: Granite Code 8B (LLM) + mxbai-embed-large (Embeddings)
- **スケーラブル**: 数十万文字の文書でも高速処理

## 🚀 クイックスタート

### 前提条件

1. **Ollamaのインストール** ([公式サイト](https://ollama.ai/))
2. **必要なモデルの取得**:
```bash
ollama pull granite-code:8b
ollama pull mxbai-embed-large
```

### インストール

```bash
# 必要なパッケージをインストール
pip install -U langchain langchain-community langchain-ollama scikit-learn numpy
```

### 基本的な使用方法

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from raptor import RAPTORRetriever

# LLMとEmbeddingsの初期化
llm = ChatOllama(model="granite-code:8b", base_url="http://localhost:11434", temperature=0)
embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")

# RAPTORレトリーバーの作成
raptor = RAPTORRetriever(
    embeddings_model=embeddings,
    llm=llm,
    max_clusters=3,   # 各レベルで最大3クラスタ
    max_depth=2,      # 最大2階層
)

# ドキュメントのインデックス化
raptor.index("your_document.txt")

# 検索実行
results = raptor.retrieve("検索クエリ", top_k=3)
for i, doc in enumerate(results, 1):
    print(f"Result {i}: {doc.page_content[:200]}...")
```

## 📊 性能ベンチマーク

### テスト環境
- **文書サイズ**: 624,212文字
- **チャンク数**: 864個
- **ハードウェア**: NVIDIA RTX 4060 Ti 16GB

### 結果

| 指標 | 値 |
|------|-----|
| **ツリー構築時間** | ~5分 (要約生成含む) |
| **検索時間** | <1秒 |
| **リーフノード数** | 9 |
| **圧縮率** | 96x (864 → 9) |
| **検索精度** | 0.63-0.67 (コサイン類似度) |

### 従来手法との比較

```
単純ベクトル検索:  O(n)   - 全チャンクをスキャン
RAPTOR:           O(log n) - 階層的探索
ColBERT:          O(n×m)  - トークンレベル比較
```

## 🏗️ アーキテクチャ

```
Root (864 docs)
├── Cluster 0 (344 docs)
│   ├── Cluster 0-0 (81 docs)  → Leaf
│   ├── Cluster 0-1 (135 docs) → Leaf
│   └── Cluster 0-2 (128 docs) → Leaf
├── Cluster 1 (219 docs)
│   ├── Cluster 1-0 (151 docs) → Leaf
│   ├── Cluster 1-1 (19 docs)  → Leaf
│   └── Cluster 1-2 (49 docs)  → Leaf
└── Cluster 2 (301 docs)
    ├── Cluster 2-0 (75 docs)  → Leaf
    ├── Cluster 2-1 (86 docs)  → Leaf
    └── Cluster 2-2 (140 docs) → Leaf
```

## 🔧 設定パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `max_clusters` | 5 | 各レベルでの最大クラスタ数 |
| `max_depth` | 3 | ツリーの最大深さ |
| `chunk_size` | 1000 | 文書チャンクのサイズ（文字数） |
| `chunk_overlap` | 200 | チャンク間のオーバーラップ |

### パラメータチューニングガイド

**小規模文書（<10万文字）**:
```python
raptor = RAPTORRetriever(
    max_clusters=2,
    max_depth=2,
    chunk_size=500
)
```

**中規模文書（10-50万文字）**:
```python
raptor = RAPTORRetriever(
    max_clusters=3,
    max_depth=2,
    chunk_size=1000
)
```

**大規模文書（>50万文字）**:
```python
raptor = RAPTORRetriever(
    max_clusters=5,
    max_depth=3,
    chunk_size=1500
)
```

## 📚 使用例

### 例1: 基本的な使用方法 (example.py)

test.txt を使った基本的なRAG検索：

```python
raptor.index("test.txt")
results = raptor.retrieve("philosophy", top_k=3)

# 出力例:
# Selected cluster 0 at depth 0 (similarity: 0.6691)
# Selected cluster 2 at depth 1 (similarity: 0.6587)
# → プラトンの哲学的信条に関する3件のドキュメントを取得
```

**実行方法**:
```bash
python example.py
```

### 例2: Wikipedia RAG (example2-wiki.py)

Wikipedia から動的にコンテンツを取得してRAG検索：

```python
import requests
from raptor import RAPTORRetriever

# Wikipedia APIからコンテンツ取得
def get_wikipedia_page(title: str) -> str:
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

# Miyazaki Hayaoのページを取得
wiki_content = get_wikipedia_page("Hayao_Miyazaki")

# 一時ファイルに保存してインデックス化
import tempfile
with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
    f.write(wiki_content)
    tmp_path = f.name

raptor.index(tmp_path)

# 検索実行
results = raptor.retrieve("What animation studio did Miyazaki found?", top_k=3)

# 出力例:
# ✅ Fetched 70,159 characters
# Split into 118 chunks
# Selected cluster 0 at depth 0 (similarity: 0.7885)
# Selected cluster 1 at depth 1 (similarity: 0.7720)
# → Studio Ghibli の設立に関する情報を取得
```

**実行方法**:
```bash
python example2-wiki.py
```

**主な機能**:
- 📥 Wikipedia API からリアルタイムでコンテンツ取得
- 🌳 70,159文字 → 118チャンク → 9リーフノードに階層化
- 🔍 複数クエリでの検索デモ（Studio Ghibli、受賞歴、代表作）
- 📊 高精度検索（類似度 0.73-0.78）

### 例3: 大規模論文処理 (example3-large-scale.py) 🚀

arXiv論文（370K文字）を使った大規模RAGの実証：

```python
from raptor import RAPTORRetriever
import requests

# arXiv論文をダウンロード
def download_arxiv_pdf(arxiv_id: str, output_path: str) -> bool:
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    response = requests.get(url, stream=True, timeout=60)
    with open(output_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    return True

# PDFからテキスト抽出
import PyPDF2
def pdf_to_text(pdf_path: str) -> str:
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text_parts = [page.extract_text() for page in reader.pages]
        return "\n\n".join(text_parts)

# 大規模文書用の最適化パラメータ
raptor = RAPTORRetriever(
    embeddings_model=embeddings,
    llm=llm,
    max_clusters=3,      # バランスの良いクラスタ数
    max_depth=2,         # 効率重視の深さ
    chunk_size=1200,     # ⭐ 単語の途切れを防ぐ最適サイズ
    chunk_overlap=250    # ⭐ 文脈保持のための十分なオーバーラップ
)

# RAG論文をインデックス化（arXiv:2508.06401）
raptor.index("rag_survey.txt")  # 370,694文字

# 複数の複雑なクエリで検証
queries = [
    "What are the main techniques used in RAG systems?",
    "What evaluation metrics are used for RAG systems?",
    "What are the main challenges in RAG implementation?",
    "How does retrieval-augmented generation compare to fine-tuning?",
    "What are the latest advancements in RAG research?"
]

for query in queries:
    results = raptor.retrieve(query, top_k=3)
    # 完全なコンテキストを持つ高品質な結果を取得
```

**実行方法**:
```bash
python example3-large-scale.py
```

**パフォーマンス実績**:
- 📊 **文書規模**: 370,694文字（0.37M）、48,399単語
- ⚡ **構築時間**: 2.5分（404チャンク処理）
- 🔍 **平均クエリ時間**: 2.55秒
- 🎯 **圧縮率**: 404チャンク → 9リーフノード（45倍圧縮）
- 🏆 **検索速度優位性**: 構築の60倍高速（ツリーナビゲーション vs 全探索）

**🎓 重要な教訓**:

1. **chunk_size の最適化が重要**
   - ❌ 1000文字: 単語が途中で途切れ、意味不明な結果
   - ✅ 1200文字: 完全なフレーズ・段落を保持、クエリ速度26%向上
   
2. **chunk_overlap の効果**
   - 250文字のオーバーラップで文脈の連続性を確保
   - チャンク間の意味的なギャップを埋める

3. **スケーラビリティの実証**
   - 370K文字でも2.5分で構築完了
   - 検索は常に2-3秒で一貫した高速性
   - メモリ使用量: ~1.5GB（効率的）

4. **パラメータ選択の指針**
   - 小規模（<100K）: `chunk_size=500-800`
   - 中規模（100-500K）: `chunk_size=1000-1200` ⭐推奨
   - 大規模（>500K）: `chunk_size=1500-2000`
- 🌳 70,159文字 → 118チャンク → 9リーフノードに階層化
- 🔍 複数クエリでの検索デモ（Studio Ghibli、受賞歴、代表作）
- 📊 高精度検索（類似度 0.73-0.78）

## 🔬 技術詳細

### クラスタリングアルゴリズム

RAPTORはK-meansクラスタリングを使用して、意味的に類似したドキュメントをグループ化します：

1. **ベクトル化**: mxbai-embed-large (1024次元)でドキュメントを埋め込み
2. **クラスタリング**: K-means (k=max_clusters)で分類
3. **要約生成**: Granite Code 8Bで各クラスタを要約
4. **再帰処理**: 各クラスタに対して再帰的に処理

### 検索アルゴリズム

```
1. ルートノードから開始
2. クエリと各クラスタの要約の類似度を計算
3. 最も類似度の高いクラスタを選択
4. 選択したクラスタの子ノードで再帰的に検索
5. リーフノードで最終的なドキュメントを取得
```

## 🎨 カスタマイズ

### カスタム要約プロンプトの使用

```python
class CustomRAPTOR(RAPTORRetriever):
    def summarize_cluster(self, documents):
        combined_text = "\n\n".join([doc.page_content for doc in documents])
        
        # カスタムプロンプト
        prompt = ChatPromptTemplate.from_template(
            "以下の技術文書を専門家向けに要約してください:\n\n{text}"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"text": combined_text[:4000]})
```

### 異なるLLMの使用

```python
# GPT-4の使用例
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)
raptor = RAPTORRetriever(embeddings_model=embeddings, llm=llm)
```

## 🐛 トラブルシューティング

### よくある問題

**Q: "Connection refused" エラーが出る**
```bash
# Ollamaが起動していることを確認
ollama serve
```

**Q: メモリ不足エラーが発生する**
```python
# chunk_sizeを小さくして調整
raptor = RAPTORRetriever(chunk_size=500)
```

**Q: 検索精度が低い**
```python
# より深い階層とより多くのクラスタを試す
raptor = RAPTORRetriever(max_clusters=5, max_depth=3)
```

## 📈 ロードマップ

- [ ] PostgreSQLベクトルストアとの統合
- [ ] 非同期処理対応
- [ ] Web UIの追加
- [ ] 複数パス検索のサポート
- [ ] 動的クラスタ数調整
- [ ] パフォーマンス最適化

## 🤝 貢献

プルリクエストを歓迎します！大きな変更の場合は、まずissueを開いて変更内容を議論してください。

## 📄 ライセンス

[MIT License](LICENSE)

## 👥 著者

- 開発者: [Your Name]
- GitHub: [@yourusername]

## 🙏 謝辞

- [LangChain](https://github.com/langchain-ai/langchain) - RAGフレームワーク
- [Ollama](https://ollama.ai/) - ローカルLLM実行環境
- RAPTOR論文著者 - 元のアルゴリズム設計

## 📖 参考文献

- RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval (Original Paper)
- LangChain Documentation
- scikit-learn K-means Implementation

---

⭐ このプロジェクトが役立った場合は、GitHubでスターをお願いします！
