# HDBSCAN版RAPTOR クイックスタート

## 📦 インストール

```bash
# HDBSCANを追加インストール
pip install hdbscan

# または全パッケージを一括インストール
pip install -r requirements.txt
```

## 🚀 5分で始める

### Step 1: Ollamaモデルの準備

```bash
# LLM (要約用)
ollama pull granite-code:8b

# Embeddings (ベクトル化用)
ollama pull mxbai-embed-large
```

### Step 2: 基本的な実行

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from raptor_hdbscan import RAPTORRetrieverHDBSCAN

# モデル初期化
llm = ChatOllama(model="granite-code:8b", base_url="http://localhost:11434")
embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://localhost:11434")

# HDBSCAN版RAPTOR
raptor = RAPTORRetrieverHDBSCAN(
    embeddings_model=embeddings,
    llm=llm,
    min_cluster_size=15,  # ⭐ 重要パラメータ
    exclude_noise=True     # ノイズ除去ON
)

# インデックス & 検索
raptor.index("test.txt")
results = raptor.retrieve("your query")
```

### Step 3: 比較実験

```bash
python example_hdbscan_comparison.py
```

## 🎯 パラメータ早見表

| パラメータ | 推奨値 | 説明 |
|-----------|--------|------|
| `min_cluster_size` | 10-20 | クラスタの最小サイズ |
| `min_samples` | 5-7 | `min_cluster_size/3`が目安 |
| `metric` | `'cosine'` | 意味的embeddings向け |
| `exclude_noise` | `True` | ノイズ除去を有効化 |
| `max_depth` | 2-3 | ツリーの深さ |

## 🔧 チューニング例

### 細かいクラスタリング

```python
raptor = RAPTORRetrieverHDBSCAN(
    min_cluster_size=5,   # 小さく
    min_samples=2,
    metric='cosine'
)
```

### 保守的なクラスタリング

```python
raptor = RAPTORRetrieverHDBSCAN(
    min_cluster_size=25,  # 大きく
    min_samples=8,
    metric='euclidean'
)
```

## 📊 結果の確認

```python
# インデックス後
print(f"除去されたノイズ: {raptor.noise_stats['total_noise_chunks']}")
print(f"深さ別ノイズ: {raptor.noise_stats['noise_by_depth']}")

# 検索結果
for i, doc in enumerate(results):
    print(f"{i+1}. 類似度: {doc.metadata.get('similarity')}")
    print(f"   内容: {doc.page_content[:100]}...")
```

## 🆚 他手法との違い

```python
# K-means (固定クラスタ数)
from raptor import RAPTORRetriever
raptor_kmeans = RAPTORRetriever(max_clusters=3)

# GMM + BIC (自動最適化)
from raptor_gmm import RAPTORRetrieverGMM
raptor_gmm = RAPTORRetrieverGMM(use_bic=True)

# HDBSCAN (自動 + ノイズ除去)
from raptor_hdbscan import RAPTORRetrieverHDBSCAN
raptor_hdbscan = RAPTORRetrieverHDBSCAN(exclude_noise=True)
```

## 💡 よくある質問

**Q: min_cluster_sizeはどう決める？**  
A: チャンクサイズの1-3%を目安に。1000文字チャンクなら10-20。

**Q: ノイズが多すぎる/少なすぎる？**  
A: `min_cluster_size`を調整。大きくするとノイズ増、小さくするとノイズ減。

**Q: どの距離メトリックを使う？**  
A: 意味的embeddings（mxbai等）は`cosine`、汎用は`euclidean`。

**Q: 実行が遅い？**  
A: `max_depth`を減らす、`chunk_size`を大きくする。

## 📖 詳細ドキュメント

- [HDBSCAN_GUIDE.md](HDBSCAN_GUIDE.md) - 完全ガイド
- [example_hdbscan_comparison.py](example_hdbscan_comparison.py) - 比較実験コード

## 🎓 次のステップ

1. ✅ 基本実行を試す
2. ✅ パラメータを調整
3. ✅ 比較実験を実行
4. ✅ 自分のデータで検証

---

Happy Clustering! 🚀
