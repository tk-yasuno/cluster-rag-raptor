# RAPTOR with HDBSCAN - ノイズ除去機能付き階層的クラスタリング

## 🎯 概要

このプロジェクトは、RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) にHDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) を統合し、**意味の薄いチャンクの自動除去**と**クラスタ数の自動決定**を実現します。

## 🌟 HDBSCAN導入の利点

### ✅ 主要な利点

1. **クラスタ数の自動決定**
   - K-meansのような`k`パラメータ不要
   - 文書構造に自然に適応
   - データ駆動のクラスタ形成

2. **ノイズ検出・除去**
   - 意味の薄いチャンクを自動検出
   - ノイズラベル `-1` で識別
   - 検索精度の向上

3. **階層性の活用**
   - Condensed treeによる真の階層構造
   - 密度ベースの自然な階層形成
   - RAPTORのツリー構造と相性良好

4. **高次元埋め込みとの相性**
   - mxbai-embed-largeのような大規模embeddings向き
   - コサイン距離メトリック対応
   - 意味的類似度を正確に捉える

## 🆚 手法比較

| 手法 | クラスタ数決定 | ノイズ除去 | 階層性 | 主な用途 |
|------|---------------|-----------|--------|---------|
| **K-means** | 手動固定 | ❌ | ❌ | 既知のクラスタ構造 |
| **GMM + BIC** | 自動（BIC最適化） | ❌ | ❌ | 複雑な分布 |
| **HDBSCAN** | 自動（密度ベース） | ✅ | ✅ | 未知構造 + 品質重視 |

## 🛠️ インストール

```bash
# 基本パッケージ
pip install -U langchain langchain-community langchain-ollama scikit-learn numpy

# HDBSCAN
pip install hdbscan
```

## 🚀 使い方

### 基本的な使用例

```python
from langchain_ollama import ChatOllama, OllamaEmbeddings
from raptor_hdbscan import RAPTORRetrieverHDBSCAN

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

# RAPTOR with HDBSCAN
raptor = RAPTORRetrieverHDBSCAN(
    embeddings_model=embeddings_model,
    llm=llm,
    min_cluster_size=15,    # 重要: クラスタの最小サイズ
    min_samples=5,          # 密度推定用
    max_depth=2,
    chunk_size=1000,
    chunk_overlap=200,
    metric='cosine',        # 'euclidean' or 'cosine'
    exclude_noise=True      # ノイズ除去ON
)

# インデックス化
raptor.index("your_document.txt")

# 検索
results = raptor.retrieve("your query", top_k=3)

# ノイズ統計を確認
print(f"Removed noise chunks: {raptor.noise_stats['total_noise_chunks']}")
```

### 比較実験の実行

```bash
# 全手法の比較（K-means, GMM+BIC, HDBSCAN）
python example_hdbscan_comparison.py

# パラメータチューニング実験
# example_hdbscan_comparison.py内のtest_different_parameters()を有効化
```

## 🔧 パラメータチューニングガイド

### `min_cluster_size` の選び方

チャンクサイズに基づく推奨値:

```python
# 小さいドキュメント (100-500文字)
min_cluster_size = 5-10

# 中規模ドキュメント (500-1500文字)
min_cluster_size = 10-20

# 大規模ドキュメント (1500文字以上)
min_cluster_size = 15-30
```

**ルール:**
- 値が**大きい**ほど保守的（ノイズ多め、クラスタ少なめ）
- 値が**小さい**ほど細かい粒度（ノイズ少なめ、クラスタ多め）

### `min_samples` の選び方

一般的なルール: `min_samples = min_cluster_size / 3`

```python
min_cluster_size = 15
min_samples = 5  # 15 / 3 = 5
```

### `metric` の選び方

| メトリック | 推奨ケース | 特徴 |
|-----------|-----------|------|
| `'euclidean'` | 汎用的な用途 | 標準的な距離 |
| `'cosine'` | 意味的埋め込み | 方向性を重視、mxbai-embed-large推奨 |

### `exclude_noise` の設定

```python
# ノイズを除外（推奨）
exclude_noise = True

# ノイズも保持（実験的）
exclude_noise = False
```

## 📊 実験結果例

### ノイズ除去効果

```
🗑️  Noise Statistics
================================================================================
   Total noise chunks excluded: 12
   Noise by depth:
     Depth 0: 8 chunks
     Depth 1: 4 chunks
================================================================================
```

### 性能比較

| Method | Build Time | Query Time | Noise Removed | Note |
|--------|-----------|-----------|---------------|------|
| K-means (fixed) | 15.23秒 | 0.045秒 | 0 | 固定3クラスタ |
| GMM + BIC | 18.45秒 | 0.047秒 | 0 | 自動最適化 |
| HDBSCAN (euclidean) | 16.89秒 | 0.044秒 | 12 | 自動+ノイズ除去 |
| HDBSCAN (cosine) | 17.12秒 | 0.043秒 | 15 | 意味的距離 |

## 🧪 ベンチマーク実験

### 実験1: 基本比較

```bash
python example_hdbscan_comparison.py
```

**確認項目:**
- リーフノード数
- 圧縮率
- 検索精度
- ノイズ除去数

### 実験2: パラメータチューニング

`example_hdbscan_comparison.py` 内で `test_different_parameters()` を有効化:

```python
if __name__ == "__main__":
    compare_clustering_methods()
    test_different_parameters()  # コメント解除
```

**テストする値:**
- `min_cluster_size`: [5, 10, 15, 20]
- `metric`: ['euclidean', 'cosine']

## 🎓 理論背景

### HDBSCANの動作原理

1. **相互到達距離の計算**
   - 各点間の密度を考慮した距離
   - ノイズに強い距離メトリック

2. **階層的クラスタリング**
   - Single linkage clustering
   - Minimum spanning tree構築

3. **Condensed tree生成**
   - 階層構造の安定部分を抽出
   - 真のクラスタを識別

4. **クラスタ抽出**
   - Excess of Mass (EoM)
   - 最も安定したクラスタを選択

### ノイズ検出の仕組み

```python
# ラベル -1 = ノイズ
cluster_labels = clusterer.fit_predict(embeddings)

# ノイズの定義:
# - 密度が低い領域の点
# - どのクラスタにも属さない孤立点
# - min_cluster_sizeを満たさない小グループ
```

## 📈 最適化のヒント

### 1. チャンクサイズの調整

```python
# 細かいチャンク → 多くのクラスタ
chunk_size = 500
min_cluster_size = 10

# 大きいチャンク → 少ないクラスタ
chunk_size = 2000
min_cluster_size = 20
```

### 2. 距離メトリックの選択

```python
# 意味的埋め込みにはcosine推奨
embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")
metric = 'cosine'

# 汎用的な埋め込みにはeuclidean
metric = 'euclidean'
```

### 3. 階層の深さ調整

```python
# 浅い階層 = 速い、粗い
max_depth = 1

# 深い階層 = 遅い、細かい
max_depth = 3
```

## 🔍 デバッグ・検証

### ノイズ統計の確認

```python
raptor.index("document.txt")

# 全体統計
print(raptor.noise_stats['total_noise_chunks'])

# 深さ別
print(raptor.noise_stats['noise_by_depth'])
```

### Condensed Treeの可視化（オプション）

```python
import matplotlib.pyplot as plt

# ツリー構造からclustererを取得
stats = raptor.tree_structure['hdbscan_stats']
clusterer = stats['clusterer']

# 可視化
clusterer.condensed_tree_.plot(select_clusters=True)
plt.show()
```

## 💡 ベストプラクティス

### ✅ DO
- **コサイン距離を使用** (意味的埋め込みの場合)
- **min_cluster_sizeを文書サイズに合わせる**
- **ノイズ統計を確認**して調整
- **複数のパラメータで実験**

### ❌ DON'T
- min_cluster_sizeを極端に小さく設定（<5）
- サンプル数が少ない時に大きいmin_cluster_size
- ノイズ除去なしでHDBSCANを使う（意味がない）

## 🆕 今後の拡張案

1. **Condensed treeの明示的な活用**
   ```python
   # condensed treeから直接階層を抽出
   tree = clusterer.condensed_tree_
   # カスタム階層マッピング
   ```

2. **動的パラメータ調整**
   ```python
   # 深さに応じてmin_cluster_sizeを変更
   min_cluster_size = base_size * (depth + 1)
   ```

3. **ソフトクラスタリング**
   ```python
   # HDBSCANの確率的メンバーシップ活用
   soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
   ```

## 📚 参考文献

- [HDBSCAN Documentation](https://hdbscan.readthedocs.io/)
- [RAPTOR Paper](https://arxiv.org/abs/2401.18059)
- [How HDBSCAN Works](https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html)

## 🤝 貢献

プルリクエスト、イシュー、フィードバックを歓迎します！

## 📄 ライセンス

MIT License

---

**作成者:** Takato Yasuno  
**日付:** 2025年10月16日
