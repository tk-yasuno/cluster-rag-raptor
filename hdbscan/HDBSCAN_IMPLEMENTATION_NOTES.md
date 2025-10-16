# HDBSCAN版RAPTOR 実装ノート

## 📋 実装の要点

### 主要な変更点

1. **クラスタリングアルゴリズム**
   ```python
   # 従来: K-means (固定クラスタ数)
   kmeans = KMeans(n_clusters=max_clusters)
   labels = kmeans.fit_predict(embeddings)
   
   # HDBSCAN版: 密度ベース (自動クラスタ数)
   clusterer = hdbscan.HDBSCAN(
       min_cluster_size=15,
       min_samples=5,
       metric='cosine'
   )
   labels = clusterer.fit_predict(embeddings)
   # labels == -1 はノイズ
   ```

2. **ノイズ処理**
   ```python
   # ノイズ除外
   for label in unique_labels:
       if label == -1:
           if self.exclude_noise:
               print(f"🗑️  Excluding {n_noise} noise points")
               continue
   ```

3. **統計情報の追跡**
   ```python
   self.noise_stats = {
       'total_noise_chunks': 0,
       'noise_by_depth': {}
   }
   ```

## 🔬 技術的詳細

### HDBSCANパラメータの影響

#### `min_cluster_size`
- **小さい値 (5-10)**: 多くの小クラスタ、少ないノイズ
- **中間値 (10-20)**: バランス型（推奨）
- **大きい値 (20-30)**: 少ない大クラスタ、多いノイズ

#### `min_samples`
- **公式推奨**: `min_cluster_size`と同じ値
- **一般的**: `min_cluster_size / 3`
- **効果**: ノイズ判定の厳しさを調整

#### `metric`
- **`'euclidean'`**: L2距離、汎用的
- **`'cosine'`**: コサイン距離、方向性重視
- **`'manhattan'`**: L1距離、外れ値に強い

### 距離メトリック選択の理論

```python
# mxbai-embed-largeのような意味的embeddings
# → 方向性が重要 → cosine推奨
metric = 'cosine'

# 一般的な数値特徴量
# → 距離が重要 → euclidean推奨
metric = 'euclidean'
```

**理由**: 意味的embeddingsは正規化されており、ベクトルの方向が意味を表す。
長さは重要でない → コサイン類似度が適切。

## 🎯 実装上の工夫

### 1. サンプル数不足への対応

```python
if n_samples < self.min_cluster_size:
    print(f"⚠️  Sample size too small")
    return np.zeros(n_samples, dtype=int), {
        'n_clusters': 1,
        'n_noise': 0
    }
```

### 2. 再帰終了条件の調整

```python
# K-means版: max_clustersと比較
if len(documents) <= self.max_clusters:
    return leaf_node

# HDBSCAN版: min_cluster_sizeと比較
if len(documents) < self.min_cluster_size:
    return leaf_node
```

### 3. クラスタ0個の処理

```python
if stats['n_clusters'] == 0:
    print(f"⚠️  No clusters found. Creating leaf node.")
    return {
        'depth': depth,
        'documents': documents,
        'is_leaf': True
    }
```

## 📊 性能最適化

### メモリ使用量の削減

```python
# condensed treeは大きいので、必要時のみ保持
# stats['clusterer'] に格納（オプション）
if need_condensed_tree:
    stats['clusterer'] = clusterer
```

### 並列処理の活用

```python
clusterer = hdbscan.HDBSCAN(
    core_dist_n_jobs=-1  # 全CPUコアを使用
)
```

## 🧪 実験設計のベストプラクティス

### 比較実験のポイント

1. **同一データで複数手法を比較**
   ```python
   methods = ['kmeans', 'gmm_bic', 'hdbscan']
   for method in methods:
       build_time = measure_build(method)
       search_quality = measure_search(method)
   ```

2. **評価指標**
   - ビルド時間
   - クエリ時間
   - 検索精度（top-k similarity）
   - ノイズ除去数
   - クラスタ数

3. **パラメータグリッドサーチ**
   ```python
   min_cluster_sizes = [5, 10, 15, 20]
   metrics = ['euclidean', 'cosine']
   
   for size in min_cluster_sizes:
       for metric in metrics:
           test_configuration(size, metric)
   ```

## 🐛 デバッグ・トラブルシューティング

### よくある問題と解決法

#### 問題1: すべてがノイズになる
```python
# 原因: min_cluster_sizeが大きすぎる
# 解決: 値を減らす
min_cluster_size = 5  # より小さく
```

#### 問題2: クラスタが1つだけ
```python
# 原因: min_cluster_sizeが小さすぎる or データが均質
# 解決: 値を増やす or metricを変更
min_cluster_size = 20
metric = 'cosine'  # euclideanから変更
```

#### 問題3: 実行が遅い
```python
# 原因: condensed tree計算のオーバーヘッド
# 解決: データを減らす or max_depthを下げる
max_depth = 1
chunk_size = 2000  # より大きく
```

## 💻 コード構造

### クラス設計

```
RAPTORRetrieverHDBSCAN
├── __init__()               # パラメータ初期化
├── cluster_documents_hdbscan()  # HDBSCANクラスタリング
├── build_tree()             # 再帰的ツリー構築
├── search_tree()            # ツリー検索
├── index()                  # インデックス化
└── retrieve()               # クエリ実行
```

### データフロー

```
ドキュメント
    ↓ load_and_split_documents()
チャンク
    ↓ embed_documents()
埋め込みベクトル
    ↓ cluster_documents_hdbscan()
クラスタラベル (-1=ノイズ)
    ↓ build_tree() (再帰)
ツリー構造
    ↓ search_tree()
検索結果
```

## 🔮 今後の拡張アイデア

### 1. Condensed Treeの活用

```python
# condensed treeから最適な切断レベルを選択
tree = clusterer.condensed_tree_
persistence = tree.to_pandas()
optimal_clusters = select_by_stability(persistence)
```

### 2. Soft Clustering

```python
# 確率的メンバーシップの活用
soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
# 複数クラスタに所属するチャンクの処理
```

### 3. 動的パラメータ調整

```python
# 深さに応じてmin_cluster_sizeを変更
def adaptive_min_cluster_size(depth, base_size=15):
    return max(5, base_size - depth * 5)
```

### 4. ノイズチャンクの再利用

```python
# ノイズを別途保存し、特殊なクエリで活用
noise_chunks = [doc for i, doc in enumerate(documents) 
                if labels[i] == -1]
self.noise_index = create_separate_index(noise_chunks)
```

## 📚 理論的背景

### Mutual Reachability Distance

HDBSCANの核心:

```
d_mreach(a, b) = max(core_distance(a), core_distance(b), d(a, b))
```

- `core_distance(a)`: 点aの k-nearest neighbor距離
- `d(a, b)`: 点a, b間の元の距離
- これによりノイズに頑健な距離を定義

### Excess of Mass (EoM)

クラスタ選択基準:

```
EoM(C) = ∫ (λ - λ_min) dλ
```

- クラスタの「安定性」を測定
- より長く存在するクラスタを優先

## ✅ 実装チェックリスト

- [x] HDBSCANクラスタリングの実装
- [x] ノイズ検出・除外機能
- [x] 統計情報の追跡
- [x] 距離メトリック選択（euclidean/cosine）
- [x] サンプル数不足への対応
- [x] 再帰終了条件の調整
- [x] 比較実験スクリプト
- [x] パラメータチューニング機能
- [x] ドキュメント作成
- [ ] Condensed tree活用（将来的）
- [ ] Soft clustering（将来的）
- [ ] 動的パラメータ調整（将来的）

## 🎓 参考実装

### K-means版との対比

```python
# K-means版
def cluster_documents(embeddings, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters)
    return kmeans.fit_predict(embeddings)

# HDBSCAN版
def cluster_documents_hdbscan(embeddings):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=15,
        metric='cosine'
    )
    labels = clusterer.fit_predict(embeddings)
    # ノイズ処理を追加
    n_noise = list(labels).count(-1)
    return labels, {'n_noise': n_noise}
```

---

**実装者:** Takato Yasuno  
**実装日:** 2025年10月16日  
**バージョン:** 1.0
