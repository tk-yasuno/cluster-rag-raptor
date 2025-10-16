# HDBSCAN版RAPTOR実装

このフォルダーには、HDBSCAN (Hierarchical Density-Based Spatial Clustering of Applications with Noise) を使用したRAPTOR実装が含まれています。

## ⚠️ 注意事項

**HDBSCANはパラメータ調整が困難**なため、実用的には以下の手法を推奨します：

1. **GMM + BIC** (`../raptor_gmm.py`) - 推奨 ⭐⭐⭐⭐⭐
2. **K-means** (`../raptor.py`) - シンプル ⭐⭐⭐⭐
3. **HDBSCAN** (このフォルダー) - 実験的 ⭐⭐⭐

## 📁 ファイル構成

### 実装ファイル
- **`raptor_hdbscan.py`** - HDBSCAN版RAPTOR本体（CPU版）
- **`example_hdbscan_comparison.py`** - 4手法比較実験
- **`example_hdbscan_quick.py`** - クイックテスト

### ドキュメント
- **`HDBSCAN_README.md`** - 統合README
- **`HDBSCAN_GUIDE.md`** - 完全ガイド
- **`HDBSCAN_QUICKSTART.md`** - クイックスタート
- **`HDBSCAN_IMPLEMENTATION_NOTES.md`** - 技術ノート

## 🚀 クイックスタート

```bash
cd hdbscan
python example_hdbscan_quick.py
```

## 🎯 主な課題

### パラメータ調整の難しさ

```python
# 問題1: データが少ないとクラスタができない
min_cluster_size=15  # 4チャンクでは全てノイズに

# 問題2: 適切な値がデータサイズに依存
- 小データ: min_cluster_size=3-5
- 中データ: min_cluster_size=10-15  
- 大データ: min_cluster_size=20-30

# 問題3: コサイン距離が直接サポートされていない
# → 正規化 + Euclidean で対応済み
```

## 💡 使用する場合

大規模データセット（1000チャンク以上）で試す価値があります：

```python
from raptor_hdbscan import RAPTORRetrieverHDBSCAN

raptor = RAPTORRetrieverHDBSCAN(
    embeddings_model=embeddings,
    llm=llm,
    min_cluster_size=20,    # データサイズに応じて調整
    min_samples=7,          # min_cluster_size/3が目安
    metric='cosine',        # 正規化後Euclideanで計算
    exclude_noise=True
)
```

## 📊 実装の成果

✅ **完成した機能:**
- コサイン距離対応（正規化+Euclidean）
- ノイズ検出・除去機能
- 統計情報の追跡
- 包括的なドキュメント

⚠️ **実用上の制約:**
- パラメータチューニングが困難
- 小データでクラスタが形成されにくい
- GMM+BICの方が安定

## 🔗 推奨代替案

プロジェクトルートの以下ファイルを使用することを推奨します：

```bash
cd ..
python example_gmm_comparison.py  # GMM+BIC推奨
```

---

**作成日:** 2025年10月16日  
**ステータス:** 実験的実装・アーカイブ済み  
**推奨:** GMM+BIC版を使用
