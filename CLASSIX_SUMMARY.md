# CLASSIX実験サマリー

**日付**: 2025年10月16日  
**データ**: test.txt (864 chunks, 624KB)  
**GPU**: NVIDIA GeForce RTX 4060 Ti 16GB

## 🏆 最適パラメータ

```python
radius = 1.0
minPts = 3
max_depth = 2
use_cosine = True
```

## 📊 実験結果

| radius | Build時間 | Query時間 | 類似度 | クラスター数 |
|--------|-----------|-----------|--------|-------------|
| 0.5 | 99秒 | 18秒 | 0.7055 | D0:3, D1:3 |
| **1.0** | **77秒 ✅** | **24秒** | **0.7131 ✅** | **D0:2, D1:2** |
| 0.7 | 548秒 ❌ | 3秒 | 0.6941 | D0:14, D1:58 |

## 💡 重要な教訓

### ✅ 推奨: radius=1.0
- 最速ビルド (77秒)
- 最高精度 (0.7131)
- 効率的な階層構造

### ❌ 避ける: radius=0.7
- 7倍遅い (548秒)
- 過剰なクラスター (58個)
- 精度低下 (0.6941)

## 🚀 GPU加速効果

- CPU実行: 推定10時間
- GPU実行: 77秒
- **高速化率: 480倍！**

## 📁 関連ファイル

- `CLASSIX_EXPERIMENT_RESULTS.md` - 詳細レポート
- `CLASSIX_BEST_PRACTICES.md` - ベストプラクティスガイド
- `classix_config.py` - 本番用設定
- `raptor_classix.py` - 実装
- `example_classix_large.py` - テストコード
