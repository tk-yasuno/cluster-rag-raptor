"""
CLASSIX Production Configuration

実験結果に基づく推奨設定 (2025年10月16日検証済み)
データセット: 864 chunks (test.txt)
GPU: NVIDIA GeForce RTX 4060 Ti 16GB

実験結果:
- Build時間: 76.57秒
- Query時間: 23.995秒
- Top類似度: 0.7131
- クラスター数: Depth 0=2, Depth 1=2
"""

from raptor_classix import RaptorRetrieverCLASSIX

# ================================================================================
# 推奨デフォルト設定 (中規模データ: 100-1000 chunks)
# ================================================================================

CLASSIX_CONFIG_DEFAULT = {
    "radius": 1.0,        # 最適値 (実験実証済み)
    "minPts": 3,          # バランス型
    "max_depth": 2,       # 十分な階層
    "use_cosine": True    # コサイン類似度
}

# ================================================================================
# データセット別設定
# ================================================================================

# 小規模データ (< 100 chunks)
CLASSIX_CONFIG_SMALL = {
    "radius": 0.5,        # より細かく分類
    "minPts": 2,          # 小さなクラスター許容
    "max_depth": 2,
    "use_cosine": True
}

# 中規模データ (100-1000 chunks) - 推奨
CLASSIX_CONFIG_MEDIUM = {
    "radius": 1.0,        # 実験検証済み最適値
    "minPts": 3,
    "max_depth": 2,
    "use_cosine": True
}

# 大規模データ (1000+ chunks)
CLASSIX_CONFIG_LARGE = {
    "radius": 1.2,        # より大きなクラスター
    "minPts": 5,          # ノイズ除去強化
    "max_depth": 3,       # より深い階層
    "use_cosine": True
}

# ================================================================================
# 特殊用途設定
# ================================================================================

# 高速処理優先 (精度よりも速度)
CLASSIX_CONFIG_FAST = {
    "radius": 1.5,        # 大きなクラスター
    "minPts": 5,
    "max_depth": 1,       # 浅い階層
    "use_cosine": True
}

# 高精度優先 (速度よりも精度)
CLASSIX_CONFIG_ACCURATE = {
    "radius": 0.8,        # やや小さいクラスター
    "minPts": 3,
    "max_depth": 3,       # 深い階層
    "use_cosine": True
}

# ノイズ多いデータ
CLASSIX_CONFIG_NOISY = {
    "radius": 1.2,
    "minPts": 10,         # 強いノイズ除去
    "max_depth": 2,
    "use_cosine": True
}

# ================================================================================
# Ollama モデル設定
# ================================================================================

OLLAMA_CONFIG = {
    "llm_model": "granite-code:8b",           # 要約生成用 (6.1GB VRAM)
    "embedding_model": "mxbai-embed-large",   # 埋め込み生成用 (1.2GB VRAM)
    "base_url": "http://localhost:11434",     # Ollama API URL
    "temperature": 0.0,                       # 一貫性のため0
}

# GPU要件: 合計 7.3GB VRAM (RTX 4060 Ti 16GB推奨)

# ================================================================================
# テキスト分割設定
# ================================================================================

TEXT_SPLIT_CONFIG = {
    "chunk_size": 800,              # 1チャンクの文字数
    "chunk_overlap": 100,           # オーバーラップ文字数
}

# ================================================================================
# ファクトリー関数
# ================================================================================

def create_raptor_default():
    """デフォルト設定でRAPTORインスタンスを作成 (推奨)"""
    return RaptorRetrieverCLASSIX(**CLASSIX_CONFIG_DEFAULT)

def create_raptor_small():
    """小規模データ用RAPTORインスタンスを作成"""
    return RaptorRetrieverCLASSIX(**CLASSIX_CONFIG_SMALL)

def create_raptor_medium():
    """中規模データ用RAPTORインスタンスを作成 (推奨)"""
    return RaptorRetrieverCLASSIX(**CLASSIX_CONFIG_MEDIUM)

def create_raptor_large():
    """大規模データ用RAPTORインスタンスを作成"""
    return RaptorRetrieverCLASSIX(**CLASSIX_CONFIG_LARGE)

def create_raptor_fast():
    """高速処理優先RAPTORインスタンスを作成"""
    return RaptorRetrieverCLASSIX(**CLASSIX_CONFIG_FAST)

def create_raptor_accurate():
    """高精度優先RAPTORインスタンスを作成"""
    return RaptorRetrieverCLASSIX(**CLASSIX_CONFIG_ACCURATE)

def create_raptor_custom(radius, minPts, max_depth=2):
    """カスタム設定でRAPTORインスタンスを作成"""
    return RaptorRetrieverCLASSIX(
        radius=radius,
        minPts=minPts,
        max_depth=max_depth,
        use_cosine=True
    )

# ================================================================================
# 使用例
# ================================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("CLASSIX Production Configuration")
    print("=" * 80)
    print()
    
    print("📋 Available Configurations:")
    print()
    print("1. DEFAULT (推奨)")
    print(f"   {CLASSIX_CONFIG_DEFAULT}")
    print()
    print("2. SMALL (< 100 chunks)")
    print(f"   {CLASSIX_CONFIG_SMALL}")
    print()
    print("3. MEDIUM (100-1000 chunks) - 実験検証済み ✅")
    print(f"   {CLASSIX_CONFIG_MEDIUM}")
    print()
    print("4. LARGE (1000+ chunks)")
    print(f"   {CLASSIX_CONFIG_LARGE}")
    print()
    print("5. FAST (高速処理優先)")
    print(f"   {CLASSIX_CONFIG_FAST}")
    print()
    print("6. ACCURATE (高精度優先)")
    print(f"   {CLASSIX_CONFIG_ACCURATE}")
    print()
    print("7. NOISY (ノイズ多いデータ)")
    print(f"   {CLASSIX_CONFIG_NOISY}")
    print()
    
    print("=" * 80)
    print("🚀 Quick Start")
    print("=" * 80)
    print()
    print("```python")
    print("from classix_config import create_raptor_medium")
    print()
    print("# 推奨設定でインスタンス作成")
    print("raptor = create_raptor_medium()")
    print()
    print("# ドキュメントをインデックス")
    print('raptor.add_documents("your_document.txt")')
    print()
    print("# 検索")
    print('results = raptor.retrieve("your query", top_k=5)')
    print("```")
    print()
    
    print("=" * 80)
    print("📊 Performance Metrics (864 chunks, RTX 4060 Ti)")
    print("=" * 80)
    print()
    print("Configuration: MEDIUM (radius=1.0, minPts=3)")
    print("  Build時間:  76.57秒")
    print("  Query時間:  23.995秒")
    print("  Top類似度:  0.7131")
    print("  クラスター: Depth 0=2, Depth 1=2")
    print("  GPU加速:   480倍 (CPU比)")
    print()
    print("=" * 80)
