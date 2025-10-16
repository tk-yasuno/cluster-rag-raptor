"""
RAPTOR HDBSCAN vs GMM vs K-means Comparison Demo
HDBSCANによるノイズ除去とクラスター数自動決定のデモンストレーション

比較対象:
1. K-means (固定クラスター数)
2. GMM + BIC (最適クラスター数自動選択)
3. HDBSCAN (クラスター数自動決定 + ノイズ除去)
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from raptor_hdbscan import RAPTORRetrieverHDBSCAN
from raptor_gmm import RAPTORRetrieverGMM
from raptor import RAPTORRetriever
import time


def compare_clustering_methods():
    """K-means、GMM+BIC、HDBSCANの比較"""
    
    print("=" * 80)
    print("🔬 RAPTOR: Advanced Clustering Method Comparison")
    print("   K-means vs GMM+BIC vs HDBSCAN (with noise removal)")
    print("=" * 80)
    print()
    
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
    
    # テストクエリ
    test_query = "philosophy"
    
    # ==========================================
    # Method 1: 従来のK-means（固定クラスター数）
    # ==========================================
    print("📊 Method 1: K-means (Fixed cluster count = 3)")
    print("-" * 80)
    
    raptor_kmeans = RAPTORRetriever(
        embeddings_model=embeddings_model,
        llm=llm,
        max_clusters=3,
        max_depth=2,
        chunk_size=1000,
        chunk_overlap=200
    )
    
    start_time = time.time()
    raptor_kmeans.index("test.txt")
    kmeans_build_time = time.time() - start_time
    
    start_time = time.time()
    results_kmeans = raptor_kmeans.retrieve(test_query, top_k=3)
    kmeans_query_time = time.time() - start_time
    
    print(f"\n✅ K-means Results:")
    print(f"   Build time: {kmeans_build_time:.2f}秒")
    print(f"   Query time: {kmeans_query_time:.3f}秒")
    print(f"\n   Top 3 Results:")
    for i, doc in enumerate(results_kmeans, 1):
        similarity = doc.metadata.get('similarity', 'N/A')
        print(f"   {i}. Similarity: {similarity}")
        print(f"      Preview: {doc.page_content[:150]}...")
    
    print("\n")
    
    # ==========================================
    # Method 2: GMM with BIC（最適クラスター数自動選択）
    # ==========================================
    print("📊 Method 2: GMM with BIC (Auto-optimal cluster count)")
    print("-" * 80)
    
    raptor_gmm = RAPTORRetrieverGMM(
        embeddings_model=embeddings_model,
        llm=llm,
        max_clusters=5,      # BICで探索する最大クラスター数
        min_clusters=2,      # BICで探索する最小クラスター数
        max_depth=2,
        chunk_size=1000,
        chunk_overlap=200,
        use_bic=True,        # BICによる最適化ON
        clustering_method="gmm"  # GMMを使用
    )
    
    start_time = time.time()
    raptor_gmm.index("test.txt")
    gmm_build_time = time.time() - start_time
    
    start_time = time.time()
    results_gmm = raptor_gmm.retrieve(test_query, top_k=3)
    gmm_query_time = time.time() - start_time
    
    print(f"\n✅ GMM+BIC Results:")
    print(f"   Build time: {gmm_build_time:.2f}秒")
    print(f"   Query time: {gmm_query_time:.3f}秒")
    print(f"\n   Top 3 Results:")
    for i, doc in enumerate(results_gmm, 1):
        similarity = doc.metadata.get('similarity', 'N/A')
        print(f"   {i}. Similarity: {similarity}")
        print(f"      Preview: {doc.page_content[:150]}...")
    
    print("\n")
    
    # ==========================================
    # Method 3: HDBSCAN（クラスター数自動決定 + ノイズ除去）
    # ==========================================
    print("📊 Method 3: HDBSCAN (Auto cluster count + Noise removal)")
    print("-" * 80)
    
    raptor_hdbscan = RAPTORRetrieverHDBSCAN(
        embeddings_model=embeddings_model,
        llm=llm,
        min_cluster_size=15,    # クラスタの最小サイズ
        min_samples=5,          # 密度推定の最小サンプル数
        max_depth=2,
        chunk_size=1000,
        chunk_overlap=200,
        metric='euclidean',     # 距離メトリック
        exclude_noise=True      # ノイズ除去ON
    )
    
    start_time = time.time()
    raptor_hdbscan.index("test.txt")
    hdbscan_build_time = time.time() - start_time
    
    start_time = time.time()
    results_hdbscan = raptor_hdbscan.retrieve(test_query, top_k=3)
    hdbscan_query_time = time.time() - start_time
    
    print(f"\n✅ HDBSCAN Results:")
    print(f"   Build time: {hdbscan_build_time:.2f}秒")
    print(f"   Query time: {hdbscan_query_time:.3f}秒")
    print(f"\n   Top 3 Results:")
    for i, doc in enumerate(results_hdbscan, 1):
        similarity = doc.metadata.get('similarity', 'N/A')
        print(f"   {i}. Similarity: {similarity}")
        print(f"      Preview: {doc.page_content[:150]}...")
    
    # ノイズ統計を表示
    print(f"\n   📊 Noise Statistics:")
    print(f"      Total noise chunks removed: {raptor_hdbscan.noise_stats['total_noise_chunks']}")
    if raptor_hdbscan.noise_stats['noise_by_depth']:
        print(f"      Noise by depth: {dict(raptor_hdbscan.noise_stats['noise_by_depth'])}")
    
    print("\n")
    
    # ==========================================
    # Method 4: HDBSCAN with cosine metric
    # ==========================================
    print("📊 Method 4: HDBSCAN (Cosine metric)")
    print("-" * 80)
    
    raptor_hdbscan_cosine = RAPTORRetrieverHDBSCAN(
        embeddings_model=embeddings_model,
        llm=llm,
        min_cluster_size=15,
        min_samples=5,
        max_depth=2,
        chunk_size=1000,
        chunk_overlap=200,
        metric='cosine',        # コサイン距離を使用
        exclude_noise=True
    )
    
    start_time = time.time()
    raptor_hdbscan_cosine.index("test.txt")
    hdbscan_cosine_build_time = time.time() - start_time
    
    start_time = time.time()
    results_hdbscan_cosine = raptor_hdbscan_cosine.retrieve(test_query, top_k=3)
    hdbscan_cosine_query_time = time.time() - start_time
    
    print(f"\n✅ HDBSCAN (Cosine) Results:")
    print(f"   Build time: {hdbscan_cosine_build_time:.2f}秒")
    print(f"   Query time: {hdbscan_cosine_query_time:.3f}秒")
    print(f"\n   Top 3 Results:")
    for i, doc in enumerate(results_hdbscan_cosine, 1):
        similarity = doc.metadata.get('similarity', 'N/A')
        print(f"   {i}. Similarity: {similarity}")
        print(f"      Preview: {doc.page_content[:150]}...")
    
    # ノイズ統計を表示
    print(f"\n   📊 Noise Statistics:")
    print(f"      Total noise chunks removed: {raptor_hdbscan_cosine.noise_stats['total_noise_chunks']}")
    if raptor_hdbscan_cosine.noise_stats['noise_by_depth']:
        print(f"      Noise by depth: {dict(raptor_hdbscan_cosine.noise_stats['noise_by_depth'])}")
    
    # ==========================================
    # 比較サマリー
    # ==========================================
    print("\n" + "=" * 80)
    print("📊 Performance Comparison Summary")
    print("=" * 80)
    print()
    
    print("| Method                  | Build Time | Query Time | Note                           |")
    print("|-------------------------|------------|------------|--------------------------------|")
    print(f"| K-means (fixed)         | {kmeans_build_time:>7.2f}秒  | {kmeans_query_time:>7.3f}秒 | Manual cluster count           |")
    print(f"| GMM + BIC               | {gmm_build_time:>7.2f}秒  | {gmm_query_time:>7.3f}秒 | Auto-optimal, flexible         |")
    print(f"| HDBSCAN (euclidean)     | {hdbscan_build_time:>7.2f}秒  | {hdbscan_query_time:>7.3f}秒 | Auto + noise removal           |")
    print(f"| HDBSCAN (cosine)        | {hdbscan_cosine_build_time:>7.2f}秒  | {hdbscan_cosine_query_time:>7.3f}秒 | Semantic distance based        |")
    
    print("\n🎯 Key Findings:")
    print("1. HDBSCAN automatically determines optimal cluster count")
    print("2. Noise removal improves semantic quality by excluding weak chunks")
    print("3. Cosine metric may better capture semantic similarity")
    print("4. Build time increases slightly but search quality improves")
    print(f"5. HDBSCAN removed {raptor_hdbscan.noise_stats['total_noise_chunks']} noise chunks (euclidean)")
    print(f"6. HDBSCAN removed {raptor_hdbscan_cosine.noise_stats['total_noise_chunks']} noise chunks (cosine)")
    
    print("\n💡 Recommendations:")
    print("• Use HDBSCAN for automatic clustering with noise detection")
    print("• Use cosine metric for semantic embeddings (e.g., mxbai-embed-large)")
    print("• Tune min_cluster_size based on document granularity")
    print("• Lower min_cluster_size for finer-grained clusters")
    print("• Higher min_cluster_size for more conservative clustering")
    
    print("\n🔧 HDBSCAN Parameter Guide:")
    print("• min_cluster_size: 最小クラスタサイズ (大きいほど保守的)")
    print("  - Small docs (100-500 chars): 5-10")
    print("  - Medium docs (500-1500 chars): 10-20")
    print("  - Large docs (1500+ chars): 15-30")
    print("• min_samples: 密度推定のサンプル数 (通常はmin_cluster_sizeの1/3)")
    print("• metric: 'euclidean' (汎用) or 'cosine' (semantic)")
    
    print("\n" + "=" * 80)


def test_different_parameters():
    """HDBSCANのパラメータチューニング実験"""
    
    print("\n" + "=" * 80)
    print("🧪 HDBSCAN Parameter Tuning Experiment")
    print("=" * 80)
    print()
    
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
    
    test_query = "philosophy"
    
    # 異なるmin_cluster_sizeでテスト
    cluster_sizes = [5, 10, 15, 20]
    
    results_summary = []
    
    for min_size in cluster_sizes:
        print(f"\n{'='*80}")
        print(f"Testing min_cluster_size = {min_size}")
        print(f"{'='*80}")
        
        raptor = RAPTORRetrieverHDBSCAN(
            embeddings_model=embeddings_model,
            llm=llm,
            min_cluster_size=min_size,
            min_samples=max(1, min_size // 3),  # 1/3ルール
            max_depth=2,
            chunk_size=1000,
            chunk_overlap=200,
            metric='cosine',
            exclude_noise=True
        )
        
        start_time = time.time()
        raptor.index("test.txt")
        build_time = time.time() - start_time
        
        results = raptor.retrieve(test_query, top_k=3)
        
        results_summary.append({
            'min_cluster_size': min_size,
            'build_time': build_time,
            'noise_removed': raptor.noise_stats['total_noise_chunks'],
            'top_similarity': results[0].metadata.get('similarity', 'N/A') if results else 'N/A'
        })
    
    # サマリー表示
    print("\n" + "=" * 80)
    print("📊 Parameter Tuning Summary")
    print("=" * 80)
    print()
    print("| min_cluster_size | Build Time | Noise Removed | Top Similarity |")
    print("|------------------|------------|---------------|----------------|")
    for res in results_summary:
        print(f"| {res['min_cluster_size']:>16} | {res['build_time']:>7.2f}秒  | {res['noise_removed']:>13} | {res['top_similarity']:>14} |")
    
    print("\n💡 Observations:")
    print("• Larger min_cluster_size → More conservative clustering")
    print("• Larger min_cluster_size → More noise detected")
    print("• Optimal value depends on document characteristics")
    print("=" * 80)


if __name__ == "__main__":
    # メイン比較実験
    compare_clustering_methods()
    
    # パラメータチューニング実験（オプション）
    # test_different_parameters()
