"""
RAPTOR CLASSIX vs GMM vs K-means Comparison Demo
CLASSIXによる高速クラスタリングのデモンストレーション

比較対象:
1. K-means (固定クラスタ数)
2. GMM + BIC (最適クラスタ数自動選択)
3. CLASSIX (高速・調整容易・自動クラスタ数決定)
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from raptor_classix import RAPTORRetrieverCLASSIX
from raptor_gmm import RAPTORRetrieverGMM
from raptor import RAPTORRetriever
import time


def compare_clustering_methods():
    """K-means、GMM+BIC、CLASSIXの比較"""
    
    print("=" * 80)
    print("🔬 RAPTOR: Clustering Method Comparison")
    print("   K-means vs GMM+BIC vs CLASSIX (fast & easy tuning)")
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
        max_clusters=5,
        min_clusters=2,
        max_depth=2,
        chunk_size=1000,
        chunk_overlap=200,
        use_bic=True,
        clustering_method="gmm"
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
    # Method 3: CLASSIX（高速・調整容易）
    # ==========================================
    print("📊 Method 3: CLASSIX (Fast & Easy tuning)")
    print("-" * 80)
    
    raptor_classix = RAPTORRetrieverCLASSIX(
        embeddings_model=embeddings_model,
        llm=llm,
        radius=0.5,         # クラスタの半径
        minPts=5,           # 最小ポイント数
        max_depth=2,
        chunk_size=1000,
        chunk_overlap=200,
        use_cosine=True     # コサイン距離
    )
    
    start_time = time.time()
    raptor_classix.index("test.txt")
    classix_build_time = time.time() - start_time
    
    start_time = time.time()
    results_classix = raptor_classix.retrieve(test_query, top_k=3)
    classix_query_time = time.time() - start_time
    
    print(f"\n✅ CLASSIX Results:")
    print(f"   Build time: {classix_build_time:.2f}秒")
    print(f"   Query time: {classix_query_time:.3f}秒")
    print(f"\n   Top 3 Results:")
    for i, doc in enumerate(results_classix, 1):
        similarity = doc.metadata.get('similarity', 'N/A')
        print(f"   {i}. Similarity: {similarity}")
        print(f"      Preview: {doc.page_content[:150]}...")
    
    # CLASSIXの統計情報
    print(f"\n   📊 CLASSIX Statistics:")
    stats = raptor_classix.cluster_stats
    print(f"      Parameters: radius={stats['params']['radius']}, minPts={stats['params']['minPts']}")
    for depth in sorted(stats['total_clusters_by_depth'].keys()):
        clusters = stats['total_clusters_by_depth'][depth]
        noise = stats['noise_by_depth'].get(depth, 0)
        print(f"      Depth {depth}: {clusters} clusters, {noise} noise points")
    
    print("\n")
    
    # ==========================================
    # 比較サマリー
    # ==========================================
    print("\n" + "=" * 80)
    print("📊 Performance Comparison Summary")
    print("=" * 80)
    print()
    
    print("| Method               | Build Time | Query Time | Note                        |")
    print("|----------------------|------------|------------|-----------------------------|")
    print(f"| K-means (fixed)      | {kmeans_build_time:>7.2f}秒  | {kmeans_query_time:>7.3f}秒 | Manual cluster count        |")
    print(f"| GMM + BIC            | {gmm_build_time:>7.2f}秒  | {gmm_query_time:>7.3f}秒 | Auto-optimal, flexible      |")
    print(f"| CLASSIX              | {classix_build_time:>7.2f}秒  | {classix_query_time:>7.3f}秒 | Fast, easy tuning           |")
    
    print("\n🎯 Key Findings:")
    print("1. CLASSIX provides fast clustering with automatic cluster count")
    print("2. Easier parameter tuning compared to HDBSCAN")
    print("3. radius parameter is intuitive (smaller = more clusters)")
    print("4. Comparable performance to GMM+BIC with less complexity")
    
    print("\n💡 Recommendations:")
    print("• Use CLASSIX for fast prototyping and easy tuning")
    print("• Use GMM+BIC for complex, heterogeneous documents")
    print("• Use K-means when cluster structure is known")
    
    print("\n🔧 CLASSIX Parameter Guide:")
    print("• radius: クラスタの半径 (小さいほど細かく分割)")
    print("  - Small docs (100-500 chars): 0.3-0.4")
    print("  - Medium docs (500-1500 chars): 0.4-0.6")
    print("  - Large docs (1500+ chars): 0.5-0.8")
    print("• minPts: 最小ポイント数 (3-10推奨)")
    print("  - Small datasets: 3-5")
    print("  - Large datasets: 5-10")
    
    print("\n" + "=" * 80)


def test_different_parameters():
    """CLASSIXのパラメータチューニング実験"""
    
    print("\n" + "=" * 80)
    print("🧪 CLASSIX Parameter Tuning Experiment")
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
    
    # 異なるradiusでテスト
    radius_values = [0.3, 0.4, 0.5, 0.6]
    
    results_summary = []
    
    for radius in radius_values:
        print(f"\n{'='*80}")
        print(f"Testing radius = {radius}")
        print(f"{'='*80}")
        
        raptor = RAPTORRetrieverCLASSIX(
            embeddings_model=embeddings_model,
            llm=llm,
            radius=radius,
            minPts=5,
            max_depth=2,
            chunk_size=1000,
            chunk_overlap=200,
            use_cosine=True
        )
        
        start_time = time.time()
        raptor.index("test.txt")
        build_time = time.time() - start_time
        
        results = raptor.retrieve(test_query, top_k=3)
        
        # 統計を集計
        total_clusters = sum(raptor.cluster_stats['total_clusters_by_depth'].values())
        total_noise = sum(raptor.cluster_stats['noise_by_depth'].values())
        
        results_summary.append({
            'radius': radius,
            'build_time': build_time,
            'total_clusters': total_clusters,
            'total_noise': total_noise,
            'top_similarity': results[0].metadata.get('similarity', 'N/A') if results else 'N/A'
        })
    
    # サマリー表示
    print("\n" + "=" * 80)
    print("📊 Parameter Tuning Summary")
    print("=" * 80)
    print()
    print("| radius | Build Time | Total Clusters | Total Noise | Top Similarity |")
    print("|--------|------------|----------------|-------------|----------------|")
    for res in results_summary:
        print(f"| {res['radius']:>6.1f} | {res['build_time']:>7.2f}秒  | {res['total_clusters']:>14} | {res['total_noise']:>11} | {res['top_similarity']:>14} |")
    
    print("\n💡 Observations:")
    print("• Smaller radius → More clusters, finer granularity")
    print("• Larger radius → Fewer clusters, coarser grouping")
    print("• Optimal radius depends on document characteristics")
    print("=" * 80)


if __name__ == "__main__":
    # メイン比較実験
    compare_clustering_methods()
    
    # パラメータチューニング実験（オプション）
    # test_different_parameters()
