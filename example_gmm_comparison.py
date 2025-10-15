"""
RAPTOR GMM vs K-means Comparison Demo
GMMとBICによる最適クラスター数選択のデモンストレーション
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from raptor_gmm import RAPTORRetrieverGMM
from raptor import RAPTORRetriever
import time


def compare_clustering_methods():
    """K-meansとGMM+BICの比較"""
    
    print("=" * 80)
    print("🔬 RAPTOR: Clustering Method Comparison")
    print("   K-means vs GMM with BIC-based optimal cluster selection")
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
    # Method 3: K-means with BIC（K-means + BIC最適化）
    # ==========================================
    print("📊 Method 3: K-means with BIC (K-means + Auto-optimal)")
    print("-" * 80)
    
    raptor_kmeans_bic = RAPTORRetrieverGMM(
        embeddings_model=embeddings_model,
        llm=llm,
        max_clusters=5,
        min_clusters=2,
        max_depth=2,
        chunk_size=1000,
        chunk_overlap=200,
        use_bic=True,
        clustering_method="kmeans"  # K-meansを使用（BIC最適化あり）
    )
    
    start_time = time.time()
    raptor_kmeans_bic.index("test.txt")
    kmeans_bic_build_time = time.time() - start_time
    
    start_time = time.time()
    results_kmeans_bic = raptor_kmeans_bic.retrieve(test_query, top_k=3)
    kmeans_bic_query_time = time.time() - start_time
    
    print(f"\n✅ K-means+BIC Results:")
    print(f"   Build time: {kmeans_bic_build_time:.2f}秒")
    print(f"   Query time: {kmeans_bic_query_time:.3f}秒")
    print(f"\n   Top 3 Results:")
    for i, doc in enumerate(results_kmeans_bic, 1):
        similarity = doc.metadata.get('similarity', 'N/A')
        print(f"   {i}. Similarity: {similarity}")
        print(f"      Preview: {doc.page_content[:150]}...")
    
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
    print(f"| K-means + BIC        | {kmeans_bic_build_time:>7.2f}秒  | {kmeans_bic_query_time:>7.3f}秒 | Auto-optimal, traditional   |")
    
    print("\n🎯 Key Findings:")
    print("1. BIC allows automatic selection of optimal cluster count")
    print("2. GMM provides more flexible clustering (elliptical clusters)")
    print("3. Build time may increase slightly due to BIC search")
    print("4. Query time remains consistent across methods")
    
    print("\n💡 Recommendations:")
    print("• Use GMM+BIC for complex, heterogeneous documents")
    print("• Use K-means+BIC for large-scale with auto-optimization")
    print("• Use K-means (fixed) when cluster structure is known")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    compare_clustering_methods()
