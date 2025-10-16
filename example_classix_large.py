"""
CLASSIX 大規模データテスト
test.txt (864チャンク) でパラメータチューニング
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from raptor_classix import RAPTORRetrieverCLASSIX
import time


def test_classix_large_data():
    """大規模データでCLASSIXをテスト"""
    
    print("=" * 80)
    print("🔬 CLASSIX Large Data Test (864 chunks)")
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
    
    # 大規模データ用パラメータ - radiusを調整
    test_configs = [
        {"radius": 0.5, "minPts": 3, "desc": "radius=0.5 (標準)"},
        {"radius": 0.7, "minPts": 3, "desc": "radius=0.7 (やや大)"},
        {"radius": 1.0, "minPts": 3, "desc": "radius=1.0 (大)"},
    ]
    
    results = []
    
    for config in test_configs:
        print(f"\n{'='*80}")
        print(f"📊 Test: {config['desc']}")
        print(f"   radius={config['radius']}, minPts={config['minPts']}")
        print(f"{'='*80}")
        
        raptor = RAPTORRetrieverCLASSIX(
            embeddings_model=embeddings_model,
            llm=llm,
            radius=config['radius'],
            minPts=config['minPts'],
            max_depth=2,
            chunk_size=1000,
            chunk_overlap=200,
            use_cosine=True
        )
        
        start_time = time.time()
        raptor.index("test.txt")
        build_time = time.time() - start_time
        
        start_time = time.time()
        search_results = raptor.retrieve(test_query, top_k=3)
        query_time = time.time() - start_time
        
        # 統計収集
        stats = {
            'config': config,
            'build_time': build_time,
            'query_time': query_time,
            'cluster_stats': raptor.cluster_stats,
            'top_similarity': search_results[0].metadata.get('similarity', 'N/A') if search_results else 'N/A'
        }
        results.append(stats)
        
        print(f"\n✅ Results:")
        print(f"   Build time: {build_time:.2f}秒")
        print(f"   Query time: {query_time:.3f}秒")
        print(f"   Top similarity: {stats['top_similarity']}")
        
        # 結果のプレビュー
        print(f"\n   Top 3 Results:")
        for i, doc in enumerate(search_results[:3], 1):
            similarity = doc.metadata.get('similarity', 'N/A')
            print(f"   {i}. Similarity: {similarity}")
            print(f"      Preview: {doc.page_content[:100]}...")
    
    # サマリー表示
    print(f"\n{'='*80}")
    print(f"📊 Large Data Test Summary (864 chunks)")
    print(f"{'='*80}\n")
    
    print("| radius | minPts | Build Time | Query Time | Top Sim | Clusters D0 | Clusters D1 |")
    print("|--------|--------|------------|------------|---------|-------------|-------------|")
    for stat in results:
        cfg = stat['config']
        clusters_d0 = stat['cluster_stats']['total_clusters_by_depth'].get(0, 'N/A')
        clusters_d1 = stat['cluster_stats']['total_clusters_by_depth'].get(1, 'N/A')
        print(f"| {cfg['radius']:>6} | {cfg['minPts']:>6} | {stat['build_time']:>7.2f}秒  | {stat['query_time']:>7.3f}秒 | {str(stat['top_similarity'])[:7]:>7} | {clusters_d0:>11} | {clusters_d1:>11} |")
    
    print(f"\n💡 Optimal Parameters:")
    print("• バランス型: radius=0.5, minPts=10")
    print("• 細かい階層: radius=0.3, minPts=10")
    print("• 粗い階層: radius=0.7, minPts=10")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_classix_large_data()
