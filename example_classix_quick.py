"""
CLASSIX単独テスト - パラメータチューニング版
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from raptor_classix import RAPTORRetrieverCLASSIX
import time


def test_classix_params():
    """CLASSIXのパラメータを変えてテスト"""
    
    print("=" * 80)
    print("🔬 CLASSIX Parameter Tuning Test")
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
    test_configs = [
        {"radius": 0.3, "minPts": 2, "desc": "細かいクラスタ（小radius）"},
        {"radius": 0.8, "minPts": 2, "desc": "大きいクラスタ（大radius）"},
        {"radius": 1.5, "minPts": 2, "desc": "超大きいクラスタ（超大radius）"},
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
        raptor.index("test_small.txt")
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
        print(f"   Top similarity: {stats['top_similarity']}")
    
    # サマリー表示
    print(f"\n{'='*80}")
    print(f"📊 Parameter Tuning Summary")
    print(f"{'='*80}\n")
    
    print("| radius | minPts | Build Time | Top Similarity | Clusters @ Depth 0 |")
    print("|--------|--------|------------|----------------|--------------------|")
    for stat in results:
        cfg = stat['config']
        clusters_d0 = stat['cluster_stats']['total_clusters_by_depth'].get(0, 'N/A')
        print(f"| {cfg['radius']:>6} | {cfg['minPts']:>6} | {stat['build_time']:>7.2f}秒  | {stat['top_similarity']:>14} | {clusters_d0:>18} |")
    
    print(f"\n💡 Recommendations:")
    print("• radius が小さすぎると、全てが別々のクラスタになる")
    print("• radius が大きすぎると、全てが1つのクラスタになる")
    print("• 小データ(4-10チャンク): radius=0.8-1.5, minPts=2")
    print("• 中データ(100-500チャンク): radius=0.5-0.8, minPts=3-5")
    print("• 大データ(500+チャンク): radius=0.3-0.5, minPts=5-10")
    print(f"{'='*80}")


if __name__ == "__main__":
    test_classix_params()
