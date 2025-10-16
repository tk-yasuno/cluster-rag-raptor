"""
HDBSCAN単独テスト - 高速版
小さいデータセットで動作確認
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from raptor_hdbscan import RAPTORRetrieverHDBSCAN
import time


def quick_test_hdbscan():
    """HDBSCANの単独テスト（高速版）"""
    
    print("=" * 80)
    print("🚀 HDBSCAN RAPTOR - Quick Test")
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
    
    # HDBSCAN（Cosine距離）
    print("📊 Testing HDBSCAN with Cosine metric")
    print("-" * 80)
    
    raptor_hdbscan = RAPTORRetrieverHDBSCAN(
        embeddings_model=embeddings_model,
        llm=llm,
        min_cluster_size=10,        # 864チャンク用に調整
        min_samples=3,              # 適度に緩く
        max_depth=2,                # 深さ2に制限
        chunk_size=1000,
        chunk_overlap=200,
        metric='cosine',            # コサイン距離（正規化後にEuclideanで計算）
        exclude_noise=True
    )
    
    start_time = time.time()
    
    # 大きいファイルでテスト（864チャンク）
    print("\n📖 Note: Using full test.txt file (864 chunks)")
    print("   min_cluster_size=10, min_samples=3")
    raptor_hdbscan.index("test.txt")
    
    build_time = time.time() - start_time
    
    start_time = time.time()
    results = raptor_hdbscan.retrieve(test_query, top_k=3)
    query_time = time.time() - start_time
    
    # 結果表示
    print(f"\n{'='*80}")
    print(f"✅ HDBSCAN Test Results")
    print(f"{'='*80}")
    print(f"Build time: {build_time:.2f}秒")
    print(f"Query time: {query_time:.3f}秒")
    
    print(f"\n📊 Noise Statistics:")
    print(f"   Total noise chunks removed: {raptor_hdbscan.noise_stats['total_noise_chunks']}")
    if raptor_hdbscan.noise_stats['noise_by_depth']:
        print(f"   Noise by depth: {dict(raptor_hdbscan.noise_stats['noise_by_depth'])}")
    
    print(f"\nTop 3 Results:")
    for i, doc in enumerate(results, 1):
        similarity = doc.metadata.get('similarity', 'N/A')
        print(f"\n{i}. Similarity: {similarity}")
        print(f"   Preview: {doc.page_content[:200]}...")
    
    print("\n" + "=" * 80)
    print("🎯 Quick Test Complete!")
    print("   - HDBSCAN automatically found optimal cluster count")
    print(f"   - Removed {raptor_hdbscan.noise_stats['total_noise_chunks']} noisy chunks")
    print("   - Used cosine distance for semantic similarity")
    print("=" * 80)


if __name__ == "__main__":
    quick_test_hdbscan()
