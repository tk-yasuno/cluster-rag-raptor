"""
GMM+BIC vs CLASSIX 比較実験

目的:
- GMM+BIC と CLASSIX の性能を公平に比較
- ビルド時間、クエリ時間、検索精度を評価
- 両手法の長所・短所を明確化

テストデータ: test.txt (864 chunks)
"""

from raptor_gmm import RAPTORRetrieverGMM
from raptor_classix import RAPTORRetrieverCLASSIX
from langchain_ollama import OllamaEmbeddings, ChatOllama
import time
from datetime import timedelta

print("=" * 80)
print("🔬 GMM+BIC vs CLASSIX 比較実験")
print("=" * 80)
print()

# 共通設定
FILE_PATH = "test.txt"
QUERY = "philosophy"
TOP_K = 5

# モデル設定（両手法で共通）
embeddings_model = OllamaEmbeddings(model="mxbai-embed-large")
llm = ChatOllama(model="granite-code:8b", temperature=0.0)

print("📊 共通設定:")
print(f"   データ: {FILE_PATH}")
print(f"   クエリ: '{QUERY}'")
print(f"   Top-K: {TOP_K}")
print(f"   LLM: granite-code:8b")
print(f"   Embeddings: mxbai-embed-large")
print()

# ================================================================================
# テスト1: GMM+BIC (BIC自動選択)
# ================================================================================

print("=" * 80)
print("📊 Test 1: GMM+BIC (BIC自動選択)")
print("=" * 80)
print()

raptor_gmm = RAPTORRetrieverGMM(
    embeddings_model=embeddings_model,
    llm=llm,
    max_clusters=5,      # BICで自動選択
    min_clusters=2,
    max_depth=2,
    use_bic=True,
    clustering_method="gmm"
)

print("🚀 GMM+BIC initialized")
print(f"   Parameters: max_clusters=5, min_clusters=2, max_depth=2")
print()

# ビルド時間測定
start_time = time.time()
raptor_gmm.index(FILE_PATH)
gmm_build_time = time.time() - start_time

print(f"\n✅ GMM+BIC Indexing complete!")
print(f"   Build time: {gmm_build_time:.2f}秒 ({timedelta(seconds=int(gmm_build_time))})")
print()

# クエリ時間測定
start_time = time.time()
gmm_results = raptor_gmm.retrieve(QUERY, top_k=TOP_K)
gmm_query_time = time.time() - start_time

print(f"🔍 Query: {QUERY}")
print(f"✅ Found {len(gmm_results)} results")
print(f"   Query time: {gmm_query_time:.3f}秒")
print()

# 結果表示
if gmm_results:
    gmm_top_similarity = gmm_results[0].metadata.get('similarity', 0.0)
    print(f"   Top similarity: {gmm_top_similarity}")
    print()
    print("   Top 3 Results:")
    for i, doc in enumerate(gmm_results[:3], 1):
        score = doc.metadata.get('similarity', 0.0)
        print(f"   {i}. Similarity: {score}")
        print(f"      Preview: {doc.page_content[:100]}...")
        print()
else:
    gmm_top_similarity = 0.0
    print("   No results found")
    print()

# ================================================================================
# テスト2: CLASSIX (radius=1.0, 最適設定)
# ================================================================================

print("=" * 80)
print("📊 Test 2: CLASSIX (radius=1.0, 最適設定)")
print("=" * 80)
print()

raptor_classix = RAPTORRetrieverCLASSIX(
    embeddings_model=embeddings_model,
    llm=llm,
    radius=1.0,          # 実験で検証済み最適値
    minPts=3,
    max_depth=2,
    use_cosine=True
)

print("🚀 CLASSIX initialized")
print(f"   Parameters: radius=1.0, minPts=3, max_depth=2")
print()

# ビルド時間測定
start_time = time.time()
raptor_classix.index(FILE_PATH)
classix_build_time = time.time() - start_time

print(f"\n✅ CLASSIX Indexing complete!")
print(f"   Build time: {classix_build_time:.2f}秒 ({timedelta(seconds=int(classix_build_time))})")
print()

# クエリ時間測定
start_time = time.time()
classix_results = raptor_classix.retrieve(QUERY, top_k=TOP_K)
classix_query_time = time.time() - start_time

print(f"🔍 Query: {QUERY}")
print(f"✅ Found {len(classix_results)} results")
print(f"   Query time: {classix_query_time:.3f}秒")
print()

# 結果表示
if classix_results:
    classix_top_similarity = classix_results[0].metadata.get('similarity', 0.0)
    print(f"   Top similarity: {classix_top_similarity}")
    print()
    print("   Top 3 Results:")
    for i, doc in enumerate(classix_results[:3], 1):
        score = doc.metadata.get('similarity', 0.0)
        print(f"   {i}. Similarity: {score}")
        print(f"      Preview: {doc.page_content[:100]}...")
        print()
else:
    classix_top_similarity = 0.0
    print("   No results found")
    print()

# ================================================================================
# テスト3: CLASSIX (radius=0.5, バランス型)
# ================================================================================

print("=" * 80)
print("📊 Test 3: CLASSIX (radius=0.5, バランス型)")
print("=" * 80)
print()

raptor_classix_balanced = RAPTORRetrieverCLASSIX(
    embeddings_model=embeddings_model,
    llm=llm,
    radius=0.5,
    minPts=3,
    max_depth=2,
    use_cosine=True
)

print("🚀 CLASSIX (balanced) initialized")
print(f"   Parameters: radius=0.5, minPts=3, max_depth=2")
print()

# ビルド時間測定
start_time = time.time()
raptor_classix_balanced.index(FILE_PATH)
classix_balanced_build_time = time.time() - start_time

print(f"\n✅ CLASSIX (balanced) Indexing complete!")
print(f"   Build time: {classix_balanced_build_time:.2f}秒 ({timedelta(seconds=int(classix_balanced_build_time))})")
print()

# クエリ時間測定
start_time = time.time()
classix_balanced_results = raptor_classix_balanced.retrieve(QUERY, top_k=TOP_K)
classix_balanced_query_time = time.time() - start_time

print(f"🔍 Query: {QUERY}")
print(f"✅ Found {len(classix_balanced_results)} results")
print(f"   Query time: {classix_balanced_query_time:.3f}秒")
print()

# 結果表示
if classix_balanced_results:
    classix_balanced_top_similarity = classix_balanced_results[0].metadata.get('similarity', 0.0)
    print(f"   Top similarity: {classix_balanced_top_similarity}")
    print()
    print("   Top 3 Results:")
    for i, doc in enumerate(classix_balanced_results[:3], 1):
        score = doc.metadata.get('similarity', 0.0)
        print(f"   {i}. Similarity: {score}")
        print(f"      Preview: {doc.page_content[:100]}...")
        print()
else:
    classix_balanced_top_similarity = 0.0
    print("   No results found")
    print()

# ================================================================================
# 比較サマリー
# ================================================================================

print("=" * 80)
print("📊 比較サマリー")
print("=" * 80)
print()

# 表形式で比較
print("| Method | Build時間 | Query時間 | Top類似度 | 総時間 |")
print("|--------|-----------|-----------|----------|--------|")
print(f"| GMM+BIC | {gmm_build_time:.2f}秒 | {gmm_query_time:.3f}秒 | {gmm_top_similarity:.5f} | {gmm_build_time + gmm_query_time:.2f}秒 |")
print(f"| CLASSIX (r=1.0) | {classix_build_time:.2f}秒 | {classix_query_time:.3f}秒 | {classix_top_similarity:.5f} | {classix_build_time + classix_query_time:.2f}秒 |")
print(f"| CLASSIX (r=0.5) | {classix_balanced_build_time:.2f}秒 | {classix_balanced_query_time:.3f}秒 | {classix_balanced_top_similarity:.5f} | {classix_balanced_build_time + classix_balanced_query_time:.2f}秒 |")
print()

# パフォーマンス比較
print("🏆 パフォーマンス比較:")
print()

# 最速ビルド
fastest_build = min(gmm_build_time, classix_build_time, classix_balanced_build_time)
if fastest_build == gmm_build_time:
    fastest_build_method = "GMM+BIC"
elif fastest_build == classix_build_time:
    fastest_build_method = "CLASSIX (r=1.0)"
else:
    fastest_build_method = "CLASSIX (r=0.5)"
print(f"✅ 最速ビルド: {fastest_build_method} ({fastest_build:.2f}秒)")

# 最速クエリ
fastest_query = min(gmm_query_time, classix_query_time, classix_balanced_query_time)
if fastest_query == gmm_query_time:
    fastest_query_method = "GMM+BIC"
elif fastest_query == classix_query_time:
    fastest_query_method = "CLASSIX (r=1.0)"
else:
    fastest_query_method = "CLASSIX (r=0.5)"
print(f"✅ 最速クエリ: {fastest_query_method} ({fastest_query:.3f}秒)")

# 最高精度
best_similarity = max(gmm_top_similarity, classix_top_similarity, classix_balanced_top_similarity)
if best_similarity == gmm_top_similarity:
    best_similarity_method = "GMM+BIC"
elif best_similarity == classix_top_similarity:
    best_similarity_method = "CLASSIX (r=1.0)"
else:
    best_similarity_method = "CLASSIX (r=0.5)"
print(f"✅ 最高精度: {best_similarity_method} ({best_similarity:.5f})")
print()

# 速度差の計算
print("📈 速度比較:")
print()
print(f"   GMM+BIC vs CLASSIX (r=1.0):")
if classix_build_time > 0:
    build_speedup = gmm_build_time / classix_build_time
    if build_speedup > 1:
        print(f"      Build: CLASSIX が {build_speedup:.2f}x 速い")
    else:
        print(f"      Build: GMM+BIC が {1/build_speedup:.2f}x 速い")
if classix_query_time > 0:
    query_speedup = gmm_query_time / classix_query_time
    if query_speedup > 1:
        print(f"      Query: CLASSIX が {query_speedup:.2f}x 速い")
    else:
        print(f"      Query: GMM+BIC が {1/query_speedup:.2f}x 速い")
print()

print(f"   GMM+BIC vs CLASSIX (r=0.5):")
if classix_balanced_build_time > 0:
    build_speedup = gmm_build_time / classix_balanced_build_time
    if build_speedup > 1:
        print(f"      Build: CLASSIX が {build_speedup:.2f}x 速い")
    else:
        print(f"      Build: GMM+BIC が {1/build_speedup:.2f}x 速い")
if classix_balanced_query_time > 0:
    query_speedup = gmm_query_time / classix_balanced_query_time
    if query_speedup > 1:
        print(f"      Query: CLASSIX が {query_speedup:.2f}x 速い")
    else:
        print(f"      Query: GMM+BIC が {1/query_speedup:.2f}x 速い")
print()

# 精度比較
print("🎯 精度比較:")
print()
print(f"   GMM+BIC:          {gmm_top_similarity:.5f}")
print(f"   CLASSIX (r=1.0):  {classix_top_similarity:.5f} (差: {classix_top_similarity - gmm_top_similarity:+.5f})")
print(f"   CLASSIX (r=0.5):  {classix_balanced_top_similarity:.5f} (差: {classix_balanced_top_similarity - gmm_top_similarity:+.5f})")
print()

# 総合評価
print("=" * 80)
print("💡 総合評価")
print("=" * 80)
print()

# どちらが優れているか判定
total_gmm = gmm_build_time + gmm_query_time
total_classix = classix_build_time + classix_query_time
total_classix_balanced = classix_balanced_build_time + classix_balanced_query_time

print("⚖️ 総合スコア (速度 + 精度):")
print()

# 正規化スコア計算
min_time = min(total_gmm, total_classix, total_classix_balanced)
max_time = max(total_gmm, total_classix, total_classix_balanced)
min_sim = min(gmm_top_similarity, classix_top_similarity, classix_balanced_top_similarity)
max_sim = max(gmm_top_similarity, classix_top_similarity, classix_balanced_top_similarity)

# スコア計算 (時間は小さいほど良い、類似度は大きいほど良い)
def calculate_score(build_time, query_time, similarity):
    total_time = build_time + query_time
    # 時間スコア: 0-50点 (速いほど高得点)
    if max_time > min_time:
        time_score = 50 * (1 - (total_time - min_time) / (max_time - min_time))
    else:
        time_score = 50
    # 精度スコア: 0-50点 (高いほど高得点)
    if max_sim > min_sim:
        accuracy_score = 50 * (similarity - min_sim) / (max_sim - min_sim)
    else:
        accuracy_score = 50
    return time_score + accuracy_score

gmm_score = calculate_score(gmm_build_time, gmm_query_time, gmm_top_similarity)
classix_score = calculate_score(classix_build_time, classix_query_time, classix_top_similarity)
classix_balanced_score = calculate_score(classix_balanced_build_time, classix_balanced_query_time, classix_balanced_top_similarity)

print(f"   GMM+BIC:          {gmm_score:.1f}/100")
print(f"   CLASSIX (r=1.0):  {classix_score:.1f}/100")
print(f"   CLASSIX (r=0.5):  {classix_balanced_score:.1f}/100")
print()

# 推奨
if classix_score > gmm_score and classix_score > classix_balanced_score:
    print("🏆 推奨: CLASSIX (radius=1.0)")
    print("   理由: 最速ビルド & 高精度のバランス")
elif classix_balanced_score > gmm_score and classix_balanced_score > classix_score:
    print("🏆 推奨: CLASSIX (radius=0.5)")
    print("   理由: バランスの取れた性能")
else:
    print("🏆 推奨: GMM+BIC")
    print("   理由: 自動クラスター選択の柔軟性")
print()

print("=" * 80)
print("実験完了！")
print("=" * 80)
