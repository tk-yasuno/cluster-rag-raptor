"""
RAPTOR使用例
このスクリプトは、RAPTORの基本的な使い方を示します。
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from raptor import RAPTORRetriever

def main():
    print("🚀 RAPTOR Example - Starting...")
    
    # ステップ1: モデルの初期化
    print("\n📦 Initializing models...")
    llm = ChatOllama(
        model="granite-code:8b",
        base_url="http://localhost:11434",
        temperature=0
    )
    
    embeddings_model = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434"
    )
    
    # ステップ2: RAPTORレトリーバーの作成
    print("🌳 Creating RAPTOR retriever...")
    raptor = RAPTORRetriever(
        embeddings_model=embeddings_model,
        llm=llm,
        max_clusters=3,   # 各レベルで最大3クラスタ
        max_depth=2,      # 最大2階層
        chunk_size=1000,  # チャンクサイズ
        chunk_overlap=200 # オーバーラップ
    )
    
    # ステップ3: ドキュメントのインデックス化
    print("\n📚 Indexing document...")
    print("Note: ../test.txt を使用します")
    print("独自の文書を使う場合は、ファイルパスを変更してください")
    
    try:
        raptor.index("../test.txt")
    except FileNotFoundError:
        print("❌ Error: ../test.txt が見つかりません")
        print("💡 ヒント: 独自の.txtファイルを用意して、パスを変更してください")
        return
    
    # ステップ4: 検索実行
    print("\n🔍 Performing searches...")
    
    # クエリ1
    query1 = "philosophy"
    print(f"\n{'='*60}")
    print(f"Query 1: '{query1}'")
    print('='*60)
    results1 = raptor.retrieve(query1, top_k=3)
    
    print(f"\n📄 Top {len(results1)} results:")
    for i, doc in enumerate(results1, 1):
        print(f"\n--- Result {i} ---")
        print(f"Preview: {doc.page_content[:250]}...")
        if doc.metadata:
            print(f"Metadata: {doc.metadata}")
    
    # クエリ2
    query2 = "ancient history"
    print(f"\n{'='*60}")
    print(f"Query 2: '{query2}'")
    print('='*60)
    results2 = raptor.retrieve(query2, top_k=3)
    
    print(f"\n📄 Top {len(results2)} results:")
    for i, doc in enumerate(results2, 1):
        print(f"\n--- Result {i} ---")
        print(f"Preview: {doc.page_content[:250]}...")
        if doc.metadata:
            print(f"Metadata: {doc.metadata}")
    
    # 統計情報
    print(f"\n{'='*60}")
    print("📊 Statistics")
    print('='*60)
    print(f"Total queries executed: 2")
    print(f"Results per query: 3")
    print(f"Tree depth: 2")
    print(f"Max clusters per level: 3")
    
    print("\n✅ Example completed successfully!")
    print("\n💡 Next steps:")
    print("   1. Try different queries")
    print("   2. Adjust max_clusters and max_depth")
    print("   3. Use your own documents")
    print("   4. Customize the summarization prompt")

if __name__ == "__main__":
    main()
