"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) Implementation
- 階層的文書クラスタリングと要約生成
- ツリー構造による効率的な検索
- オープンソースモデル使用: Granite Code 8B (LLM), mxbai-embed-large (Embeddings)

Required packages:
pip install -U langchain langchain-community langchain-ollama scikit-learn numpy
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Dict, Optional
import uuid


class RAPTORRetriever:
    """
    RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval
    階層的にドキュメントをクラスタリングし、各レベルで要約を生成
    """
    
    def __init__(
        self,
        embeddings_model,
        llm,
        max_clusters: int = 5,
        max_depth: int = 3,
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):
        self.embeddings_model = embeddings_model
        self.llm = llm
        self.max_clusters = max_clusters
        self.max_depth = max_depth
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.tree_structure = {}
        
    def load_and_split_documents(self, file_path: str, encoding: str = "utf-8") -> List[Document]:
        """ドキュメントを読み込み、チャンクに分割"""
        loader = TextLoader(file_path, encoding=encoding)
        docs = loader.load()
        
        print(f"Loaded document length: {len(docs[0].page_content)} characters")
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = splitter.split_documents(docs)
        print(f"Split into {len(chunks)} chunks")
        
        return chunks
    
    def embed_documents(self, documents: List[Document]) -> np.ndarray:
        """ドキュメントをベクトル化"""
        texts = [doc.page_content for doc in documents]
        embeddings = self.embeddings_model.embed_documents(texts)
        return np.array(embeddings)
    
    def cluster_documents(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """K-meansでドキュメントをクラスタリング"""
        # クラスタ数がドキュメント数より多い場合は調整
        n_clusters = min(n_clusters, len(embeddings))
        if n_clusters < 2:
            return np.zeros(len(embeddings), dtype=int)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        return cluster_labels
    
    def summarize_cluster(self, documents: List[Document]) -> str:
        """クラスタ内のドキュメントを要約"""
        # 複数のドキュメントを結合
        combined_text = "\n\n".join([doc.page_content for doc in documents])
        
        # 要約プロンプト
        prompt = ChatPromptTemplate.from_template(
            "以下のテキストを簡潔に要約してください。重要なポイントを保持しながら、"
            "全体の内容を200-300文字程度でまとめてください:\n\n{text}"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        summary = chain.invoke({"text": combined_text[:4000]})  # トークン制限対策
        
        return summary
    
    def build_tree(self, documents: List[Document], depth: int = 0) -> Dict:
        """再帰的にツリー構造を構築"""
        print(f"\n=== Building tree at depth {depth} with {len(documents)} documents ===")
        
        if depth >= self.max_depth or len(documents) <= self.max_clusters:
            print(f"Reached max depth or minimal documents. Creating leaf node.")
            return {
                'depth': depth,
                'documents': documents,
                'is_leaf': True
            }
        
        # ドキュメントをベクトル化
        embeddings = self.embed_documents(documents)
        
        # クラスタリング
        cluster_labels = self.cluster_documents(embeddings, self.max_clusters)
        
        # 各クラスタを処理
        clusters = {}
        summaries = []
        
        for cluster_id in range(max(cluster_labels) + 1):
            cluster_docs = [doc for i, doc in enumerate(documents) if cluster_labels[i] == cluster_id]
            
            if len(cluster_docs) == 0:
                continue
                
            print(f"Cluster {cluster_id}: {len(cluster_docs)} documents")
            
            # クラスタの要約を生成
            summary = self.summarize_cluster(cluster_docs)
            summary_doc = Document(
                page_content=summary,
                metadata={
                    'type': 'summary',
                    'depth': depth,
                    'cluster_id': cluster_id,
                    'num_source_docs': len(cluster_docs)
                }
            )
            summaries.append(summary_doc)
            
            # 再帰的に子ノードを構築
            clusters[cluster_id] = {
                'summary': summary_doc,
                'documents': cluster_docs,
                'children': self.build_tree(cluster_docs, depth + 1)
            }
        
        return {
            'depth': depth,
            'clusters': clusters,
            'summaries': summaries,
            'is_leaf': False
        }
    
    def search_tree(self, tree: Dict, query: str, top_k: int = 3) -> List[Document]:
        """ツリー構造を検索"""
        if tree.get('is_leaf', False):
            # リーフノード: 最も関連性の高いドキュメントを返す
            docs = tree['documents']
            if len(docs) == 0:
                return []
            
            # クエリとドキュメントの類似度を計算
            query_embedding = np.array(self.embeddings_model.embed_query(query))
            doc_embeddings = self.embed_documents(docs)
            
            # コサイン類似度
            similarities = np.dot(doc_embeddings, query_embedding) / (
                np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # 上位k件を取得
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [docs[i] for i in top_indices]
        
        # 内部ノード: 最も関連性の高いクラスタを選択
        clusters = tree['clusters']
        summaries = tree['summaries']
        
        if len(summaries) == 0:
            return []
        
        # クエリと要約の類似度を計算
        query_embedding = np.array(self.embeddings_model.embed_query(query))
        summary_embeddings = self.embed_documents(summaries)
        
        similarities = np.dot(summary_embeddings, query_embedding) / (
            np.linalg.norm(summary_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 最も関連性の高いクラスタを選択
        best_cluster_idx = np.argmax(similarities)
        best_cluster_id = list(clusters.keys())[best_cluster_idx]
        
        print(f"Selected cluster {best_cluster_id} at depth {tree['depth']} (similarity: {similarities[best_cluster_idx]:.4f})")
        
        # 選択したクラスタの子ノードを再帰的に検索
        return self.search_tree(clusters[best_cluster_id]['children'], query, top_k)
    
    def index(self, file_path: str):
        """ドキュメントをインデックス化してツリー構造を構築"""
        print("=== Starting RAPTOR Indexing ===")
        documents = self.load_and_split_documents(file_path)
        self.tree_structure = self.build_tree(documents)
        print("\n=== RAPTOR Tree Construction Complete ===")
        
    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """クエリに基づいてツリーを検索"""
        print(f"\n=== Searching for: '{query}' ===")
        return self.search_tree(self.tree_structure, query, top_k)


# メイン実行
if __name__ == "__main__":
    # Ollama LLMとEmbeddingsの初期化
    llm = ChatOllama(
        model="granite-code:8b",
        base_url="http://localhost:11434",
        temperature=0
    )
    
    embeddings_model = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434"
    )
    
    # RAPTORレトリーバーの初期化
    raptor = RAPTORRetriever(
        embeddings_model=embeddings_model,
        llm=llm,
        max_clusters=3,  # 各レベルで最大3クラスタ
        max_depth=2,     # 最大2階層（処理時間を考慮）
        chunk_size=1000,
        chunk_overlap=200
    )
    
    # ドキュメントのインデックス化
    raptor.index("./test.txt")
    
    # 検索実行
    query = "philosophy"
    results = raptor.retrieve(query, top_k=3)
    
    print(f"\n=== Top {len(results)} Results for '{query}' ===")
    for i, doc in enumerate(results, 1):
        print(f"\n--- Result {i} ---")
        print(f"Content preview: {doc.page_content[:200]}...")
        if doc.metadata:
            print(f"Metadata: {doc.metadata}")
    
    # 別のクエリで検索
    query2 = "ancient history"
    results2 = raptor.retrieve(query2, top_k=3)
    
    print(f"\n=== Top {len(results2)} Results for '{query2}' ===")
    for i, doc in enumerate(results2, 1):
        print(f"\n--- Result {i} ---")
        print(f"Content preview: {doc.page_content[:200]}...")
