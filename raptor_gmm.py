"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) Implementation
with GMM (Gaussian Mixture Model) and BIC-based optimal cluster selection

改良点:
- K-means → GMM (Gaussian Mixture Model) に変更
- BIC (Bayesian Information Criterion) による最適クラスター数の自動選択
- より柔軟なクラスタリング（楕円形クラスタに対応）

Required packages:
pip install -U langchain langchain-community langchain-ollama scikit-learn numpy
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
import numpy as np
from typing import List, Dict, Optional, Tuple
import uuid


class RAPTORRetrieverGMM:
    """
    RAPTOR with GMM: Gaussian Mixture Model based clustering
    BICによる最適クラスター数の自動選択機能付き
    """
    
    def __init__(
        self,
        embeddings_model,
        llm,
        max_clusters: int = 5,
        min_clusters: int = 2,
        max_depth: int = 3,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_bic: bool = True,
        clustering_method: str = "gmm"  # "gmm" or "kmeans"
    ):
        """
        Args:
            embeddings_model: Embeddings model for vectorization
            llm: LLM for summarization
            max_clusters: Maximum number of clusters to consider
            min_clusters: Minimum number of clusters to consider
            max_depth: Maximum tree depth
            chunk_size: Document chunk size
            chunk_overlap: Overlap between chunks
            use_bic: If True, use BIC to automatically select optimal cluster count
            clustering_method: "gmm" (Gaussian Mixture) or "kmeans" (K-means)
        """
        self.embeddings_model = embeddings_model
        self.llm = llm
        self.max_clusters = max_clusters
        self.min_clusters = min_clusters
        self.max_depth = max_depth
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_bic = use_bic
        self.clustering_method = clustering_method
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
    
    def find_optimal_clusters_bic(self, embeddings: np.ndarray) -> Tuple[int, List[float]]:
        """
        BIC (Bayesian Information Criterion) を使って最適なクラスター数を選択
        
        BICが最小となるクラスター数が最適
        BIC = -2 * log-likelihood + k * log(n)
        ここで、k = パラメータ数、n = サンプル数
        
        Returns:
            optimal_n_clusters: 最適なクラスター数
            bic_scores: 各クラスター数でのBICスコア
        """
        n_samples = len(embeddings)
        
        # 最小・最大クラスター数の制約
        min_k = max(self.min_clusters, 2)
        max_k = min(self.max_clusters, n_samples - 1)
        
        if max_k < min_k:
            print(f"⚠️  Sample size ({n_samples}) too small for clustering. Using 1 cluster.")
            return 1, [0.0]
        
        print(f"\n🔍 Searching for optimal cluster count using BIC...")
        print(f"   Range: {min_k} to {max_k} clusters")
        
        bic_scores = []
        cluster_range = range(min_k, max_k + 1)
        
        for n_clusters in cluster_range:
            try:
                if self.clustering_method == "gmm":
                    gmm = GaussianMixture(
                        n_components=n_clusters,
                        covariance_type='full',
                        random_state=42,
                        n_init=3
                    )
                    gmm.fit(embeddings)
                    bic = gmm.bic(embeddings)
                else:  # kmeans doesn't have native BIC, so we approximate
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    kmeans.fit(embeddings)
                    # Approximate BIC for K-means
                    labels = kmeans.labels_
                    centers = kmeans.cluster_centers_
                    # Calculate within-cluster sum of squares
                    wcss = sum([np.sum((embeddings[labels == i] - centers[i]) ** 2) 
                               for i in range(n_clusters)])
                    # Approximate log-likelihood
                    log_likelihood = -wcss
                    # BIC approximation: -2*log(L) + k*log(n)
                    k = n_clusters * embeddings.shape[1]  # パラメータ数
                    bic = -2 * log_likelihood + k * np.log(n_samples)
                
                bic_scores.append(bic)
                print(f"   k={n_clusters}: BIC={bic:.2f}")
                
            except Exception as e:
                print(f"   k={n_clusters}: Failed ({e})")
                bic_scores.append(float('inf'))
        
        # BICが最小となるクラスター数を選択
        optimal_idx = np.argmin(bic_scores)
        optimal_n_clusters = list(cluster_range)[optimal_idx]
        
        print(f"\n✅ Optimal cluster count: {optimal_n_clusters} (BIC={bic_scores[optimal_idx]:.2f})")
        
        return optimal_n_clusters, bic_scores
    
    def cluster_documents_gmm(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """
        Gaussian Mixture Model でドキュメントをクラスタリング
        
        GMMの利点:
        - ソフトクラスタリング（確率的割り当て）
        - 楕円形のクラスタに対応
        - 共分散行列を考慮
        """
        # クラスタ数がドキュメント数より多い場合は調整
        n_clusters = min(n_clusters, len(embeddings))
        if n_clusters < 2:
            return np.zeros(len(embeddings), dtype=int)
        
        gmm = GaussianMixture(
            n_components=n_clusters,
            covariance_type='full',  # 'full', 'tied', 'diag', 'spherical'
            random_state=42,
            n_init=3
        )
        cluster_labels = gmm.fit_predict(embeddings)
        
        # 各クラスタの確率も取得可能
        # probabilities = gmm.predict_proba(embeddings)
        
        return cluster_labels
    
    def cluster_documents_kmeans(self, embeddings: np.ndarray, n_clusters: int) -> np.ndarray:
        """K-meansでドキュメントをクラスタリング（従来の方法）"""
        n_clusters = min(n_clusters, len(embeddings))
        if n_clusters < 2:
            return np.zeros(len(embeddings), dtype=int)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        return cluster_labels
    
    def cluster_documents(self, embeddings: np.ndarray, n_clusters: Optional[int] = None) -> np.ndarray:
        """
        ドキュメントをクラスタリング
        
        Args:
            embeddings: Document embeddings
            n_clusters: クラスター数（Noneの場合はBICで自動選択）
        """
        # BICによる最適クラスター数の自動選択
        if n_clusters is None and self.use_bic:
            n_clusters, bic_scores = self.find_optimal_clusters_bic(embeddings)
        elif n_clusters is None:
            n_clusters = self.max_clusters
        
        # クラスタリング実行
        if self.clustering_method == "gmm":
            return self.cluster_documents_gmm(embeddings, n_clusters)
        else:
            return self.cluster_documents_kmeans(embeddings, n_clusters)
    
    def summarize_cluster(self, documents: List[Document]) -> str:
        """クラスタ内のドキュメントを要約"""
        combined_text = "\n\n".join([doc.page_content for doc in documents])
        
        prompt = ChatPromptTemplate.from_template(
            "以下のテキストを簡潔に要約してください。重要なポイントを保持しながら、"
            "全体の内容を200-300文字程度でまとめてください:\n\n{text}"
        )
        
        chain = prompt | self.llm | StrOutputParser()
        summary = chain.invoke({"text": combined_text[:4000]})
        
        return summary
    
    def build_tree(self, documents: List[Document], depth: int = 0) -> Dict:
        """再帰的にツリー構造を構築"""
        print(f"\n=== Building tree at depth {depth} with {len(documents)} documents ===")
        
        if depth >= self.max_depth or len(documents) <= self.max_clusters:
            print(f"Reached max depth or minimal documents. Creating leaf node.")
            return {
                "type": "leaf",
                "documents": documents,
                "depth": depth
            }
        
        # Embeddings生成
        embeddings = self.embed_documents(documents)
        
        # クラスタリング（BICによる最適数選択）
        cluster_labels = self.cluster_documents(embeddings, n_clusters=None)
        
        # 各クラスタに対して再帰的に処理
        clusters = {}
        for cluster_id in np.unique(cluster_labels):
            cluster_docs = [documents[i] for i in range(len(documents)) 
                           if cluster_labels[i] == cluster_id]
            
            print(f"Cluster {cluster_id}: {len(cluster_docs)} documents")
            
            # クラスタの要約を生成
            if len(cluster_docs) > 1:
                summary = self.summarize_cluster(cluster_docs)
                summary_doc = Document(
                    page_content=summary,
                    metadata={"type": "summary", "cluster_id": cluster_id, "depth": depth}
                )
            else:
                summary_doc = cluster_docs[0]
            
            # 子ノードを構築
            child_node = self.build_tree(cluster_docs, depth + 1)
            clusters[cluster_id] = {
                "summary": summary_doc,
                "child": child_node
            }
        
        return {
            "type": "internal",
            "clusters": clusters,
            "depth": depth
        }
    
    def index(self, file_path: str):
        """ドキュメントをインデックス化"""
        print("\n=== Starting RAPTOR Indexing (GMM-BIC) ===")
        print(f"Clustering method: {self.clustering_method.upper()}")
        print(f"BIC optimization: {'ON' if self.use_bic else 'OFF'}")
        
        documents = self.load_and_split_documents(file_path)
        self.tree_structure = self.build_tree(documents)
        print("\n=== RAPTOR Tree Construction Complete ===")
    
    def search_tree(self, query: str, node: Dict, depth: int = 0) -> List[Document]:
        """ツリーを検索"""
        if node["type"] == "leaf":
            return node["documents"]
        
        # クエリのembeddingを取得
        query_embedding = np.array(self.embeddings_model.embed_query(query))
        
        # 各クラスタの要約とクエリの類似度を計算
        best_similarity = -1
        best_cluster = None
        
        for cluster_id, cluster_data in node["clusters"].items():
            summary_embedding = np.array(
                self.embeddings_model.embed_query(cluster_data["summary"].page_content)
            )
            
            # コサイン類似度
            similarity = np.dot(query_embedding, summary_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(summary_embedding)
            )
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_cluster = cluster_id
        
        print(f"Selected cluster {best_cluster} at depth {depth} (similarity: {best_similarity:.4f})")
        
        # 最も類似したクラスタを再帰的に検索
        return self.search_tree(
            query, 
            node["clusters"][best_cluster]["child"],
            depth + 1
        )
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """クエリに関連するドキュメントを検索"""
        print(f"\n=== Searching for: '{query}' ===")
        
        # ツリーを探索
        candidate_docs = self.search_tree(query, self.tree_structure)
        
        # クエリとの類似度でランキング
        query_embedding = np.array(self.embeddings_model.embed_query(query))
        doc_embeddings = self.embed_documents(candidate_docs)
        
        similarities = []
        for doc_emb in doc_embeddings:
            sim = np.dot(query_embedding, doc_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_emb)
            )
            similarities.append(sim)
        
        # Top-k を返す
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        results = [candidate_docs[i] for i in top_indices]
        
        # 類似度をメタデータに追加
        for i, idx in enumerate(top_indices):
            results[i].metadata['similarity'] = similarities[idx]
        
        return results
