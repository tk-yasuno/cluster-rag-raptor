"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) Implementation
with HDBSCAN (Hierarchical Density-Based Spatial Clustering)

改良点:
- K-means/GMM → HDBSCAN に変更
- クラスター数の自動決定（パラメータチューニング不要）
- ノイズ検出機能により意味の薄いチャンクを除外
- 階層性を持つcondensed treeの活用

利点:
✅ クラスタ数の自動決定
✅ ノイズ（意味の薄いチャンク）の検出・除外
✅ 密度ベースで自然なクラスタ形成
✅ 高次元埋め込み（mxbai-embed-large等）と相性良好
✅ condensed treeによる真の階層性

Required packages:
pip install -U langchain langchain-community langchain-ollama scikit-learn numpy hdbscan
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
import hdbscan
import numpy as np
from typing import List, Dict, Optional, Tuple
import uuid


class RAPTORRetrieverHDBSCAN:
    """
    RAPTOR with HDBSCAN: 密度ベースの階層的クラスタリング
    ノイズ検出による意味の薄いチャンクの除外機能付き
    """
    
    def __init__(
        self,
        embeddings_model,
        llm,
        min_cluster_size: int = 15,
        min_samples: int = 5,
        max_depth: int = 3,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        cluster_selection_method: str = 'eom',  # 'eom' or 'leaf'
        metric: str = 'euclidean',  # 'euclidean' or 'cosine'
        exclude_noise: bool = True
    ):
        """
        Args:
            embeddings_model: Embeddings model for vectorization
            llm: LLM for summarization
            min_cluster_size: HDBSCANの最小クラスタサイズ（重要パラメータ）
            min_samples: 密度推定のための最小サンプル数
            max_depth: Maximum tree depth
            chunk_size: Document chunk size
            chunk_overlap: Overlap between chunks
            cluster_selection_method: 'eom' (Excess of Mass) or 'leaf'
            metric: Distance metric ('euclidean' or 'cosine')
            exclude_noise: ノイズチャンクを除外するか
        """
        self.embeddings_model = embeddings_model
        self.llm = llm
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.cluster_selection_method = cluster_selection_method
        self.metric = metric
        self.exclude_noise = exclude_noise
        self.tree_structure = {}
        self.noise_stats = {
            'total_noise_chunks': 0,
            'noise_by_depth': {}
        }
        
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
    
    def cluster_documents_hdbscan(
        self, 
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        HDBSCANでドキュメントをクラスタリング
        
        Returns:
            cluster_labels: クラスタラベル（-1はノイズ）
            stats: クラスタリング統計情報
        """
        n_samples = len(embeddings)
        
        # サンプル数が少なすぎる場合
        if n_samples < self.min_cluster_size:
            print(f"⚠️  Sample size ({n_samples}) < min_cluster_size ({self.min_cluster_size})")
            print(f"   Creating single cluster with all documents")
            return np.zeros(n_samples, dtype=int), {
                'n_clusters': 1,
                'n_noise': 0,
                'noise_ratio': 0.0
            }
        
        # コサイン距離を使う場合は、埋め込みを正規化してからEuclidean距離を使用
        # これによりコサイン距離と同等の結果が得られる
        if self.metric == 'cosine':
            print(f"🔄 Normalizing embeddings for cosine similarity...")
            # L2正規化
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings / (norms + 1e-10)  # ゼロ除算回避
            metric_to_use = 'euclidean'
            embeddings_to_use = embeddings_normalized
        else:
            metric_to_use = self.metric
            embeddings_to_use = embeddings
        
        # HDBSCAN実行
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric=metric_to_use,
            cluster_selection_method=self.cluster_selection_method,
            core_dist_n_jobs=-1  # 並列処理
        )
        
        cluster_labels = clusterer.fit_predict(embeddings_to_use)
        
        # 統計情報を収集
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        noise_ratio = n_noise / n_samples if n_samples > 0 else 0
        
        stats = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'cluster_sizes': {},
            'clusterer': clusterer  # condensed treeへのアクセス用
        }
        
        # 各クラスタのサイズ
        for label in unique_labels:
            if label != -1:
                stats['cluster_sizes'][label] = list(cluster_labels).count(label)
        
        print(f"\n🔍 HDBSCAN Clustering Results:")
        print(f"   Clusters found: {n_clusters}")
        print(f"   Noise points: {n_noise} ({noise_ratio*100:.1f}%)")
        if n_clusters > 0:
            print(f"   Cluster sizes: {stats['cluster_sizes']}")
        
        return cluster_labels, stats
    
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
        """再帰的にツリー構造を構築（HDBSCAN版）"""
        print(f"\n{'='*80}")
        print(f"Building tree at depth {depth} with {len(documents)} documents")
        print(f"{'='*80}")
        
        if depth >= self.max_depth or len(documents) < self.min_cluster_size:
            print(f"✋ Reached max depth or insufficient documents. Creating leaf node.")
            return {
                'depth': depth,
                'documents': documents,
                'is_leaf': True
            }
        
        # ドキュメントをベクトル化
        embeddings = self.embed_documents(documents)
        
        # HDBSCANでクラスタリング
        cluster_labels, stats = self.cluster_documents_hdbscan(embeddings)
        
        # ノイズ統計を記録
        if depth not in self.noise_stats['noise_by_depth']:
            self.noise_stats['noise_by_depth'][depth] = 0
        self.noise_stats['noise_by_depth'][depth] += stats['n_noise']
        self.noise_stats['total_noise_chunks'] += stats['n_noise']
        
        # クラスタが見つからない場合はリーフノードとして扱う
        if stats['n_clusters'] == 0:
            print(f"⚠️  No clusters found. Creating leaf node.")
            return {
                'depth': depth,
                'documents': documents,
                'is_leaf': True
            }
        
        # 各クラスタを処理
        clusters = {}
        summaries = []
        
        unique_labels = set(cluster_labels)
        
        for cluster_id in unique_labels:
            # ノイズを除外
            if cluster_id == -1:
                if self.exclude_noise:
                    print(f"🗑️  Excluding {stats['n_noise']} noise points")
                    continue
                else:
                    print(f"⚠️  Including {stats['n_noise']} noise points in separate cluster")
            
            cluster_docs = [doc for i, doc in enumerate(documents) if cluster_labels[i] == cluster_id]
            
            if len(cluster_docs) == 0:
                continue
            
            cluster_label = "noise" if cluster_id == -1 else str(cluster_id)
            print(f"\n📦 Cluster {cluster_label}: {len(cluster_docs)} documents")
            
            # クラスタの要約を生成
            summary = self.summarize_cluster(cluster_docs)
            summary_doc = Document(
                page_content=summary,
                metadata={
                    'type': 'summary',
                    'depth': depth,
                    'cluster_id': cluster_id,
                    'num_source_docs': len(cluster_docs),
                    'is_noise': cluster_id == -1
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
            'is_leaf': False,
            'hdbscan_stats': stats
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
            
            # コサイン類似度を計算
            similarities = np.dot(doc_embeddings, query_embedding) / (
                np.linalg.norm(doc_embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Top-k を選択
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                doc = docs[idx]
                doc.metadata['similarity'] = float(similarities[idx])
                results.append(doc)
            
            return results
        
        # 内部ノード: 要約を検索し、最も関連性の高いクラスタに進む
        summaries = tree.get('summaries', [])
        if not summaries:
            return []
        
        # クエリと要約の類似度を計算
        query_embedding = np.array(self.embeddings_model.embed_query(query))
        summary_embeddings = self.embed_documents(summaries)
        
        similarities = np.dot(summary_embeddings, query_embedding) / (
            np.linalg.norm(summary_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        # 最も関連性の高いクラスタを選択
        best_cluster_idx = np.argmax(similarities)
        cluster_id = summaries[best_cluster_idx].metadata['cluster_id']
        
        # 選択されたクラスタの子ノードを検索
        best_cluster = tree['clusters'][cluster_id]
        return self.search_tree(best_cluster['children'], query, top_k)
    
    def index(self, file_path: str, encoding: str = "utf-8"):
        """ドキュメントをインデックス化"""
        print(f"\n{'='*80}")
        print(f"🚀 RAPTOR with HDBSCAN - Document Indexing")
        print(f"{'='*80}")
        print(f"📄 File: {file_path}")
        print(f"📊 Parameters:")
        print(f"   - min_cluster_size: {self.min_cluster_size}")
        print(f"   - min_samples: {self.min_samples}")
        print(f"   - max_depth: {self.max_depth}")
        print(f"   - metric: {self.metric}")
        print(f"   - exclude_noise: {self.exclude_noise}")
        print(f"{'='*80}")
        
        # ドキュメントの読み込みと分割
        documents = self.load_and_split_documents(file_path, encoding)
        
        # ツリー構造の構築
        self.tree_structure = self.build_tree(documents)
        
        # ノイズ統計を表示
        print(f"\n{'='*80}")
        print(f"🗑️  Noise Statistics")
        print(f"{'='*80}")
        print(f"   Total noise chunks excluded: {self.noise_stats['total_noise_chunks']}")
        print(f"   Noise by depth:")
        for depth, count in sorted(self.noise_stats['noise_by_depth'].items()):
            print(f"     Depth {depth}: {count} chunks")
        print(f"{'='*80}")
        
        print(f"\n✅ Indexing complete!")
    
    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """クエリに対して関連ドキュメントを検索"""
        if not self.tree_structure:
            raise ValueError("No documents indexed. Call index() first.")
        
        print(f"\n🔍 Query: {query}")
        results = self.search_tree(self.tree_structure, query, top_k)
        print(f"✅ Found {len(results)} results")
        
        return results


if __name__ == "__main__":
    # デモ実行
    print("RAPTOR with HDBSCAN - Demo")
    print("=" * 80)
    
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
    
    # RAPTOR with HDBSCAN
    raptor = RAPTORRetrieverHDBSCAN(
        embeddings_model=embeddings_model,
        llm=llm,
        min_cluster_size=15,
        min_samples=5,
        max_depth=2,
        chunk_size=1000,
        chunk_overlap=200,
        exclude_noise=True
    )
    
    # インデックス化
    raptor.index("test.txt")
    
    # 検索
    results = raptor.retrieve("philosophy", top_k=3)
    
    print("\n" + "=" * 80)
    print("Search Results:")
    print("=" * 80)
    for i, doc in enumerate(results, 1):
        similarity = doc.metadata.get('similarity', 'N/A')
        print(f"\n{i}. Similarity: {similarity}")
        print(f"   Content: {doc.page_content[:200]}...")
    print("\n" + "=" * 80)
