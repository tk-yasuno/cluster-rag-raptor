"""
RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval) Implementation
with CLASSIX (Clustering via Approximate Supervised Similarity Index)

CLASSIXの利点:
✅ クラスタ数の自動決定
✅ 高速・軽量（HDBSCANより高速）
✅ パラメータが少なく調整が容易
✅ 距離ベースで意味的埋め込みに対応
✅ ノイズ除去機能

パラメータ:
- radius: クラスタの半径（小さいほど細かく分割）
- minPts: クラスタの最小サイズ

Required packages:
pip install -U langchain langchain-community langchain-ollama scikit-learn numpy classix
"""

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from classix import CLASSIX
import numpy as np
from typing import List, Dict, Optional, Tuple
import uuid


class RAPTORRetrieverCLASSIX:
    """
    RAPTOR with CLASSIX: 高速で調整が容易な密度ベースクラスタリング
    """
    
    def __init__(
        self,
        embeddings_model,
        llm,
        radius: float = 0.5,
        minPts: int = 5,
        max_depth: int = 3,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        use_cosine: bool = True
    ):
        """
        Args:
            embeddings_model: Embeddings model for vectorization
            llm: LLM for summarization
            radius: クラスタの半径（0.3-0.8推奨、小さいほど細かく分割）
            minPts: クラスタの最小サイズ（3-10推奨）
            max_depth: Maximum tree depth
            chunk_size: Document chunk size
            chunk_overlap: Overlap between chunks
            use_cosine: コサイン距離を使用するか（True推奨）
        """
        self.embeddings_model = embeddings_model
        self.llm = llm
        self.radius = radius
        self.minPts = minPts
        self.max_depth = max_depth
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_cosine = use_cosine
        self.tree_structure = {}
        self.cluster_stats = {
            'total_clusters_by_depth': {},
            'noise_by_depth': {},
            'params': {
                'radius': radius,
                'minPts': minPts,
                'use_cosine': use_cosine
            }
        }
        
        print(f"🚀 RAPTOR with CLASSIX initialized")
        print(f"   Parameters: radius={radius}, minPts={minPts}, use_cosine={use_cosine}")
        
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
    
    def cluster_documents_classix(
        self, 
        embeddings: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        CLASSIXでドキュメントをクラスタリング
        
        Returns:
            cluster_labels: クラスタラベル（-1はノイズ）
            stats: クラスタリング統計情報
        """
        n_samples = len(embeddings)
        
        # サンプル数が少なすぎる場合
        if n_samples < self.minPts:
            print(f"⚠️  Sample size ({n_samples}) < minPts ({self.minPts})")
            print(f"   Creating single cluster with all documents")
            return np.zeros(n_samples, dtype=int), {
                'n_clusters': 1,
                'n_noise': 0,
                'noise_ratio': 0.0
            }
        
        # コサイン距離を使う場合は正規化
        if self.use_cosine:
            print(f"🔄 Normalizing embeddings for cosine similarity...")
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings_normalized = embeddings / (norms + 1e-10)
            embeddings_to_use = embeddings_normalized
        else:
            embeddings_to_use = embeddings
        
        # CLASSIX実行
        print(f"⚡ Running CLASSIX clustering (radius={self.radius}, minPts={self.minPts})...")
        
        clusterer = CLASSIX(
            radius=self.radius,
            minPts=self.minPts,
            verbose=0
        )
        
        clusterer.fit(embeddings_to_use)
        cluster_labels = clusterer.labels_
        
        # 統計情報を収集
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = int(np.sum(cluster_labels == -1))
        noise_ratio = n_noise / n_samples if n_samples > 0 else 0
        
        stats = {
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': noise_ratio,
            'cluster_sizes': {}
        }
        
        # 各クラスタのサイズ
        for label in unique_labels:
            if label != -1:
                stats['cluster_sizes'][int(label)] = int(np.sum(cluster_labels == label))
        
        print(f"\n🔍 CLASSIX Clustering Results:")
        print(f"   Clusters found: {n_clusters}")
        print(f"   Noise points: {n_noise} ({noise_ratio*100:.1f}%)")
        if n_clusters > 0:
            cluster_sizes_list = list(stats['cluster_sizes'].values())
            print(f"   Cluster size range: {min(cluster_sizes_list)} - {max(cluster_sizes_list)}")
            print(f"   Average cluster size: {np.mean(cluster_sizes_list):.1f}")
        
        return cluster_labels, stats
    
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
        """再帰的にツリー構造を構築（CLASSIX版）"""
        print(f"\n{'='*80}")
        print(f"Building tree at depth {depth} with {len(documents)} documents")
        print(f"{'='*80}")
        
        if depth >= self.max_depth or len(documents) < self.minPts:
            print(f"✋ Reached max depth or insufficient documents. Creating leaf node.")
            return {
                'depth': depth,
                'documents': documents,
                'is_leaf': True
            }
        
        # ドキュメントをベクトル化
        embeddings = self.embed_documents(documents)
        
        # CLASSIXでクラスタリング
        cluster_labels, stats = self.cluster_documents_classix(embeddings)
        
        # 統計を記録
        if depth not in self.cluster_stats['total_clusters_by_depth']:
            self.cluster_stats['total_clusters_by_depth'][depth] = 0
            self.cluster_stats['noise_by_depth'][depth] = 0
        
        self.cluster_stats['total_clusters_by_depth'][depth] += stats['n_clusters']
        self.cluster_stats['noise_by_depth'][depth] += stats['n_noise']
        
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
                print(f"🗑️  Excluding {stats['n_noise']} noise points")
                continue
            
            cluster_docs = [doc for i, doc in enumerate(documents) if cluster_labels[i] == cluster_id]
            
            if len(cluster_docs) == 0:
                continue
            
            print(f"\n📦 Cluster {cluster_id}: {len(cluster_docs)} documents")
            
            # クラスタの要約を生成
            summary = self.summarize_cluster(cluster_docs)
            summary_doc = Document(
                page_content=summary,
                metadata={
                    'type': 'summary',
                    'depth': depth,
                    'cluster_id': int(cluster_id),
                    'num_source_docs': len(cluster_docs)
                }
            )
            summaries.append(summary_doc)
            
            # 再帰的に子ノードを構築
            clusters[int(cluster_id)] = {
                'summary': summary_doc,
                'documents': cluster_docs,
                'children': self.build_tree(cluster_docs, depth + 1)
            }
        
        return {
            'depth': depth,
            'clusters': clusters,
            'summaries': summaries,
            'is_leaf': False,
            'classix_stats': stats
        }
    
    def search_tree(self, tree: Dict, query: str, top_k: int = 3) -> List[Document]:
        """ツリー構造を検索"""
        if tree.get('is_leaf', False):
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
        
        # 内部ノード
        summaries = tree.get('summaries', [])
        if not summaries:
            return []
        
        query_embedding = np.array(self.embeddings_model.embed_query(query))
        summary_embeddings = self.embed_documents(summaries)
        
        similarities = np.dot(summary_embeddings, query_embedding) / (
            np.linalg.norm(summary_embeddings, axis=1) * np.linalg.norm(query_embedding)
        )
        
        best_cluster_idx = np.argmax(similarities)
        cluster_id = summaries[best_cluster_idx].metadata['cluster_id']
        
        best_cluster = tree['clusters'][cluster_id]
        return self.search_tree(best_cluster['children'], query, top_k)
    
    def index(self, file_path: str, encoding: str = "utf-8"):
        """ドキュメントをインデックス化"""
        print(f"\n{'='*80}")
        print(f"🚀 RAPTOR with CLASSIX - Document Indexing")
        print(f"{'='*80}")
        print(f"📄 File: {file_path}")
        print(f"📊 Parameters:")
        print(f"   - radius: {self.radius}")
        print(f"   - minPts: {self.minPts}")
        print(f"   - max_depth: {self.max_depth}")
        print(f"   - use_cosine: {self.use_cosine}")
        print(f"{'='*80}")
        
        documents = self.load_and_split_documents(file_path, encoding)
        self.tree_structure = self.build_tree(documents)
        
        print(f"\n{'='*80}")
        print(f"📊 Clustering Statistics")
        print(f"{'='*80}")
        print(f"   Total clusters by depth:")
        for depth, count in sorted(self.cluster_stats['total_clusters_by_depth'].items()):
            noise = self.cluster_stats['noise_by_depth'].get(depth, 0)
            print(f"     Depth {depth}: {count} clusters, {noise} noise points")
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
    print("RAPTOR with CLASSIX - Demo")
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
    
    # CLASSIX版RAPTOR
    raptor = RAPTORRetrieverCLASSIX(
        embeddings_model=embeddings_model,
        llm=llm,
        radius=0.5,         # クラスタの半径
        minPts=5,           # 最小ポイント数
        max_depth=2,
        chunk_size=1000,
        chunk_overlap=200,
        use_cosine=True     # コサイン距離使用
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
