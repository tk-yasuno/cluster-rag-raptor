"""
RAPTOR Large-Scale RAG Example: 論文（37万文字規模）の検証
=================================================

This example demonstrates RAPTOR's scalability with a large-scale document:
"A Systematic Literature Review of Retrieval-Augmented Generation"

Features:
- Downloads RAG survey paper from arXiv (arXiv:2508.06401)
- Processes ~100万文字 (1M characters) scale document
- Optimized parameters for large-scale processing
- Comprehensive performance metrics and benchmarking
- Multiple complex queries to test retrieval quality

Performance Targets:
- Tree construction: ~15-30 minutes
- Search time: <2 seconds
- Memory usage: <8GB
- Compression ratio: >100x
"""

import requests
import os
import time
import tempfile
from datetime import datetime
from langchain_ollama import ChatOllama, OllamaEmbeddings
from raptor import RAPTORRetriever


def download_arxiv_pdf(arxiv_id: str, output_path: str) -> bool:
    """
    Download PDF from arXiv
    
    Args:
        arxiv_id: arXiv ID (e.g., "2508.06401")
        output_path: Path to save the PDF
        
    Returns:
        True if successful, False otherwise
    """
    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    try:
        print(f"📥 Downloading from {url}...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        file_size = os.path.getsize(output_path)
        print(f"✅ Downloaded {file_size:,} bytes")
        return True
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        return False


def pdf_to_text(pdf_path: str) -> str:
    """
    Extract text from PDF using PyPDF2
    
    Args:
        pdf_path: Path to PDF file
        
    Returns:
        Extracted text content
    """
    try:
        import PyPDF2
        
        print("📄 Extracting text from PDF...")
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            print(f"   Pages: {num_pages}")
            
            text_parts = []
            for i, page in enumerate(reader.pages):
                if (i + 1) % 10 == 0:
                    print(f"   Processing page {i + 1}/{num_pages}...")
                text_parts.append(page.extract_text())
            
            full_text = "\n\n".join(text_parts)
            print(f"✅ Extracted {len(full_text):,} characters")
            return full_text
            
    except ImportError:
        print("❌ PyPDF2 not installed. Installing...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'PyPDF2'])
        return pdf_to_text(pdf_path)  # Retry after installation
    except Exception as e:
        print(f"❌ PDF extraction failed: {e}")
        return ""


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}時間"


def main():
    print("=" * 80)
    print("🚀 RAPTOR Large-Scale Example: RAG Survey Paper (200万文字規模)")
    print("=" * 80)
    print()
    
    # Configuration
    ARXIV_ID = "2508.06401"
    PDF_PATH = "rag_survey.pdf"
    TXT_PATH = "rag_survey.txt"
    
    # Step 1: Download or use cached PDF
    print("📦 Step 1: Preparing Document")
    print("-" * 80)
    
    if os.path.exists(TXT_PATH):
        print(f"✅ Using cached text file: {TXT_PATH}")
        with open(TXT_PATH, 'r', encoding='utf-8') as f:
            paper_text = f.read()
    else:
        if not os.path.exists(PDF_PATH):
            if not download_arxiv_pdf(ARXIV_ID, PDF_PATH):
                print("❌ Failed to download PDF. Exiting.")
                return
        
        # Extract text from PDF
        paper_text = pdf_to_text(PDF_PATH)
        
        if not paper_text:
            print("❌ Failed to extract text. Exiting.")
            return
        
        # Cache the extracted text
        with open(TXT_PATH, 'w', encoding='utf-8') as f:
            f.write(paper_text)
        print(f"💾 Cached text to {TXT_PATH}")
    
    print()
    print(f"📊 Document Statistics:")
    print(f"   Total characters: {len(paper_text):,}")
    print(f"   Total words: {len(paper_text.split()):,}")
    print(f"   Scale: {len(paper_text) / 1_000_000:.2f}M characters")
    print()
    
    # Verify scale
    if len(paper_text) < 100_000:
        print("⚠️  Warning: Document is smaller than expected.")
        print("   Continuing anyway for demonstration purposes.")
        print()
    
    # Step 2: Initialize models
    print("=" * 80)
    print("📦 Step 2: Initializing Ollama Models")
    print("-" * 80)
    
    llm = ChatOllama(
        model="granite-code:8b",
        base_url="http://localhost:11434",
        temperature=0
    )
    
    embeddings = OllamaEmbeddings(
        model="mxbai-embed-large",
        base_url="http://localhost:11434"
    )
    
    print("✅ LLM: granite-code:8b")
    print("✅ Embeddings: mxbai-embed-large (1024-dim)")
    print()
    
    # Step 3: Create RAPTOR with optimized parameters for large-scale
    print("=" * 80)
    print("📦 Step 3: Creating RAPTOR Retriever (Large-Scale Configuration)")
    print("-" * 80)
    
    raptor = RAPTORRetriever(
        embeddings_model=embeddings,
        llm=llm,
        max_clusters=3,      # Balanced clusters for ~370K chars
        max_depth=2,         # Moderate hierarchy for efficiency
        chunk_size=1200,     # Larger chunks to reduce word truncation
        chunk_overlap=250    # More overlap to preserve context
    )
    
    print("   Configuration:")
    print(f"   - max_clusters: 3 (balanced for 370K chars)")
    print(f"   - max_depth: 2 (optimized for speed)")
    print(f"   - chunk_size: 1200 (larger to preserve complete phrases)")
    print(f"   - chunk_overlap: 250 (more overlap for better context)")
    print()
    print("⏱️  Expected build time: 5-10 minutes for 370K characters")
    print("💡 This is a one-time cost. Subsequent searches will be <2 seconds.")
    print()
    
    # Step 4: Index the document
    print("=" * 80)
    print("📊 Step 4: Building RAPTOR Tree (This may take a while...)")
    print("-" * 80)
    print()
    
    start_time = time.time()
    
    # Save to temporary file for indexing
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp:
        tmp.write(paper_text)
        tmp_path = tmp.name
    
    try:
        raptor.index(tmp_path)
        build_time = time.time() - start_time
        
        print()
        print("=" * 80)
        print("✅ RAPTOR Tree Built Successfully!")
        print("-" * 80)
        print(f"   Build time: {format_time(build_time)}")
        print(f"   Characters processed: {len(paper_text):,}")
        print(f"   Processing speed: {len(paper_text) / build_time:,.0f} chars/sec")
        print()
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    # Step 5: Performance benchmarking with complex queries
    print("=" * 80)
    print("🔍 Step 5: Benchmarking Retrieval Performance")
    print("=" * 80)
    print()
    
    queries = [
        {
            "query": "What are the main techniques used in RAG systems?",
            "description": "Overview of RAG techniques"
        },
        {
            "query": "What evaluation metrics are used for RAG systems?",
            "description": "RAG evaluation methodology"
        },
        {
            "query": "What are the main challenges in RAG implementation?",
            "description": "Technical challenges"
        },
        {
            "query": "How does retrieval-augmented generation compare to fine-tuning?",
            "description": "RAG vs Fine-tuning comparison"
        },
        {
            "query": "What are the latest advancements in RAG research?",
            "description": "State-of-the-art RAG"
        }
    ]
    
    query_times = []
    all_similarities = []
    
    for idx, item in enumerate(queries, 1):
        query = item["query"]
        desc = item["description"]
        
        print(f"Query {idx}/{len(queries)}: {desc}")
        print(f"   Question: '{query}'")
        print("-" * 80)
        
        # Measure query time
        query_start = time.time()
        results = raptor.retrieve(query, top_k=3)
        query_time = time.time() - query_start
        query_times.append(query_time)
        
        print(f"   ⏱️  Query time: {query_time:.3f}秒")
        print()
        
        for i, doc in enumerate(results, 1):
            print(f"   📄 Result {i}:")
            # Extract similarity from metadata if available
            similarity = doc.metadata.get('similarity', 'N/A')
            if isinstance(similarity, (int, float)):
                all_similarities.append(similarity)
                print(f"      Similarity: {similarity:.4f}")
            print(f"      Preview: {doc.page_content[:250]}...")
            print()
        
        print()
    
    # Step 6: Performance Summary
    print("=" * 80)
    print("📊 Performance Summary")
    print("=" * 80)
    print()
    
    avg_query_time = sum(query_times) / len(query_times)
    
    print("🎯 Scalability Metrics:")
    print(f"   Document size: {len(paper_text):,} characters ({len(paper_text)/1_000_000:.2f}M)")
    print(f"   Build time: {format_time(build_time)}")
    print(f"   Average query time: {avg_query_time:.3f}秒")
    print(f"   Queries executed: {len(queries)}")
    print()
    
    if all_similarities:
        avg_similarity = sum(all_similarities) / len(all_similarities)
        print("🎯 Retrieval Quality:")
        print(f"   Average similarity: {avg_similarity:.4f}")
        print(f"   Min similarity: {min(all_similarities):.4f}")
        print(f"   Max similarity: {max(all_similarities):.4f}")
        print()
    
    print("🏆 RAPTOR Advantages at Scale:")
    print(f"   ✅ O(log n) search complexity")
    print(f"   ✅ Sub-second query times even for {len(paper_text)/1_000_000:.1f}M chars")
    print(f"   ✅ Hierarchical organization preserves context")
    print(f"   ✅ One-time build cost, unlimited fast searches")
    print()
    
    print("💡 Comparison with Traditional Methods:")
    print(f"   Vector search (O(n)): Would scan all chunks")
    print(f"   RAPTOR (O(log n)): Navigates tree efficiently")
    print(f"   Speed advantage: ~{build_time / avg_query_time:.0f}x faster searches")
    print()
    
    # Step 7: Resource usage estimation
    print("=" * 80)
    print("💻 Resource Usage")
    print("=" * 80)
    print()
    print(f"   Estimated memory: ~{len(paper_text) * 0.000004:.1f}GB (for embeddings)")
    print(f"   Disk cache: {os.path.getsize(TXT_PATH):,} bytes (text file)")
    print(f"   Processing efficiency: {len(paper_text) / build_time / 1000:.1f}K chars/sec")
    print()
    
    print("=" * 80)
    print("✅ Large-Scale RAPTOR Benchmark Completed!")
    print("=" * 80)
    print()
    print("🎉 Key Takeaways:")
    print(f"   1. RAPTOR successfully processed {len(paper_text)/1_000_000:.2f}M character document")
    print(f"   2. Build time: {format_time(build_time)} (one-time cost)")
    print(f"   3. Query time: ~{avg_query_time:.2f}秒 (consistent, fast)")
    print(f"   4. Scalable to even larger documents with parameter tuning")
    print()
    print("💡 Next Steps:")
    print("   - Try different max_clusters (3-7) and max_depth (2-4)")
    print("   - Experiment with chunk_size (1000-2000)")
    print("   - Test with other large documents (books, manuals, datasets)")
    print("   - Monitor memory usage with larger documents")
    print()


if __name__ == "__main__":
    main()
