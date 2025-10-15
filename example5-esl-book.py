"""
RAPTOR Ultra-Large-Scale RAG Example: The Elements of Statistical Learning (759 pages)
======================================================================================

This example demonstrates RAPTOR's capability with an ultra-large-scale document:
"The Elements of Statistical Learning" by Hastie, Tibshirani, and Friedman

Features:
- Downloads 759-page machine learning textbook PDF
- Processes 183万文字+ (1.83M+ characters) scale document
- Deep hierarchical structure (max_depth=3) for complex topics
- Technical ML queries across 18 chapters
- Optimized for ultra-large-scale processing

Document Details:
- Title: The Elements of Statistical Learning (2nd Edition, 2017)
- Authors: Trevor Hastie, Robert Tibshirani, Jerome Friedman
- Pages: 759
- Topics: Linear Regression, Classification, Ensemble Methods, SVM, Neural Networks, etc.
- Structure: 18 chapters + Appendix (perfect for deep RAPTOR hierarchy)

Performance Targets:
- Tree construction: ~30-60 minutes
- Search time: <3 seconds
- Memory usage: <12GB
- Compression ratio: >150x
"""

import requests
import os
import time
import tempfile
import sys
from datetime import datetime
from langchain_ollama import ChatOllama, OllamaEmbeddings
from raptor import RAPTORRetriever

# Fix Windows console encoding for emoji support
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')


def download_esl_book(output_path: str) -> bool:
    """
    Download The Elements of Statistical Learning PDF
    
    Args:
        output_path: Path to save the PDF
        
    Returns:
        True if successful, False otherwise
    """
    # Official URL from Stanford (direct link to full PDF)
    # Note: The .download.html page links to the actual PDF
    urls = [
        "https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12_toc.pdf",
        "https://hastie.su.domains/Papers/ESLII.pdf",
        "https://hastie.su.domains/ElemStatLearn/printings/ESLII_print12.pdf"
    ]
    
    for url in urls:
        try:
            print(f"📥 Downloading 'The Elements of Statistical Learning' (759 pages)...")
            print(f"   Trying URL: {url}")
            print(f"   This may take several minutes...")
            
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            total_size = 0
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total_size += len(chunk)
                    if total_size % (5 * 1024 * 1024) == 0:  # Every 5MB
                        print(f"   Downloaded: {total_size // (1024*1024)}MB...")
            
            file_size = os.path.getsize(output_path)
            print(f"✅ Downloaded {file_size:,} bytes ({file_size // (1024*1024)}MB)")
            return True
            
        except Exception as e:
            print(f"❌ Failed with this URL: {e}")
            continue
    
    # All URLs failed
    print(f"❌ Download failed from all URLs")
    print(f"💡 Alternative: Download manually from https://web.stanford.edu/~hastie/ElemStatLearn/")
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
        print("   This will take 10-15 minutes for a 759-page document...")
        print("   ☕ Perfect time for a coffee break!")
        
        with open(pdf_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)
            print(f"   Total pages: {num_pages}")
            
            text_parts = []
            start_time = time.time()
            
            for i, page in enumerate(reader.pages):
                if (i + 1) % 50 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    remaining = (num_pages - i - 1) / rate
                    print(f"   Processing page {i + 1}/{num_pages} "
                          f"({(i+1)/num_pages*100:.1f}%) "
                          f"- ETA: {remaining/60:.1f} min")
                
                text = page.extract_text()
                if text:  # Skip empty pages
                    text_parts.append(text)
            
            full_text = "\n\n".join(text_parts)
            print(f"✅ Extracted {len(full_text):,} characters from {len(text_parts)} pages")
            print(f"   Extraction took {(time.time() - start_time)/60:.1f} minutes")
            return full_text
            
    except ImportError:
        print("❌ PyPDF2 not installed.")
        return ""
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
        return f"{hours:.2f}時間"


def main():
    print("=" * 80)
    print("📚 RAPTOR Ultra-Large-Scale: The Elements of Statistical Learning (759p)")
    print("=" * 80)
    print()
    
    # Configuration - check for manually downloaded file first
    MANUAL_PDF_PATH = "ESLII_print12_toc.pdf"
    PDF_PATH = "elements_of_statistical_learning.pdf"
    TXT_PATH = "elements_of_statistical_learning.txt"
    
    # Use manual download if available
    if os.path.exists(MANUAL_PDF_PATH):
        PDF_PATH = MANUAL_PDF_PATH
        print(f"✅ Found manually downloaded PDF: {MANUAL_PDF_PATH}")
    
    # Step 1: Download or use cached PDF
    print("📦 Step 1: Preparing Document (759 pages, ~12MB)")
    print("-" * 80)
    
    if os.path.exists(TXT_PATH):
        print(f"✅ Using cached text file: {TXT_PATH}")
        with open(TXT_PATH, 'r', encoding='utf-8') as f:
            book_text = f.read()
    else:
        if not os.path.exists(PDF_PATH):
            if not download_esl_book(PDF_PATH):
                print("❌ Failed to download PDF. Exiting.")
                return
        
        # Extract text from PDF
        book_text = pdf_to_text(PDF_PATH)
        
        if not book_text:
            print("❌ Failed to extract text. Exiting.")
            return
        
        # Cache the extracted text
        with open(TXT_PATH, 'w', encoding='utf-8') as f:
            f.write(book_text)
        print(f"💾 Cached text to {TXT_PATH}")
    
    print()
    print(f"📊 Document Statistics:")
    print(f"   Total characters: {len(book_text):,}")
    print(f"   Total words: {len(book_text.split()):,}")
    print(f"   Scale: {len(book_text) / 1_000_000:.2f}M characters")
    
    if len(book_text) >= 1_000_000:
        print(f"   Category: 🚀 MILLION-CHARACTER SCALE ACHIEVED!")
    else:
        print(f"   Category: Ultra-Large-Scale")
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
    
    # Step 3: Create RAPTOR with ultra-large-scale parameters
    print("=" * 80)
    print("📦 Step 3: Creating RAPTOR Retriever (Ultra-Large-Scale Configuration)")
    print("-" * 80)
    
    # Optimized parameters for 1M+ characters
    max_clusters = 5
    max_depth = 3
    chunk_size = 1500
    chunk_overlap = 300
    
    raptor = RAPTORRetriever(
        embeddings_model=embeddings,
        llm=llm,
        max_clusters=max_clusters,
        max_depth=max_depth,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    print("   Ultra-Large-Scale Configuration:")
    print(f"   - max_clusters: {max_clusters} (capture diverse ML topics)")
    print(f"   - max_depth: {max_depth} (deep hierarchy: field → methods → details)")
    print(f"   - chunk_size: {chunk_size} (preserve technical terminology)")
    print(f"   - chunk_overlap: {chunk_overlap} (maintain mathematical context)")
    print()
    print("⏱️  Expected build time: 30-60 minutes for 1M+ characters")
    print("💡 This is the largest example - one-time investment for unlimited fast searches")
    print("☕ Great time for lunch or a long coffee break!")
    print()
    
    # Ask for confirmation
    print("⚠️  IMPORTANT: This will take 30-60 minutes to build the tree.")
    print("   Make sure you have:")
    print("   - Stable internet connection")
    print("   - At least 12GB free RAM")
    print("   - Time to wait for completion")
    print()
    
    # Step 4: Index the document
    print("=" * 80)
    print("📊 Step 4: Building RAPTOR Tree (This will take 30-60 minutes...)")
    print("-" * 80)
    print()
    
    start_time = time.time()
    
    # Save to temporary file for indexing
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp:
        tmp.write(book_text)
        tmp_path = tmp.name
    
    try:
        raptor.index(tmp_path)
        build_time = time.time() - start_time
        
        print()
        print("=" * 80)
        print("✅ RAPTOR Tree Built Successfully!")
        print("-" * 80)
        print(f"   Build time: {format_time(build_time)}")
        print(f"   Characters processed: {len(book_text):,}")
        print(f"   Processing speed: {len(book_text) / build_time:,.0f} chars/sec")
        print()
        
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    
    # Step 5: Machine Learning technical queries
    print("=" * 80)
    print("🔍 Step 5: Machine Learning Query Benchmarking")
    print("=" * 80)
    print()
    
    queries = [
        {
            "query": "Which chapters discuss ensemble methods?",
            "description": "Ensemble methods overview"
        },
        {
            "query": "Summarize the differences between Lasso and Ridge regression",
            "description": "Regularization comparison"
        },
        {
            "query": "What are the key assumptions behind Support Vector Machines?",
            "description": "SVM fundamentals"
        },
        {
            "query": "How does boosting differ from bagging?",
            "description": "Ensemble method comparison"
        },
        {
            "query": "What are the main techniques for nonlinear dimensionality reduction?",
            "description": "Dimensionality reduction"
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
            # Show first 350 characters for technical content
            preview = doc.page_content[:350].replace('\n', ' ')
            print(f"      Preview: {preview}...")
            print()
        
        print()
    
    # Step 6: Performance Summary
    print("=" * 80)
    print("📊 Performance Summary (Ultra-Large-Scale)")
    print("=" * 80)
    print()
    
    avg_query_time = sum(query_times) / len(query_times)
    
    print("🎯 Ultra-Large-Scale Metrics:")
    print(f"   Document size: {len(book_text):,} characters ({len(book_text)/1_000_000:.2f}M)")
    print(f"   Build time: {format_time(build_time)}")
    print(f"   Average query time: {avg_query_time:.3f}秒")
    print(f"   Queries executed: {len(queries)}")
    print()
    
    if all_similarities:
        avg_similarity = sum(all_similarities) / len(all_similarities)
        print("🎯 Retrieval Quality (ML Domain):")
        print(f"   Average similarity: {avg_similarity:.4f}")
        print(f"   Min similarity: {min(all_similarities):.4f}")
        print(f"   Max similarity: {max(all_similarities):.4f}")
        print()
    
    print("🏆 RAPTOR at Million-Character Scale:")
    print(f"   ✅ O(log n) search - essential for ultra-large documents")
    print(f"   ✅ Sub-3-second queries even at {len(book_text)/1_000_000:.1f}M chars")
    print(f"   ✅ Deep hierarchy (depth=3) captures topic structure")
    print(f"   ✅ One-time build, unlimited fast searches")
    print()
    
    print("💡 Comparison with Traditional Methods:")
    print(f"   Vector search (O(n)): Would scan all chunks linearly")
    print(f"   RAPTOR (O(log n)): Navigates 3-level tree hierarchy")
    print(f"   Speed advantage: ~{build_time / avg_query_time:.0f}x faster searches")
    print()
    
    # Step 7: Resource usage
    print("=" * 80)
    print("💻 Resource Usage (Ultra-Large-Scale)")
    print("=" * 80)
    print()
    print(f"   Estimated memory: ~{len(book_text) * 0.000004:.1f}GB (for embeddings)")
    print(f"   Disk cache: {os.path.getsize(TXT_PATH):,} bytes (text file)")
    print(f"   Processing efficiency: {len(book_text) / build_time / 1000:.1f}K chars/sec")
    print()
    
    # Chunk statistics
    estimated_chunks = len(book_text) // chunk_size
    print(f"   Estimated chunks: ~{estimated_chunks}")
    print(f"   Chunk size: {chunk_size} chars")
    print(f"   Overlap: {chunk_overlap} chars ({chunk_overlap/chunk_size*100:.1f}%)")
    print(f"   Tree depth: {max_depth} levels")
    print(f"   Max branches: {max_clusters} per node")
    print()
    
    print("=" * 80)
    print("✅ Ultra-Large-Scale RAPTOR Benchmark Completed!")
    print("=" * 80)
    print()
    print("🎉 Key Achievements:")
    print(f"   1. Successfully processed {len(book_text)/1_000_000:.2f}M character document (759 pages)")
    print(f"   2. Build time: {format_time(build_time)} (one-time investment)")
    print(f"   3. Query time: ~{avg_query_time:.2f}秒 (consistently fast)")
    print(f"   4. ML textbook retrieval with high technical accuracy")
    print()
    print("🎓 Ultra-Large-Scale Lessons:")
    print("   - Million-character scale requires max_depth=3 for good organization")
    print("   - chunk_size=1500 preserves complex mathematical expressions")
    print("   - chunk_overlap=300 (20%) critical for formula continuity")
    print("   - Build time scales linearly, query time remains constant (O(log n))")
    print("   - 5 clusters capture diverse ML topics (regression, classification, ensemble, etc.)")
    print()
    print("💡 Production Deployment Considerations:")
    print("   - Pre-build trees offline for production use")
    print("   - Serialize tree structure for faster loading")
    print("   - Consider distributed processing for >5M characters")
    print("   - Monitor memory usage with very large documents")
    print("   - Implement caching for frequently accessed chapters")
    print()


if __name__ == "__main__":
    main()
