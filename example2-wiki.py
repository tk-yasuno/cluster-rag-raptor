"""
RAPTOR Wikipedia RAG Example: Miyazaki Hayao
============================================

This example demonstrates using RAPTOR for hierarchical retrieval
from Wikipedia content about Miyazaki Hayao.

Features:
- Fetches content from Wikipedia API
- Builds hierarchical RAPTOR tree for efficient search
- Performs multi-query RAG with similarity scores
- Demonstrates real-world knowledge base application
"""

import requests
import os
import tempfile
from raptor import RAPTORRetriever
from langchain_ollama import ChatOllama, OllamaEmbeddings


def get_wikipedia_page(title: str) -> str:
    """
    Retrieve the full text content of a Wikipedia page.
    
    Args:
        title: Title of the Wikipedia page
        
    Returns:
        Full text content of the page as raw string
    """
    # Wikipedia API endpoint
    URL = "https://en.wikipedia.org/w/api.php"
    
    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "titles": title,
        "prop": "extracts",
        "explaintext": True,
    }
    
    # Custom User-Agent header to comply with Wikipedia's best practices
    headers = {"User-Agent": "RAPTOR_RAG_Example/1.0"}
    
    response = requests.get(URL, params=params, headers=headers)
    data = response.json()
    
    # Extracting page content
    page = next(iter(data["query"]["pages"].values()))
    return page["extract"] if "extract" in page else None


def main():
    print("=" * 70)
    print("🎬 RAPTOR Wikipedia RAG: Miyazaki Hayao")
    print("=" * 70)
    print()
    
    # Step 1: Initialize models
    print("📦 Step 1: Initializing Ollama models...")
    llm = ChatOllama(model="granite-code:8b", temperature=0)
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    print("✅ Models initialized")
    print()
    
    # Step 2: Fetch Wikipedia content
    print("🌐 Step 2: Fetching Wikipedia page for 'Hayao_Miyazaki'...")
    wiki_content = get_wikipedia_page("Hayao_Miyazaki")
    
    if not wiki_content:
        print("❌ Failed to fetch Wikipedia page")
        return
    
    content_length = len(wiki_content)
    print(f"✅ Fetched {content_length:,} characters")
    print(f"📄 Preview: {wiki_content[:200]}...")
    print()
    
    # Step 3: Create RAPTOR retriever
    print("🌳 Step 3: Creating RAPTOR retriever...")
    raptor = RAPTORRetriever(
        embeddings_model=embeddings,
        llm=llm,
        max_clusters=3,  # Adjusted for Wikipedia article size
        max_depth=2
    )
    print("✅ RAPTOR retriever created")
    print()
    
    # Step 4: Index the Wikipedia content
    print("📊 Step 4: Indexing Wikipedia content...")
    print("   This will:")
    print("   - Split text into chunks")
    print("   - Generate embeddings")
    print("   - Build hierarchical tree structure")
    print()
    
    # Save Wikipedia content to temporary file
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as tmp_file:
        tmp_file.write(wiki_content)
        tmp_file_path = tmp_file.name
    
    try:
        raptor.index(tmp_file_path)
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_file_path):
            os.remove(tmp_file_path)
    
    print()
    print("✅ Indexing complete!")
    print()
    
    # Step 5: Perform RAG queries
    print("=" * 70)
    print("🔍 Step 5: Performing RAG Queries")
    print("=" * 70)
    print()
    
    # Query 1: Animation studio
    query1 = "What animation studio did Miyazaki found?"
    print(f"Query 1: '{query1}'")
    print("-" * 70)
    
    results1 = raptor.retrieve(query1, top_k=3)
    
    print(f"\n📄 Top {len(results1)} results:")
    for i, doc in enumerate(results1, 1):
        print(f"\n--- Result {i} ---")
        print(f"Content: {doc.page_content[:300]}...")
        if doc.metadata:
            print(f"Metadata: {doc.metadata}")
    
    print()
    print("=" * 70)
    print()
    
    # Query 2: Awards and recognition
    query2 = "What awards has Miyazaki received?"
    print(f"Query 2: '{query2}'")
    print("-" * 70)
    
    results2 = raptor.retrieve(query2, top_k=3)
    
    print(f"\n📄 Top {len(results2)} results:")
    for i, doc in enumerate(results2, 1):
        print(f"\n--- Result {i} ---")
        print(f"Content: {doc.page_content[:300]}...")
        if doc.metadata:
            print(f"Metadata: {doc.metadata}")
    
    print()
    print("=" * 70)
    print()
    
    # Query 3: Famous works
    query3 = "What are Miyazaki's most famous films?"
    print(f"Query 3: '{query3}'")
    print("-" * 70)
    
    results3 = raptor.retrieve(query3, top_k=3)
    
    print(f"\n📄 Top {len(results3)} results:")
    for i, doc in enumerate(results3, 1):
        print(f"\n--- Result {i} ---")
        print(f"Content: {doc.page_content[:300]}...")
        if doc.metadata:
            print(f"Metadata: {doc.metadata}")
    
    print()
    print("=" * 70)
    print()
    
    # Display statistics
    print("📊 RAPTOR Statistics:")
    print("-" * 70)
    print(f"   Total queries executed: 3")
    print(f"   Results per query: 3")
    print(f"   Tree depth: 2")
    print(f"   Max clusters per level: 3")
    print()
    
    print("=" * 70)
    print("✅ Wikipedia RAG Example Completed Successfully!")
    print("=" * 70)
    print()
    
    print("💡 Tips:")
    print("   - Try different Wikipedia pages by changing 'Hayao_Miyazaki'")
    print("   - Adjust max_clusters and max_depth for different document sizes")
    print("   - Use raptor.retrieve() with different queries")
    print("   - RAPTOR builds hierarchical tree for efficient O(log n) search")
    print()


if __name__ == "__main__":
    main()
