#!/usr/bin/env python3
"""
Diagnose Q&A quality and check if specific technical information was captured
"""

import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

def diagnose_qa_quality():
    """Check if specific technical Q&A was captured correctly."""
    
    print("=== Q&A Quality Diagnosis ===")
    
    # Load the database
    try:
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = Chroma(
            persist_directory="./technical_optimized_chroma_db",
            embedding_function=embeddings
        )
        
        collection = vector_store.get()
        doc_count = len(collection.get('documents', []))
        print(f"✅ Database loaded: {doc_count} documents")
        
    except Exception as e:
        print(f"❌ Failed to load database: {e}")
        return
    
    # Test specific technical queries
    test_queries = [
        "What Node.js version is required for IAP 2023.1?",
        "Node.js requirements for IAP 2023.1", 
        "Python version for IAP 2023.1",
        "MongoDB version for IAP 2023.1",
        "What versions of IAP are available?",
        "IAP 2023.1 dependencies",
        "IAP 2023.1 system requirements"
    ]
    
    print("\n=== Testing Specific Technical Queries ===")
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        try:
            # Get top results
            results = vector_store.similarity_search_with_score(query, k=3)
            
            if not results:
                print("❌ No results found")
                continue
            
            print(f"📊 Found {len(results)} results:")
            
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n  Result {i} (Score: {score:.3f}):")
                
                # Check metadata
                metadata = doc.metadata
                content_type = metadata.get('content_type', 'unknown')
                question = metadata.get('question', '')
                product = metadata.get('product', '')
                dependency = metadata.get('dependency', '')
                
                print(f"    Type: {content_type}")
                print(f"    Product: {product}")
                print(f"    Dependency: {dependency}")
                
                if question:
                    print(f"    Q&A Question: {question}")
                
                # Show content preview
                content = doc.page_content
                print(f"    Content: {content[:200]}...")
                
                # Check if it contains specific version info
                version_indicators = ['2023.1', 'node.js', 'nodejs', '>=', '<=', 'version']
                found_indicators = [v for v in version_indicators if v.lower() in content.lower()]
                
                if found_indicators:
                    print(f"    Version indicators: {found_indicators}")
                else:
                    print("    ⚠️ No specific version indicators found")
        
        except Exception as e:
            print(f"❌ Query failed: {e}")
    
    # Check for Q&A pairs specifically
    print("\n=== Checking for Technical Q&A Pairs ===")
    
    try:
        # Search for Q&A content specifically
        qa_results = vector_store.similarity_search("Question: What version", k=10)
        qa_count = len(qa_results)
        
        print(f"📋 Found {qa_count} Q&A pairs")
        
        if qa_count > 0:
            print("\nSample Q&A pairs:")
            for i, doc in enumerate(qa_results[:3], 1):
                metadata = doc.metadata
                content_type = metadata.get('content_type', 'unknown')
                
                if 'qa' in content_type.lower():
                    print(f"\n  Q&A {i}:")
                    print(f"    Type: {content_type}")
                    print(f"    Content: {doc.page_content[:300]}...")
        
        # Check for dependency-specific content
        dep_results = vector_store.similarity_search("IAP 2023.1 Node.js", k=5)
        print(f"\n🔗 Dependency-specific results: {len(dep_results)}")
        
        for i, doc in enumerate(dep_results[:2], 1):
            print(f"\n  Dependency result {i}:")
            print(f"    Content: {doc.page_content[:200]}...")
            
    except Exception as e:
        print(f"❌ Q&A check failed: {e}")
    
    print("\n=== Recommendations ===")
    print("If specific technical Q&A is missing:")
    print("1. Check if scraper captured dependency tables")
    print("2. Verify embedder created version-specific Q&A pairs") 
    print("3. Look for 'dependencies' or 'requirements' pages in scraped data")
    print("4. Consider re-running pipeline with dependency page URLs")

if __name__ == "__main__":
    diagnose_qa_quality()