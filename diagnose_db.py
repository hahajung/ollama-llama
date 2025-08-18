#!/usr/bin/env python3
"""
Quick diagnostic to check what's wrong with your database
"""

import os
from pathlib import Path

def diagnose_database():
    """Diagnose the technical_optimized_chroma_db database."""
    
    db_path = "./technical_optimized_chroma_db"
    
    print(f"=== Diagnosing {db_path} ===")
    
    # Check if directory exists
    if not os.path.exists(db_path):
        print("❌ Database directory doesn't exist!")
        return
    
    print("✅ Database directory exists")
    
    # Check contents
    try:
        files = list(Path(db_path).rglob("*"))
        print(f"📁 Found {len(files)} files/folders")
        
        if len(files) == 0:
            print("❌ Database directory is EMPTY!")
            print("   Run: python complete_technical_pipeline.py")
            return
        
        # Check for key Chroma files
        key_files = []
        for f in files:
            if f.is_file():
                size = f.stat().st_size
                print(f"   📄 {f.relative_to(db_path)}: {size:,} bytes")
                
                # Check for important Chroma files
                if f.name in ['chroma.sqlite3', 'data_level0.bin', 'header.bin', 'length.bin']:
                    key_files.append(f.name)
        
        print(f"\n🔑 Key Chroma files found: {key_files}")
        
        if 'chroma.sqlite3' not in key_files:
            print("❌ Missing chroma.sqlite3 - database may be corrupted")
            print("   Solution: Delete database and re-run pipeline")
            return
        
        # Try to load the database
        print("\n🧪 Testing database loading...")
        
        try:
            # Import with fallbacks
            try:
                from langchain_ollama import OllamaEmbeddings
                from langchain_chroma import Chroma
                print("   Using langchain_ollama and langchain_chroma")
            except ImportError:
                try:
                    from langchain_community.embeddings import OllamaEmbeddings
                    from langchain_community.vectorstores import Chroma
                    print("   Using langchain_community")
                except ImportError:
                    from langchain.embeddings import OllamaEmbeddings
                    from langchain.vectorstores import Chroma
                    print("   Using legacy langchain")
            
            # Initialize embeddings
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
            print("   ✅ Embeddings initialized")
            
            # Load the database
            vector_store = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
            print("   ✅ Chroma database loaded")
            
            # Test with a simple query
            test_results = vector_store.similarity_search("test", k=1)
            print(f"   ✅ Query test: {len(test_results)} results")
            
            # Get total document count
            try:
                collection = vector_store.get()
                doc_count = len(collection.get('documents', []))
                print(f"   📊 Total documents: {doc_count:,}")
                
                if doc_count == 0:
                    print("   ❌ Database is empty (0 documents)")
                    print("   Solution: Re-run technical_embedder.py")
                else:
                    print("   ✅ Database has documents - should work!")
                    
                    # Show sample document
                    if len(test_results) > 0:
                        sample_content = test_results[0].page_content[:100] + "..."
                        print(f"   📝 Sample content: {sample_content}")
                
            except Exception as e:
                print(f"   ⚠️  Could not get document count: {e}")
            
        except Exception as e:
            print(f"   ❌ Database loading failed: {e}")
            print("   Possible solutions:")
            print("     1. Check if Ollama is running: ollama serve")
            print("     2. Check if model exists: ollama pull nomic-embed-text")
            print("     3. Delete and rebuild database: rm -rf technical_optimized_chroma_db && python complete_technical_pipeline.py")
            return
    
    except Exception as e:
        print(f"❌ Error reading database directory: {e}")

def test_ollama():
    """Test if Ollama is working properly."""
    print("\n=== Testing Ollama ===")
    
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print("✅ Ollama is running")
            print(f"📦 Available models: {len(models)}")
            
            # Check for required models
            model_names = [m.get('name', '') for m in models]
            
            if any('nomic-embed-text' in name for name in model_names):
                print("   ✅ nomic-embed-text found")
            else:
                print("   ❌ nomic-embed-text missing - run: ollama pull nomic-embed-text")
            
            if any('mistral' in name for name in model_names):
                print("   ✅ mistral found")
            else:
                print("   ⚠️  mistral missing - run: ollama pull mistral:7b")
                
        else:
            print(f"❌ Ollama responded with status {response.status_code}")
            
    except Exception as e:
        print(f"❌ Ollama not accessible: {e}")
        print("   Start Ollama: ollama serve")

if __name__ == "__main__":
    test_ollama()
    diagnose_database()
    
    print("\n=== Summary ===")
    print("If database is empty or corrupted:")
    print("   rm -rf technical_optimized_chroma_db")
    print("   python complete_technical_pipeline.py")
    print("\nIf Ollama issues:")
    print("   ollama serve")
    print("   ollama pull nomic-embed-text")
    print("   ollama pull mistral:7b")