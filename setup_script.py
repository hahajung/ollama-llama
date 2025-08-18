#!/usr/bin/env python3
"""
Setup script to check and install required packages for the enhanced RAG system.
Run this before using the enhanced files.
"""

import subprocess
import sys
import importlib.util
from pathlib import Path

def check_package(package_name, import_name=None):
    """Check if a package is installed."""
    if import_name is None:
        import_name = package_name.replace('-', '_')
    
    spec = importlib.util.find_spec(import_name)
    return spec is not None

def install_package(package_name):
    """Install a package using pip."""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package_name])
        return True
    except subprocess.CalledProcessError:
        return False

def get_package_version(package_name):
    """Get the version of an installed package."""
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', package_name],
            capture_output=True, text=True, check=True
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Version:'):
                return line.split(':')[1].strip()
    except subprocess.CalledProcessError:
        pass
    return None

def setup_enhanced_rag():
    """Set up the enhanced RAG system with proper dependencies."""
    
    print("🔧 Enhanced RAG Setup")
    print("=" * 50)
    
    # Required packages with fallback options
    required_packages = [
        # Core packages
        ('langchain', 'langchain'),
        ('langchain-community', 'langchain_community'),
        ('chromadb', 'chromadb'),
        
        # Enhanced functionality
        ('pandas', 'pandas'),
        ('beautifulsoup4', 'bs4'),
        ('aiohttp', 'aiohttp'),
        ('loguru', 'loguru'),
        ('streamlit', 'streamlit'),
        ('numpy', 'numpy'),
    ]
    
    # Optional newer packages
    optional_packages = [
        ('langchain-ollama', 'langchain_ollama'),
        ('langchain-chroma', 'langchain_chroma'),
        ('langchain-core', 'langchain_core'),
    ]
    
    print("\n📦 Checking Required Packages...")
    
    missing_packages = []
    installed_packages = []
    
    for package, import_name in required_packages:
        if check_package(package, import_name):
            version = get_package_version(package)
            installed_packages.append((package, version))
            print(f"   ✅ {package} ({version or 'unknown version'})")
        else:
            missing_packages.append(package)
            print(f"   ❌ {package} - MISSING")
    
    print("\n📦 Checking Optional Packages...")
    
    optional_available = []
    for package, import_name in optional_packages:
        if check_package(package, import_name):
            version = get_package_version(package)
            optional_available.append((package, version))
            print(f"   ✅ {package} ({version or 'unknown version'})")
        else:
            print(f"   ⚠️  {package} - Not installed (will use fallback)")
    
    # Install missing required packages
    if missing_packages:
        print(f"\n🔄 Installing {len(missing_packages)} missing packages...")
        
        for package in missing_packages:
            print(f"   Installing {package}...")
            if install_package(package):
                print(f"   ✅ Successfully installed {package}")
            else:
                print(f"   ❌ Failed to install {package}")
                print(f"   💡 Try manually: pip install {package}")
    
    # Check langchain version and recommend upgrades
    print("\n🔍 Langchain Version Analysis...")
    
    langchain_version = get_package_version('langchain')
    if langchain_version:
        print(f"   Current langchain version: {langchain_version}")
        
        # Parse version
        try:
            version_parts = [int(x) for x in langchain_version.split('.')]
            if version_parts[0] == 0 and version_parts[1] < 1:
                print("   ⚠️  You have an older langchain version.")
                print("   💡 Consider upgrading for better compatibility:")
                print("   pip install --upgrade langchain langchain-community")
            else:
                print("   ✅ Langchain version looks good")
        except:
            print("   ⚠️  Could not parse langchain version")
    
    # Create compatibility imports file
    print("\n📝 Creating compatibility imports...")
    
    compatibility_content = '''# Auto-generated compatibility imports
"""Universal imports for different langchain versions"""

def get_ollama_embeddings():
    try:
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings
    except ImportError:
        try:
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings
        except ImportError:
            from langchain.embeddings import OllamaEmbeddings
            return OllamaEmbeddings

def get_ollama_llm():
    try:
        from langchain_ollama import ChatOllama
        return ChatOllama
    except ImportError:
        try:
            from langchain_community.chat_models import ChatOllama
            return ChatOllama
        except ImportError:
            from langchain.chat_models import ChatOllama
            return ChatOllama

def get_chroma():
    try:
        from langchain_chroma import Chroma
        return Chroma
    except ImportError:
        try:
            from langchain_community.vectorstores import Chroma
            return Chroma
        except ImportError:
            from langchain.vectorstores import Chroma
            return Chroma

def get_text_splitter():
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter

def get_documents():
    try:
        from langchain_core.documents import Document
        return Document
    except ImportError:
        try:
            from langchain.docstore.document import Document
            return Document
        except ImportError:
            from langchain.schema import Document
            return Document

def get_prompts():
    try:
        from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        return ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    except ImportError:
        from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        return ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
'''
    
    with open('compatibility_imports.py', 'w') as f:
        f.write(compatibility_content)
    print("   ✅ Created compatibility_imports.py")
    
    # Test the imports
    print("\n🧪 Testing Imports...")
    try:
        exec("from compatibility_imports import *")
        
        tests = [
            ("OllamaEmbeddings", "get_ollama_embeddings()"),
            ("ChatOllama", "get_ollama_llm()"),
            ("Chroma", "get_chroma()"),
            ("Document", "get_documents()"),
        ]
        
        for name, func_call in tests:
            try:
                result = eval(func_call)
                print(f"   ✅ {name}: {result.__module__}")
            except Exception as e:
                print(f"   ❌ {name}: {str(e)}")
                
    except Exception as e:
        print(f"   ❌ Import testing failed: {e}")
    
    print("\n🎉 Setup Complete!")
    print("\n📋 Next Steps:")
    print("   1. Copy the enhanced files to your project:")
    print("      - enhanced_itential_scraper.py -> scripts/scraping/")
    print("      - enhanced_embedder.py -> root directory")  
    print("      - enhanced_retriever.py -> app/core/")
    print("      - updated improved_chatbot.py -> app/core/ (backup first)")
    print("")
    print("   2. Run the enhanced pipeline:")
    print("      python scripts/scraping/enhanced_itential_scraper.py")
    print("      python enhanced_embedder.py")
    print("      streamlit run app/improved_ui.py")
    print("")
    print("   3. Test your query: 'What version of Python should I use for IAP 2023.1?'")

if __name__ == "__main__":
    setup_enhanced_rag()