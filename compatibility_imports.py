# compatibility_imports.py
"""
Universal imports that work with both old and new langchain versions
Place this file in your project root and import from it
"""

# Try different import paths for maximum compatibility
def get_ollama_embeddings():
    """Get Ollama embeddings with backwards compatibility."""
    try:
        # New langchain structure (0.1.0+)
        from langchain_ollama import OllamaEmbeddings
        return OllamaEmbeddings
    except ImportError:
        try:
            # Older langchain-community structure
            from langchain_community.embeddings import OllamaEmbeddings
            return OllamaEmbeddings
        except ImportError:
            try:
                # Even older structure
                from langchain.embeddings import OllamaEmbeddings
                return OllamaEmbeddings
            except ImportError:
                raise ImportError(
                    "Could not import OllamaEmbeddings. Please install langchain-community:\n"
                    "pip install langchain-community"
                )

def get_ollama_llm():
    """Get Ollama LLM with backwards compatibility."""
    try:
        # New langchain structure
        from langchain_ollama import ChatOllama
        return ChatOllama
    except ImportError:
        try:
            # Older langchain-community structure
            from langchain_community.chat_models import ChatOllama
            return ChatOllama
        except ImportError:
            try:
                # Even older structure
                from langchain.chat_models import ChatOllama
                return ChatOllama
            except ImportError:
                raise ImportError(
                    "Could not import ChatOllama. Please install langchain-community:\n"
                    "pip install langchain-community"
                )

def get_chroma():
    """Get Chroma with backwards compatibility."""
    try:
        # New structure
        from langchain_chroma import Chroma
        return Chroma
    except ImportError:
        try:
            # Older structure
            from langchain_community.vectorstores import Chroma
            return Chroma
        except ImportError:
            try:
                # Even older
                from langchain.vectorstores import Chroma
                return Chroma
            except ImportError:
                raise ImportError(
                    "Could not import Chroma. Please install chromadb and langchain-chroma:\n"
                    "pip install chromadb langchain-chroma"
                )

def get_text_splitter():
    """Get text splitter with backwards compatibility."""
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        return RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            return RecursiveCharacterTextSplitter
        except ImportError:
            raise ImportError(
                "Could not import RecursiveCharacterTextSplitter. Please install langchain:\n"
                "pip install langchain"
            )

def get_documents():
    """Get Document class with backwards compatibility."""
    try:
        from langchain_core.documents import Document
        return Document
    except ImportError:
        try:
            from langchain.docstore.document import Document
            return Document
        except ImportError:
            try:
                from langchain.schema import Document
                return Document
            except ImportError:
                raise ImportError(
                    "Could not import Document. Please install langchain-core:\n"
                    "pip install langchain-core"
                )

def get_prompts():
    """Get prompt templates with backwards compatibility."""
    try:
        from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        return ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    except ImportError:
        try:
            from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
            return ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        except ImportError:
            raise ImportError(
                "Could not import prompt templates. Please install langchain-core:\n"
                "pip install langchain-core"
            )

# Test all imports and provide diagnostic info
def test_imports():
    """Test all imports and provide diagnostic information."""
    import sys
    
    print("🔍 Testing Langchain Imports...")
    print(f"Python version: {sys.version}")
    
    results = {}
    
    # Test each import
    tests = [
        ("OllamaEmbeddings", get_ollama_embeddings),
        ("ChatOllama", get_ollama_llm),
        ("Chroma", get_chroma),
        ("RecursiveCharacterTextSplitter", get_text_splitter),
        ("Document", get_documents),
        ("Prompts", get_prompts)
    ]
    
    for name, func in tests:
        try:
            result = func()
            results[name] = f"✅ {result.__module__}"
            print(f"   ✅ {name}: {result.__module__}")
        except ImportError as e:
            results[name] = f"❌ {str(e)}"
            print(f"   ❌ {name}: {str(e)}")
    
    return results

if __name__ == "__main__":
    test_imports()