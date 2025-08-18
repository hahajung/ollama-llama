#!/usr/bin/env python3
"""
Enhanced UI for Itential Technical Documentation Assistant
Updated to work with the new technical pipeline and database paths.
"""

import streamlit as st
import os
import sys
from pathlib import Path
import time

# Add core directory to path for imports
current_dir = Path(__file__).parent
core_dir = current_dir / "core"
root_dir = current_dir.parent

sys.path.insert(0, str(core_dir))
sys.path.insert(0, str(root_dir))

# Import compatibility layer
try:
    from compatibility_imports import (
        get_ollama_embeddings, get_ollama_llm, 
        get_chroma, get_documents, get_prompts
    )
    OllamaEmbeddings = get_ollama_embeddings()
    ChatOllama = get_ollama_llm()
    Chroma = get_chroma()
    Document = get_documents()
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate = get_prompts()
except ImportError:
    # Direct imports as fallback
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import Chroma
        from langchain_community.chat_models import ChatOllama
        from langchain_core.prompts import ChatPromptTemplate
    except ImportError:
        try:
            from langchain.embeddings import OllamaEmbeddings
            from langchain.vectorstores import Chroma
            from langchain.chat_models import ChatOllama
            from langchain.prompts import ChatPromptTemplate
        except ImportError:
            st.error("❌ Missing langchain dependencies. Install with: pip install langchain langchain-community")
            st.stop()

# Page configuration
st.set_page_config(
    page_title="Itential Technical Assistant", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def init_technical_chatbot():
    """Initialize the technical chatbot with updated database paths."""
    
    st.write("🔧 **Initializing Enhanced Technical Assistant...**")
    
    # Updated database paths from the new pipeline
    database_options = [
        ("./technical_optimized_chroma_db", "Technical Optimized (New Pipeline)"),
        ("./qa_enhanced_chroma_db", "QA Enhanced"),
        ("./super_enhanced_chroma_db", "Super Enhanced"),
        ("./enhanced_chroma_db", "Enhanced"),
        ("./chroma_db", "Original")
    ]
    
    # Find the best available database
    db_path = None
    db_description = None
    
    for path, description in database_options:
        if os.path.exists(path) and os.path.isdir(path):
            db_path = path
            db_description = description
            break
    
    if not db_path:
        st.error("❌ No vector database found!")
        st.write("**Available directories:**")
        for item in sorted(os.listdir(".")):
            if os.path.isdir(item):
                st.write(f"  📁 {item}")
        
        st.write("**To create the database:**")
        st.code("python complete_technical_pipeline.py", language="bash")
        st.stop()
    
    st.write(f"📚 **Using database:** `{db_description}` at `{db_path}`")
    
    try:
        # Initialize components
        with st.spinner("🤖 Loading embeddings model..."):
            embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        with st.spinner("📚 Loading vector database..."):
            vector_store = Chroma(
                persist_directory=db_path,
                embedding_function=embeddings
            )
            
            # Test database
            test_results = vector_store.similarity_search("test", k=1)
            if not test_results:
                st.warning("⚠️ Database appears to be empty")
        
        with st.spinner("🧠 Loading language model..."):
            llm = ChatOllama(
                model="mistral:7b", 
                temperature=0.1,
                num_predict=512  # Limit response length for speed
            )
        
        # Test Ollama connection
        try:
            test_response = llm.invoke("Test")
            st.write("✅ **All components loaded successfully!**")
        except Exception as e:
            st.error(f"❌ Ollama connection failed: {e}")
            st.write("**Troubleshooting:**")
            st.write("1. Is Ollama running? Run: `ollama serve`")
            st.write("2. Is the model available? Run: `ollama pull mistral:7b`")
            st.stop()
        
        return {
            'vector_store': vector_store,
            'llm': llm,
            'embeddings': embeddings,
            'db_path': db_path,
            'db_description': db_description
        }
        
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        st.write("**Debug info:**")
        st.write(f"- Database path: {db_path}")
        st.write(f"- Path exists: {os.path.exists(db_path)}")
        st.write(f"- Is directory: {os.path.isdir(db_path)}")
        
        if os.path.exists(db_path):
            try:
                files = list(os.listdir(db_path))
                st.write(f"- Files in database: {len(files)}")
            except Exception:
                st.write("- Could not list database files")
        
        st.stop()

def enhanced_query(components, question, k=5):
    """Enhanced query function with technical optimization."""
    try:
        # Search for relevant documents
        docs = components['vector_store'].similarity_search(question, k=k)
        
        if not docs:
            return "No relevant documents found. Try rephrasing your question or check if the database contains the information you're looking for."
        
        # Build context from documents
        context_parts = []
        for i, doc in enumerate(docs, 1):
            content = doc.page_content[:800]  # Limit content length
            source = doc.metadata.get('source', 'Unknown')
            title = doc.metadata.get('title', 'Untitled')
            
            context_parts.append(f"Source {i} ({title}):\n{content}")
        
        context = "\n\n".join(context_parts)
        
        # Create enhanced prompt for technical queries
        system_prompt = """You are an expert Itential technical documentation assistant. Provide accurate, detailed answers based on the provided context.

GUIDELINES:
1. Give direct, actionable answers
2. Include specific version numbers, requirements, and technical details
3. Use code blocks for commands, configurations, or code examples
4. Be precise about system requirements and dependencies
5. If the context doesn't contain enough information, say so clearly
6. Cite relevant sources when possible

Focus on helping users with Itential Automation Platform (IAP), Itential Automation Gateway (IAG), and related technical requirements."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", f"Context:\n{context}\n\nQuestion: {question}\n\nProvide a comprehensive answer:")
        ])
        
        # Generate response
        messages = prompt.format_messages()
        response = components['llm'].invoke(messages)
        
        return response.content
        
    except Exception as e:
        return f"Error processing query: {e}\n\nPlease try again or check your Ollama connection."

def get_database_stats(vector_store):
    """Get statistics about the vector database."""
    try:
        collection = vector_store.get()
        documents = collection.get('documents', [])
        metadatas = collection.get('metadatas', [])
        
        stats = {
            'total_docs': len(documents),
            'sources': set(),
            'content_types': {},
            'avg_length': 0
        }
        
        total_length = 0
        for doc, metadata in zip(documents, metadatas):
            if metadata:
                source = metadata.get('source', 'unknown')
                if source != 'unknown':
                    stats['sources'].add(source)
                
                content_type = metadata.get('content_type', 'general')
                stats['content_types'][content_type] = stats['content_types'].get(content_type, 0) + 1
            
            if doc:
                total_length += len(doc)
        
        if documents:
            stats['avg_length'] = total_length // len(documents)
        
        stats['sources'] = len(stats['sources'])
        
        return stats
    except Exception as e:
        return {'error': str(e)}

# Initialize components
if "components" not in st.session_state:
    st.session_state.components = init_technical_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Main UI
st.title("🤖 Itential Technical Documentation Assistant")
st.caption("Enhanced with technical optimization pipeline")

# Sidebar with database info and controls
with st.sidebar:
    st.header("📊 Database Info")
    
    components = st.session_state.components
    st.write(f"**Database:** {components['db_description']}")
    st.write(f"**Path:** `{components['db_path']}`")
    
    # Database statistics
    with st.spinner("Loading stats..."):
        stats = get_database_stats(components['vector_store'])
    
    if 'error' not in stats:
        st.metric("Total Documents", f"{stats['total_docs']:,}")
        st.metric("Unique Sources", stats['sources'])
        st.metric("Avg Document Length", f"{stats['avg_length']:,} chars")
        
        if stats['content_types']:
            st.write("**Content Types:**")
            for content_type, count in sorted(stats['content_types'].items()):
                st.write(f"- {content_type}: {count}")
    else:
        st.error(f"Stats error: {stats['error']}")
    
    st.divider()
    
    # Model settings
    st.header("⚙️ Settings")
    search_k = st.slider("Search Results", 3, 15, 5)
    
    if st.button("🔄 Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    if st.button("🔄 Reload Database"):
        st.cache_resource.clear()
        st.rerun()

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            st.markdown(message["content"])
        else:
            st.write(message["content"])

if prompt := st.chat_input("Ask about Itential technical documentation..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching technical documentation..."):
            start_time = time.time()
            answer = enhanced_query(st.session_state.components, prompt, k=search_k)
            response_time = time.time() - start_time
            
            st.markdown(answer)
            st.caption(f"Response time: {response_time:.2f}s")
            
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Quick action buttons
st.subheader("🚀 Quick Technical Queries")

col1, col2, col3 = st.columns(3)

with col1:
    st.write("**Version Requirements**")
    if st.button("🐍 Python versions for IAP"):
        query = "What Python versions are required for different versions of IAP?"
        st.session_state.messages.append({"role": "user", "content": query})
        answer = enhanced_query(st.session_state.components, query, k=search_k)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
    
    if st.button("🟢 Node.js requirements"):
        query = "What Node.js versions are required for IAP and IAG?"
        st.session_state.messages.append({"role": "user", "content": query})
        answer = enhanced_query(st.session_state.components, query, k=search_k)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

with col2:
    st.write("**System Requirements**")
    if st.button("💾 Database requirements"):
        query = "What are the database requirements for Itential products?"
        st.session_state.messages.append({"role": "user", "content": query})
        answer = enhanced_query(st.session_state.components, query, k=search_k)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
    
    if st.button("🖥️ System specs"):
        query = "What are the minimum system requirements for IAP installation?"
        st.session_state.messages.append({"role": "user", "content": query})
        answer = enhanced_query(st.session_state.components, query, k=search_k)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

with col3:
    st.write("**Installation Help**")
    if st.button("📦 Installation guide"):
        query = "How do I install IAP? What are the installation steps?"
        st.session_state.messages.append({"role": "user", "content": query})
        answer = enhanced_query(st.session_state.components, query, k=search_k)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()
    
    if st.button("🔧 Configuration help"):
        query = "How do I configure IAP after installation?"
        st.session_state.messages.append({"role": "user", "content": query})
        answer = enhanced_query(st.session_state.components, query, k=search_k)
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# Footer
st.divider()
st.caption("💡 **Tip:** Ask specific questions about versions, requirements, installation, or configuration for best results.")

# Example queries section
with st.expander("📝 Example Queries"):
    st.write("""
    **Version-specific queries:**
    - "What Python version do I need for IAP 2023.2?"
    - "Node.js requirements for IAG 2023.1"
    - "What versions of IAP are available?"
    
    **System requirements:**
    - "What are the minimum system requirements for IAP?"
    - "Database requirements for Itential products"
    - "Memory and CPU requirements for IAP"
    
    **Installation and configuration:**
    - "How do I install IAP on Ubuntu?"
    - "MongoDB configuration for IAP"
    - "Redis setup for Itential"
    - "Docker installation for IAP"
    
    **Troubleshooting:**
    - "Common IAP installation errors"
    - "How to troubleshoot database connection issues"
    - "Performance optimization for IAP"
    """)