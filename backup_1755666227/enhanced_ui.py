#!/usr/bin/env python3
"""
Enhanced UI with Hybrid Router - Clean Complete Version
No syntax errors, ready to use.
"""

import streamlit as st
import os
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Imports
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
        st.error("Missing langchain dependencies")
        st.stop()

st.set_page_config(
    page_title="Hybrid Itential Assistant", 
    layout="wide",
    page_icon="🎯"
)

class HybridQueryRouter:
    """Intelligent query router for hybrid storage system."""
    
    def __init__(self, storage_layers: Dict[str, Any]):
        self.storage_layers = storage_layers
        
    def route_query(self, query: str) -> Tuple[str, List[Any], str]:
        """Route query to the most appropriate storage layer."""
        query_lower = query.lower()
        
        # Version-specific routing
        if any(pattern in query_lower for pattern in [
            'what version of', 'version required', 'version needed',
            'is required', 'does need', 'is needed',
            'what mongodb', 'what redis', 'what rabbitmq', 'what python', 'what node'
        ]):
            layer = 'primary_qa'
            reason = "Version-specific query → Q&A Database"
            
        elif any(pattern in query_lower for pattern in [
            'table', 'matrix', 'show me', 'dependency table',
            'requirements table', 'version matrix'
        ]):
            layer = 'structured_data'
            reason = "Table/matrix query → Structured Data"
            
        elif any(pattern in query_lower for pattern in [
            'how to', 'install', 'configure', 'setup', 'deployment',
            'step by step', 'procedure', 'process'
        ]):
            layer = 'complete_pages'
            reason = "Procedural query → Complete Pages"
            
        else:
            layer = 'primary_qa'
            reason = "General query → Q&A Database"
        
        # Search the selected layer
        if layer in self.storage_layers and self.storage_layers[layer]:
            try:
                results = self.storage_layers[layer].similarity_search(query, k=5)
                if results:
                    return layer, results, reason
            except Exception as e:
                st.error(f"Error searching {layer}: {e}")
        
        # Fallback: search all available layers
        all_results = []
        for layer_name, layer_db in self.storage_layers.items():
            if layer_db:
                try:
                    layer_results = layer_db.similarity_search(query, k=2)
                    for result in layer_results:
                        result.metadata['searched_layer'] = layer_name
                    all_results.extend(layer_results)
                except Exception:
                    continue
        
        return 'fallback', all_results, "Fallback → All Layers"

@st.cache_resource
def init_hybrid_chatbot():
    """Initialize hybrid chatbot with multiple storage layers."""
    
    st.write("🎯 **Initializing Hybrid Itential Assistant...**")
    
    # Storage layer paths
    storage_paths = {
        'primary_qa': "./technical_optimized_chroma_db",
        'structured_data': "./technical_structured_db", 
        'complete_pages': "./technical_complete_pages_db"
    }
    
    # Check which databases exist
    st.write("📂 **Checking Storage Layers:**")
    available_layers = {}
    
    for layer_name, path in storage_paths.items():
        exists = os.path.exists(path) and os.path.isdir(path)
        status = "✅" if exists else "❌"
        st.write(f"  {status} {layer_name}: `{path}`")
        
        if exists:
            try:
                embeddings = OllamaEmbeddings(model="nomic-embed-text")
                layer_db = Chroma(
                    persist_directory=path,
                    embedding_function=embeddings
                )
                available_layers[layer_name] = layer_db
                st.write(f"    📊 Loaded successfully")
            except Exception as e:
                st.write(f"    ⚠️ Load error: {e}")
                available_layers[layer_name] = None
        else:
            available_layers[layer_name] = None
    
    if not any(available_layers.values()):
        st.error("❌ No storage layers found!")
        st.write("**Setup Instructions:**")
        st.write("1. Run: `python robust_technical_embedder.py`")
        st.write("2. Wait for hybrid system creation")
        st.write("3. Refresh this page")
        st.stop()
    
    # Initialize other components
    try:
        st.write("🤖 **Initializing Components...**")
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        llm = ChatOllama(model="mistral:7b", temperature=0.1)
        router = HybridQueryRouter(available_layers)
        
        st.write("✅ **Hybrid System Ready!**")
        
        return {
            'storage_layers': available_layers,
            'router': router,
            'llm': llm,
            'embeddings': embeddings
        }
        
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        st.write("**Troubleshooting:**")
        st.write("1. Is Ollama running? `ollama serve`")
        st.write("2. Are models available? `ollama pull nomic-embed-text && ollama pull mistral:7b`")
        st.stop()

def hybrid_query(components, question: str) -> Dict[str, Any]:
    """Process query using hybrid routing system."""
    try:
        start_time = time.time()
        
        # Route the query
        router = components['router']
        layer_used, docs, routing_reason = router.route_query(question)
        
        if not docs:
            return {
                'answer': "I couldn't find relevant information for your query. Please try rephrasing your question.",
                'layer_used': layer_used,
                'routing_reason': routing_reason,
                'sources': [],
                'response_time': time.time() - start_time
            }
        
        # Prepare context from retrieved documents
        context_parts = []
        sources = []
        
        for i, doc in enumerate(docs[:5]):
            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown')
            
            context_parts.append(f"[Source {i+1}] {content}")
            sources.append(source)
        
        context = '\n\n'.join(context_parts)
        
        # Enhanced prompt for technical accuracy
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a technical expert on Itential Automation Platform (IAP) and Itential Automation Gateway (IAG). 

CRITICAL ACCURACY RULES:
1. ONLY use information from the provided context
2. For version-specific queries, be precise about which version you're referring to
3. If a dependency shows "NOT_REQUIRED" or "N/A", clearly state it's not needed
4. Quote exact version numbers from the context
5. If information is missing, say so rather than guessing

RESPONSE FORMAT:
- Give direct, accurate answers first
- Use bullet points for clarity
- Include specific version numbers
- Cite sources when helpful

Focus on being accurate rather than comprehensive."""),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ])
        
        # Generate response
        llm = components['llm']
        formatted_prompt = prompt_template.format(context=context, question=question)
        response = llm.invoke(formatted_prompt)
        
        # Extract response text
        if hasattr(response, 'content'):
            answer = response.content
        else:
            answer = str(response)
        
        return {
            'answer': answer,
            'layer_used': layer_used,
            'routing_reason': routing_reason,
            'sources': sources,
            'context_preview': context[:500] + "..." if len(context) > 500 else context,
            'response_time': time.time() - start_time,
            'num_sources': len(docs)
        }
        
    except Exception as e:
        return {
            'answer': f"Error processing query: {str(e)}",
            'layer_used': 'error',
            'routing_reason': 'Error occurred',
            'sources': [],
            'response_time': 0,
            'error': str(e)
        }

def main():
    """Main Streamlit interface."""
    
    st.title("🎯 Hybrid Itential Assistant")
    st.markdown("*Enhanced with intelligent routing for maximum accuracy*")
    
    # Initialize the hybrid system
    if 'hybrid_chatbot' not in st.session_state:
        with st.spinner("Loading hybrid system..."):
            st.session_state.hybrid_chatbot = init_hybrid_chatbot()
    
    # Sidebar with system info
    with st.sidebar:
        st.header("🔧 System Status")
        
        components = st.session_state.hybrid_chatbot
        storage_layers = components['storage_layers']
        
        for layer_name, layer_db in storage_layers.items():
            status = "🟢" if layer_db else "🔴"
            layer_display = layer_name.replace('_', ' ').title()
            st.write(f"{status} {layer_display}")
        
        st.markdown("---")
        st.header("💡 Query Examples")
        
        example_queries = [
            "What version of RabbitMQ is required for IAP 2023.2?",
            "Is RabbitMQ required for IAP 2023.2?",
            "What Node.js version is required for IAP 2023.2?",
            "What are the system requirements for IAP 2023.2?",
            "Show me the dependency table for IAP versions",
            "How to install IAP 2023.2?"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{hash(query)}"):
                st.session_state.example_query = query
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col2:
        show_debug = st.checkbox("Show Debug Info", True)
        show_context = st.checkbox("Show Context", False)
    
    # Handle example query
    initial_query = ""
    if 'example_query' in st.session_state:
        initial_query = st.session_state.example_query
        del st.session_state.example_query
    
    # Chat input
    if prompt := (initial_query or st.chat_input("Ask about Itential documentation...")):
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process and display response
        with st.chat_message("assistant"):
            with st.spinner("Searching hybrid knowledge base..."):
                
                components = st.session_state.hybrid_chatbot
                result = hybrid_query(components, prompt)
                
                # Display main answer
                st.write(result['answer'])
                
                # Show debug information
                if show_debug:
                    st.markdown("---")
                    st.markdown("**🔍 Debug Information:**")
                    
                    debug_cols = st.columns(4)
                    
                    with debug_cols[0]:
                        st.metric("Layer Used", result['layer_used'])
                    
                    with debug_cols[1]:
                        st.metric("Response Time", f"{result['response_time']:.2f}s")
                    
                    with debug_cols[2]:
                        st.metric("Sources Found", result['num_sources'])
                    
                    with debug_cols[3]:
                        accuracy_score = "High" if result['layer_used'] == 'primary_qa' else "Medium"
                        st.metric("Accuracy", accuracy_score)
                    
                    # Routing information
                    st.info(f"🧭 Routing: {result['routing_reason']}")
                
                # Show context preview
                if show_context and 'context_preview' in result:
                    with st.expander("📄 Context Used"):
                        st.text(result['context_preview'])
                
                # Show sources
                sources = result.get('sources', [])
                if sources and sources != ['Unknown']:
                    st.markdown("**📚 Sources:**")
                    unique_sources = list(dict.fromkeys(sources))[:3]
                    for i, source in enumerate(unique_sources, 1):
                        if source.startswith('http'):
                            st.markdown(f"{i}. [{source}]({source})")
                        else:
                            st.markdown(f"{i}. {source}")
                
                # Show error info if present
                if 'error' in result:
                    st.error(f"Error details: {result['error']}")

    # Instructions
    st.markdown("---")
    st.markdown("### 🎯 How This Works")
    
    st.markdown("""
    This hybrid system uses **intelligent routing** to find the most accurate answers:
    
    - **🎯 Version Queries** → Q&A Database (precise answers)
    - **📊 Table Queries** → Structured Data (exact tables) 
    - **📚 How-To Queries** → Complete Pages (full context)
    - **🔄 Fallback** → Search all layers
    
    **Example queries that work well:**
    - "What version of [dependency] is required for IAP [version]?"
    - "Is [dependency] required for IAP [version]?"
    - "What are the system requirements for IAP [version]?"
    - "Show me the dependency table for IAP versions"
    """)

    # Footer with performance info
    st.markdown("---")
    st.markdown("### 📊 System Performance")
    
    if 'hybrid_chatbot' in st.session_state:
        components = st.session_state.hybrid_chatbot
        storage_layers = components['storage_layers']
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            available_layers = sum(1 for db in storage_layers.values() if db is not None)
            st.metric("Available Layers", f"{available_layers}/3")
        
        with col2:
            primary_status = "✅" if storage_layers.get('primary_qa') else "❌"
            st.metric("Primary DB", primary_status)
        
        with col3:
            hybrid_status = "✅" if available_layers >= 2 else "⚠️"
            st.metric("Hybrid Mode", hybrid_status)

if __name__ == "__main__":
    main()