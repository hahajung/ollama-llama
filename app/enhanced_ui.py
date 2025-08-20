#!/usr/bin/env python3
"""
Comprehensive Enhanced UI
Handles the full breadth of Itential technical documentation including:
- Database Migration Scripts, Adapter Configurations, Automation Studio
- ServiceNow Components, Network Configs, and all technical procedures
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
    page_title="Comprehensive Itential Assistant", 
    layout="wide",
    page_icon="📚"
)

@st.cache_resource
def init_comprehensive_chatbot():
    """Initialize comprehensive chatbot with full technical coverage."""
    
    st.write("**Initializing Comprehensive Itential Assistant...**")
    
    # Check for database
    db_path = "./technical_optimized_chroma_db"
    st.write(f"**Database Path:** `{db_path}`")
    
    if not os.path.exists(db_path):
        st.error("Comprehensive database not found!")
        st.write("**Setup Instructions:**")
        st.write("1. Run: `python robust_technical_embedder.py`")
        st.write("2. Wait for comprehensive system creation")
        st.write("3. Refresh this page")
        st.stop()
    
    try:
        st.write("**Loading Comprehensive System...**")
        
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        llm = ChatOllama(model="mistral:7b", temperature=0.1)
        
        st.write("**Comprehensive System Ready!**")
        
        return {
            'vector_store': vector_store,
            'llm': llm,
            'embeddings': embeddings
        }
        
    except Exception as e:
        st.error(f"Initialization failed: {e}")
        st.write("**Troubleshooting:**")
        st.write("1. Is Ollama running? `ollama serve`")
        st.write("2. Are models available? `ollama pull nomic-embed-text && ollama pull mistral:7b`")
        st.stop()

def comprehensive_query(components, question: str) -> Dict[str, Any]:
    """Process query using comprehensive technical search."""
    try:
        start_time = time.time()
        
        vector_store = components['vector_store']
        
        # Multi-stage comprehensive search
        # Stage 1: Direct search
        docs_direct = vector_store.similarity_search(question, k=6)
        
        # Stage 2: Category-specific search
        docs_category = _search_by_category(vector_store, question)
        
        # Stage 3: Enhanced keyword search
        docs_keywords = _search_with_enhanced_keywords(vector_store, question)
        
        # Combine and prioritize results
        all_docs = docs_direct + docs_category + docs_keywords
        unique_docs = _deduplicate_and_prioritize(all_docs, question)
        
        if not unique_docs:
            return {
                'answer': "I couldn't find information for your query. Please try rephrasing or check if the topic is covered in the documentation.",
                'sources': [],
                'categories_searched': [],
                'response_time': time.time() - start_time
            }
        
        # Prepare comprehensive context
        context_parts = []
        sources = []
        categories_found = set()
        
        for i, doc in enumerate(unique_docs[:8]):  # Top 8 results for comprehensive coverage
            content = doc.page_content
            source = doc.metadata.get('source', 'Unknown')
            content_type = doc.metadata.get('content_type', 'unknown')
            category = doc.metadata.get('category', 'general')
            
            # Track categories found
            categories_found.add(category)
            
            # Prioritize by content type
            if content_type in ['dependency_qa', 'migration_script', 'adapter_config', 'automation_studio']:
                context_parts.insert(0, f"[PRIORITY {i+1}] {content}")
            else:
                context_parts.append(f"[Source {i+1}] {content}")
            
            sources.append(source)
        
        context = '\n\n'.join(context_parts)
        
        # Comprehensive prompt for all content types
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a comprehensive technical expert on all Itential products and documentation.

COMPREHENSIVE KNOWLEDGE AREAS:
- Database Migration Scripts and procedures
- Adapter Configurations (LDAP, Email, Local AAA, etc.)
- Automation Studio features and Search Object Attributes
- ServiceNow Application Components and integrations
- Network configurations and troubleshooting
- Code examples and JSON configurations
- Step-by-step procedures and installation guides
- Version dependencies and requirements

RESPONSE GUIDELINES:
1. Use ONLY the provided context information
2. For migration scripts: Include specific commands and file paths
3. For configurations: Provide exact JSON examples when available
4. For procedures: Include step-by-step instructions
5. For Automation Studio: Explain formData, query tasks, and search attributes
6. For dependencies: Give exact version numbers
7. Be specific about file locations, commands, and parameters

RESPONSE FORMAT:
- Direct, actionable answers first
- Include relevant examples from context
- Specify exact commands, paths, or configurations
- Cite specific procedures when available

Answer comprehensively using the provided context."""),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ])
        
        # Generate comprehensive response
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
            'sources': sources,
            'categories_searched': list(categories_found),
            'context_preview': context[:600] + "..." if len(context) > 600 else context,
            'response_time': time.time() - start_time,
            'num_sources': len(unique_docs),
            'comprehensive_coverage': len(categories_found)
        }
        
    except Exception as e:
        return {
            'answer': f"Error processing comprehensive query: {str(e)}",
            'sources': [],
            'categories_searched': [],
            'response_time': 0,
            'error': str(e)
        }

def _search_by_category(vector_store, question: str) -> List[Any]:
    """Search by specific categories based on question content."""
    question_lower = question.lower()
    category_docs = []
    
    # Migration scripts
    if any(term in question_lower for term in ['migration', 'script', 'database migration', 'properties.json']):
        migration_docs = vector_store.similarity_search(
            question + " migration script migratePropertiesToDatabase", k=3)
        category_docs.extend(migration_docs)
    
    # Adapter configurations
    if any(term in question_lower for term in ['adapter', 'ldap', 'email', 'local aaa', 'configuration']):
        adapter_docs = vector_store.similarity_search(
            question + " adapter configuration JSON properties", k=3)
        category_docs.extend(adapter_docs)
    
    # Automation Studio
    if any(term in question_lower for term in ['automation studio', 'search object', 'formdata', 'workflow']):
        automation_docs = vector_store.similarity_search(
            question + " automation studio formData query task", k=3)
        category_docs.extend(automation_docs)
    
    # ServiceNow
    if any(term in question_lower for term in ['servicenow', 'service now', 'snow', 'application components']):
        servicenow_docs = vector_store.similarity_search(
            question + " servicenow application components integration", k=3)
        category_docs.extend(servicenow_docs)
    
    return category_docs

def _search_with_enhanced_keywords(vector_store, question: str) -> List[Any]:
    """Search with enhanced keywords for comprehensive coverage."""
    question_lower = question.lower()
    enhanced_terms = []
    
    # Add context-specific terms
    if 'example' in question_lower or 'how' in question_lower:
        enhanced_terms.extend(['example', 'configuration', 'setup', 'procedure'])
    
    if 'install' in question_lower or 'setup' in question_lower:
        enhanced_terms.extend(['installation', 'steps', 'procedure', 'command'])
    
    if 'config' in question_lower or 'configure' in question_lower:
        enhanced_terms.extend(['configuration', 'properties', 'JSON', 'settings'])
    
    if any(term in question_lower for term in ['error', 'issue', 'problem', 'troubleshoot']):
        enhanced_terms.extend(['troubleshooting', 'error', 'solution', 'fix'])
    
    if enhanced_terms:
        enhanced_query = question + " " + " ".join(enhanced_terms)
        return vector_store.similarity_search(enhanced_query, k=4)
    
    return []

def _deduplicate_and_prioritize(docs: List[Any], question: str) -> List[Any]:
    """Remove duplicates and prioritize results by relevance to question."""
    seen_content = set()
    unique_docs = []
    question_words = set(question.lower().split())
    
    # Score and sort documents
    scored_docs = []
    for doc in docs:
        content_hash = hash(doc.page_content[:200])
        if content_hash not in seen_content:
            seen_content.add(content_hash)
            
            # Calculate relevance score
            doc_words = set(doc.page_content.lower().split())
            overlap = len(question_words.intersection(doc_words))
            content_type = doc.metadata.get('content_type', '')
            
            # Boost specific content types
            type_boost = 0
            if content_type in ['dependency_qa', 'migration_script', 'adapter_config']:
                type_boost = 5
            elif content_type in ['automation_studio', 'code_example']:
                type_boost = 3
            
            score = overlap + type_boost
            scored_docs.append((score, doc))
    
    # Sort by score and return top documents
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for score, doc in scored_docs]

def main():
    """Main comprehensive interface."""
    
    st.title("Comprehensive Itential Assistant")
    st.markdown("*Complete technical documentation coverage for all Itential products*")
    
    # Initialize the comprehensive system
    if 'comprehensive_chatbot' not in st.session_state:
        with st.spinner("Loading comprehensive system..."):
            st.session_state.comprehensive_chatbot = init_comprehensive_chatbot()
    
    # Comprehensive sidebar with categorized examples
    with st.sidebar:
        st.header("Coverage Areas")
        st.write("[ACTIVE] Dependencies & Versions")
        st.write("[ACTIVE] Migration Scripts")
        st.write("[ACTIVE] Adapter Configurations")
        st.write("[ACTIVE] Automation Studio")
        st.write("[ACTIVE] ServiceNow Integration")
        st.write("[ACTIVE] Code Examples")
        st.write("[ACTIVE] Procedures & Troubleshooting")
        
        st.markdown("---")
        st.header("Example Queries by Category")
        
        # Categorized example queries
        categories = {
            "Dependencies": [
                "What version of Redis is required for IAP 2023.2?",
                "MongoDB version for IAP 2023.2"
            ],
            "Migration Scripts": [
                "How to run database migration script?",
                "migratePropertiesToDatabase.js command example"
            ],
            "Adapter Config": [
                "LDAP adapter configuration example",
                "Email adapter JSON configuration"
            ],
            "Automation Studio": [
                "What are Search Object Attributes?",
                "How to access formData in workflow?"
            ],
            "ServiceNow": [
                "ServiceNow Application Components",
                "ServiceNow integration examples"
            ],
            "Procedures": [
                "Adapter installation steps",
                "Troubleshooting adapter connectivity"
            ]
        }
        
        for category, queries in categories.items():
            with st.expander(f"{category} Examples"):
                for query in queries:
                    if st.button(query, key=f"cat_{hash(query)}"):
                        st.session_state.category_query = query
    
    # Main interface
    col1, col2 = st.columns([3, 1])
    
    with col2:
        show_debug = st.checkbox("Show Debug Info", True)
        show_context = st.checkbox("Show Context", False)
        show_categories = st.checkbox("Show Categories", True)
    
    # Handle category query
    initial_query = ""
    if 'category_query' in st.session_state:
        initial_query = st.session_state.category_query
        del st.session_state.category_query
    
    # Chat input
    if prompt := (initial_query or st.chat_input("Ask about any Itential technical topic...")):
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Process and display response
        with st.chat_message("assistant"):
            with st.spinner("Searching comprehensive documentation..."):
                
                components = st.session_state.comprehensive_chatbot
                result = comprehensive_query(components, prompt)
                
                # Display main answer
                st.write(result['answer'])
                
                # Show comprehensive debug info
                if show_debug:
                    st.markdown("---")
                    st.markdown("**Debug Information:**")
                    
                    debug_cols = st.columns(4)
                    
                    with debug_cols[0]:
                        st.metric("Sources Found", result['num_sources'])
                    
                    with debug_cols[1]:
                        st.metric("Response Time", f"{result['response_time']:.2f}s")
                    
                    with debug_cols[2]:
                        st.metric("Categories Covered", result['comprehensive_coverage'])
                    
                    with debug_cols[3]:
                        coverage_quality = "Excellent" if result['comprehensive_coverage'] >= 3 else "Good" if result['comprehensive_coverage'] >= 2 else "Basic"
                        st.metric("Coverage Quality", coverage_quality)
                
                # Show categories found
                if show_categories and result.get('categories_searched'):
                    st.markdown("**Content Categories Found:**")
                    categories_str = ", ".join(result['categories_searched'])
                    st.info(f"Found content in: {categories_str}")
                
                # Show context preview
                if show_context and 'context_preview' in result:
                    with st.expander("Context Used"):
                        st.text(result['context_preview'])
                
                # Show sources
                sources = result.get('sources', [])
                if sources and sources != ['Unknown']:
                    st.markdown("**Sources:**")
                    unique_sources = list(dict.fromkeys(sources))[:4]
                    for i, source in enumerate(unique_sources, 1):
                        if source.startswith('http'):
                            st.markdown(f"{i}. [{source}]({source})")
                        else:
                            st.markdown(f"{i}. {source}")
                
                # Show error info if present
                if 'error' in result:
                    st.error(f"Error details: {result['error']}")

    # Comprehensive coverage information
    st.markdown("---")
    st.markdown("### Comprehensive Coverage")
    
    st.markdown("""
    This system provides **complete coverage** of Itential technical documentation:
    
    **🔧 Core Infrastructure:**
    - Database Migration Scripts with exact commands
    - Dependency versions and requirements matrices
    - Installation and upgrade procedures
    
    **⚙️ Adapter Configurations:**
    - LDAP Adapter (Active Directory, OpenLDAP)
    - Email Adapter (SMTP, service providers)
    - Local AAA Adapter (MongoDB integration)
    - Network Adapters and troubleshooting
    
    **🤖 Automation Platform:**
    - Automation Studio features and capabilities
    - Search Object Attributes and formData handling
    - Workflow design and execution
    - Form integration and query tasks
    
    **🔗 Integrations:**
    - ServiceNow Application Components
    - API configurations and examples
    - Third-party system connections
    
    **📝 Technical Procedures:**
    - Step-by-step installation guides
    - Configuration procedures
    - Troubleshooting and error resolution
    - Code examples and JSON configurations
    """)

    # System status
    st.markdown("---")
    st.markdown("### System Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Documentation Scope", "Complete")
    
    with col2:
        st.metric("Content Categories", "7+")
    
    with col3:
        st.metric("Search Strategy", "Multi-Stage")
    
    with col4:
        st.metric("Response Quality", "Comprehensive")

if __name__ == "__main__":
    main()