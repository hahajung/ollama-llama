# simple_working_ui.py - Direct database access without complex chatbot

import streamlit as st
import os
from pathlib import Path

# Simple direct imports
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

st.set_page_config(page_title="Simple Itential Assistant", layout="wide")

@st.cache_resource
def init_simple_chatbot():
    """Direct chatbot initialization without complex classes."""
    
    # Clear any cached resources first
    st.cache_resource.clear()
    
    st.write("🔍 **Initializing Simple Chatbot...**")
    
    # Direct database path
    db_path = "./technical_optimized_chroma_db"
    
    st.write(f"📂 Database path: `{db_path}`")
    st.write(f"📂 Exists: {os.path.exists(db_path)}")
    st.write(f"📂 Is directory: {os.path.isdir(db_path)}")
    
    if not os.path.exists(db_path):
        st.error("❌ Database not found!")
        st.write("Available directories:")
        for item in os.listdir("."):
            if os.path.isdir(item):
                st.write(f"  📁 {item}")
        st.stop()
    
    try:
        # Direct initialization
        st.write("🤖 Initializing embeddings...")
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        st.write("📚 Loading vector store...")
        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings
        )
        
        st.write("🧠 Initializing LLM...")
        llm = ChatOllama(model="mistral:7b", temperature=0.1)
        
        st.write("✅ **All components initialized successfully!**")
        
        return {
            'vector_store': vector_store,
            'llm': llm,
            'embeddings': embeddings
        }
        
    except Exception as e:
        st.error(f"❌ Initialization failed: {e}")
        st.write("**Troubleshooting:**")
        st.write("1. Is Ollama running? `ollama serve`")
        st.write("2. Are models available? `ollama pull nomic-embed-text && ollama pull mistral:7b`")
        st.stop()

def simple_query(components, question, k=5):
    """Simple query function."""
    try:
        # Search documents
        docs = components['vector_store'].similarity_search(question, k=k)
        
        if not docs:
            return "No relevant documents found."
        
        # Create context
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant for Itential documentation. Answer based on the provided context."),
            ("human", f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:")
        ])
        
        # Generate response
        messages = prompt.format_messages()
        response = components['llm'].invoke(messages)
        
        return response.content
        
    except Exception as e:
        return f"Error: {e}"

# Initialize components
if "components" not in st.session_state:
    st.session_state.components = init_simple_chatbot()

if "messages" not in st.session_state:
    st.session_state.messages = []

# UI
st.title("🤖 Simple Itential Assistant")
st.write("**Direct database access - no complex chatbot classes**")

# Show database info
try:
    collection = st.session_state.components['vector_store'].get()
    doc_count = len(collection.get('documents', []))
    st.success(f"✅ Connected to database with {doc_count:,} documents")
except Exception as e:
    st.warning(f"⚠️ Database info error: {e}")

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask about Itential documentation..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching..."):
            answer = simple_query(st.session_state.components, prompt)
            st.write(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

# Test queries
st.sidebar.header("🧪 Test Queries")
test_queries = [
    "What versions of IAP are available?",
    "Python requirements for IAP 2023.1",
    "Node.js version for IAP 2023.2",
    "MongoDB requirements"
]

for query in test_queries:
    if st.sidebar.button(query):
        st.session_state.messages.append({"role": "user", "content": query})
        answer = simple_query(st.session_state.components, query)
        st.session_state.messages.append({"role": "assistant", "content": answer})