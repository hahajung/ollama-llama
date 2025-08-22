#!/usr/bin/env python3
"""
Complete Streamlit Chatbot for Itential Documentation
Works with scraped content from fixed_ultimate_scraper.py
"""

import streamlit as st
import json
from pathlib import Path
import time
from typing import List, Dict, Any, Optional
import hashlib
from datetime import datetime

# Flexible imports for LangChain
try:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.embeddings import OllamaEmbeddings
    from langchain.llms import Ollama
    from langchain.vectorstores import Chroma

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

class DocumentationProcessor:
    """Process scraped documentation into vector database."""
    
    def __init__(self, content_file: str = "./docs_data/content.jsonl"):
        self.content_file = Path(content_file)
        self.vector_db_path = "./docs_vector_db"
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
    def load_documents(self) -> List[Document]:
        """Load scraped documents."""
        if not self.content_file.exists():
            st.error(f"Content file not found: {self.content_file}")
            st.info("Please run the scraper first: python fixed_ultimate_scraper.py")
            return []
        
        documents = []
        doc_count = 0
        
        with open(self.content_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    doc_count += 1
                    
                    # Create document with full content
                    content = f"Title: {data['title']}\n"
                    content += f"URL: {data['url']}\n\n"
                    
                    # Add headings for structure
                    if data.get('headings'):
                        content += "Sections:\n"
                        for h in data['headings']:
                            content += f"  {h['level']}: {h['text']}\n"
                        content += "\n"
                    
                    # Add main text
                    content += data.get('text', '')
                    
                    # Add code blocks if present
                    if data.get('code_blocks'):
                        content += "\n\nCode Examples:\n"
                        for code in data['code_blocks'][:3]:
                            content += f"```\n{code}\n```\n"
                    
                    # Create document
                    doc = Document(
                        page_content=content,
                        metadata={
                            'source': data['url'],
                            'title': data['title'],
                            'has_code': len(data.get('code_blocks', [])) > 0,
                            'heading_count': len(data.get('headings', []))
                        }
                    )
                    documents.append(doc)
                    
                except Exception as e:
                    st.warning(f"Error loading document {doc_count}: {e}")
                    continue
        
        return documents

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        """Create vector store from documents."""
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = []
        for doc in documents:
            doc_chunks = text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
        
        # Create vector store
        vector_store = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=self.vector_db_path
        )
        
        return vector_store

    def get_or_create_vector_store(self) -> Optional[Chroma]:
        """Get existing vector store or create new one."""
        # Check if vector store exists
        if Path(self.vector_db_path).exists():
            try:
                return Chroma(
                    persist_directory=self.vector_db_path,
                    embedding_function=self.embeddings
                )
            except:
                pass
        
        # Create new vector store
        st.info("Creating vector database from scraped content...")
        documents = self.load_documents()
        
        if not documents:
            return None
        
        st.info(f"Loaded {len(documents)} documents. Creating embeddings...")
        
        with st.spinner(f"Processing {len(documents)} documents..."):
            vector_store = self.create_vector_store(documents)
        
        st.success("Vector database created successfully!")
        return vector_store

class ItentialDocsChatbot:
    """Main chatbot for Itential documentation."""
    
    def __init__(self):
        self.processor = DocumentationProcessor()
        self.vector_store = None
        self.qa_chain = None
        self.cache = {}
        
        # Initialize LLM
        self.llm = Ollama(
            model="llama2",
            temperature=0.2,
            top_p=0.9
        )
        
    def initialize(self):
        """Initialize the chatbot."""
        # Get or create vector store
        self.vector_store = self.processor.get_or_create_vector_store()
        
        if not self.vector_store:
            return False
        
        # Create QA chain
        prompt_template = """You are an expert Itential platform engineer helping internal engineers.
Use the following documentation context to answer the question accurately.

Context from documentation:
{context}

Question: {question}

Instructions:
1. Be precise and technical - this is for internal engineers
2. Include specific commands, versions, and configurations when mentioned
3. If the answer isn't in the context, say so clearly
4. Reference the source URL when possible

Answer:"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}
            ),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return True
    
    def search(self, query: str) -> Dict[str, Any]:
        """Search documentation."""
        # Check cache
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['timestamp'] < 3600:  # 1 hour cache
                return {**cached['result'], 'cached': True}
        
        # Perform search
        start_time = time.time()
        
        try:
            result = self.qa_chain({"query": query})
            
            # Process sources
            sources = []
            seen_urls = set()
            
            if 'source_documents' in result:
                for doc in result['source_documents']:
                    url = doc.metadata.get('source', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        sources.append({
                            'url': url,
                            'title': doc.metadata.get('title', 'Unknown'),
                            'preview': doc.page_content[:200] + '...'
                        })
            
            response = {
                'answer': result.get('result', 'No answer found'),
                'sources': sources[:3],
                'response_time': time.time() - start_time,
                'cached': False
            }
            
            # Cache result
            self.cache[cache_key] = {
                'result': response,
                'timestamp': time.time()
            }
            
            return response
            
        except Exception as e:
            return {
                'answer': f"Error: {str(e)}",
                'sources': [],
                'response_time': time.time() - start_time,
                'error': True,
                'cached': False
            }

def main():
    """Main Streamlit application."""
    st.set_page_config(
        page_title="Itential Docs Assistant",
        layout="wide"
    )
    
    st.title("Itential Documentation Assistant")
    st.markdown("*Internal tool for searching scraped documentation*")
    
    # Initialize chatbot
    if 'chatbot' not in st.session_state:
        with st.spinner("Initializing chatbot..."):
            chatbot = ItentialDocsChatbot()
            if chatbot.initialize():
                st.session_state.chatbot = chatbot
                st.success("Chatbot ready!")
            else:
                st.error("Failed to initialize. Please check if content.jsonl exists.")
                st.stop()
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    # Sidebar
    with st.sidebar:
        st.header("Information")
        
        # Check scraped data
        content_file = Path("./docs_data/content.jsonl")
        if content_file.exists():
            # Count documents - FIX: Add UTF-8 encoding
            doc_count = sum(1 for _ in open(content_file, encoding='utf-8'))
            st.success(f"Found {doc_count} documents available")
        else:
            st.error("No scraped data found")
            st.info("Run: python fixed_ultimate_scraper.py")
        
        # Quick searches
        st.header("Quick Searches")
        quick_searches = [
            "How to install Platform 6?",
            "Python requirements for IAP",
            "MongoDB configuration",
            "Troubleshooting guide",
            "CLI documentation",
            "Event deduplication"
        ]
        
        for search in quick_searches:
            if st.button(search, key=f"quick_{search}"):
                st.session_state.pending_search = search
        
        # Clear chat
        if st.button("Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Chat interface
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show sources if available
            if "sources" in message and message["sources"]:
                with st.expander("Sources"):
                    for source in message["sources"]:
                        st.markdown(f"**{source['title']}**")
                        st.markdown(f"Link: {source['url']}")
                        st.text(source['preview'])
            
            # Show metrics
            if "metrics" in message:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Response Time", f"{message['metrics']['time']:.2f}s")
                with col2:
                    st.metric("Cached", "Yes" if message['metrics']['cached'] else "No")
    
    # Handle pending search from sidebar
    if 'pending_search' in st.session_state:
        query = st.session_state.pending_search
        del st.session_state.pending_search
    else:
        query = st.chat_input("Ask about Itential documentation...")
    
    if query:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(query)
        
        # Get response
        with st.chat_message("assistant"):
            with st.spinner("Searching documentation..."):
                response = st.session_state.chatbot.search(query)
                
                # Display answer
                st.write(response['answer'])
                
                # Display sources
                if response.get('sources'):
                    with st.expander("Sources"):
                        for source in response['sources']:
                            st.markdown(f"**{source['title']}**")
                            st.markdown(f"Link: {source['url']}")
                            st.text(source['preview'])
                
                # Display metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Response Time", f"{response['response_time']:.2f}s")
                with col2:
                    st.metric("Cached", "Yes" if response.get('cached') else "No")
                
                # Add to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['answer'],
                    "sources": response.get('sources', []),
                    "metrics": {
                        'time': response['response_time'],
                        'cached': response.get('cached', False)
                    }
                })

if __name__ == "__main__":
    main()