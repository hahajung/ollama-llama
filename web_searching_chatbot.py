#!/usr/bin/env python3
"""
Web-Searching Chatbot
Combines local RAG with real-time web search for the best of both worlds.
"""

import requests
import asyncio
import aiohttp
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from typing import Dict, Any, List, Optional
import sys
from pathlib import Path

# Add core to path
sys.path.insert(0, str(Path("app/core")))

class WebSearchingChatbot:
    """Chatbot that can search the web in real-time like Claude."""
    
    def __init__(self):
        # Initialize local RAG (for speed)
        try:
            from improved_chatbot import ProductionChatbot
            self.local_rag = ProductionChatbot()
            self.has_local_rag = True
            print("✅ Local RAG system loaded")
        except Exception as e:
            print(f"⚠️ Local RAG not available: {e}")
            self.has_local_rag = False
        
        # Initialize web search
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
        # Initialize LLM for processing web content
        try:
            from compatibility_imports import get_ollama_llm
            ChatOllama = get_ollama_llm()
            self.llm = ChatOllama(model="mistral:7b", temperature=0.1)
            print("✅ Local LLM loaded")
        except Exception as e:
            print(f"❌ LLM not available: {e}")
            self.llm = None

    def should_search_web(self, query: str) -> bool:
        """Decide whether to search web or use local RAG."""
        # Web search indicators
        web_indicators = [
            'latest', 'recent', 'current', 'new', 'updated', 'today', 'now',
            '2024', '2025', 'this year', 'this month', 'recently released'
        ]
        
        query_lower = query.lower()
        
        # If query mentions current/latest info, use web
        if any(indicator in query_lower for indicator in web_indicators):
            return True
        
        # If local RAG is not available, use web
        if not self.has_local_rag:
            return True
        
        # Default to local RAG for speed
        return False

    def search_itential_docs(self, query: str, max_pages: int = 3) -> List[Dict[str, str]]:
        """Search docs.itential.com directly for current content."""
        print(f"🌐 Searching docs.itential.com for: {query}")
        
        search_urls = [
            f"https://docs.itential.com/search?q={query.replace(' ', '+')}",
            "https://docs.itential.com/installation/",
            "https://docs.itential.com/dependencies/",
            "https://docs.itential.com/release-notes/"
        ]
        
        # Add specific version URLs if mentioned
        if any(version in query.lower() for version in ['2023.1', '2023.2', '2024.1']):
            for version in ['2023.1', '2023.2', '2024.1']:
                if version in query.lower():
                    search_urls.append(f"https://docs.itential.com/iap/{version}/")
                    break
        
        results = []
        
        for url in search_urls[:max_pages]:
            try:
                print(f"  📄 Fetching: {url}")
                response = self.session.get(url, timeout=10)
                
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Remove unwanted elements
                    for tag in soup(['script', 'style', 'nav', 'footer', 'header']):
                        tag.decompose()
                    
                    # Get title
                    title = soup.title.get_text() if soup.title else "Itential Documentation"
                    
                    # Get main content
                    main_content = (soup.find('main') or 
                                  soup.find('.content') or 
                                  soup.find('article') or
                                  soup.find('body'))
                    
                    if main_content:
                        content = main_content.get_text(separator='\n', strip=True)
                        
                        # Extract relevant sections (look for query terms)
                        relevant_content = self.extract_relevant_content(content, query)
                        
                        if relevant_content:
                            results.append({
                                'title': title.strip(),
                                'url': url,
                                'content': relevant_content,
                                'relevance_score': self.calculate_relevance(relevant_content, query)
                            })
                
            except Exception as e:
                print(f"  ❌ Error fetching {url}: {e}")
                continue
        
        # Sort by relevance
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        print(f"  ✅ Found {len(results)} relevant pages")
        
        return results

    def extract_relevant_content(self, content: str, query: str) -> str:
        """Extract content sections relevant to the query."""
        lines = content.split('\n')
        relevant_lines = []
        query_terms = query.lower().split()
        
        # Look for lines containing query terms
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check if line contains query terms
            matches = sum(1 for term in query_terms if term in line_lower)
            
            if matches >= 2 or any(term in line_lower for term in ['version', 'requirement', 'dependency']):
                # Include context around relevant lines
                start_idx = max(0, i - 2)
                end_idx = min(len(lines), i + 3)
                context = '\n'.join(lines[start_idx:end_idx])
                
                if context not in relevant_lines:
                    relevant_lines.append(context)
        
        # If we found relevant content, return it
        if relevant_lines:
            return '\n\n'.join(relevant_lines[:5])  # Limit to 5 sections
        
        # Fallback: return first part of content
        return content[:2000] if content else ""

    def calculate_relevance(self, content: str, query: str) -> float:
        """Calculate how relevant content is to the query."""
        content_lower = content.lower()
        query_terms = query.lower().split()
        
        score = 0.0
        
        # Count query term matches
        for term in query_terms:
            count = content_lower.count(term)
            score += count * 1.0
        
        # Boost for technical terms
        technical_terms = ['version', 'requirement', 'dependency', 'install', 'python', 'node', 'mongodb']
        for term in technical_terms:
            if term in content_lower:
                score += 2.0
        
        # Boost for version numbers
        version_patterns = [r'[0-9]{4}\.[0-9]', r'[0-9]+\.[0-9]+\.[0-9]+']
        for pattern in version_patterns:
            matches = re.findall(pattern, content)
            score += len(matches) * 1.5
        
        return score

    async def web_search_response(self, query: str) -> Dict[str, Any]:
        """Generate response using real-time web search."""
        start_time = time.time()
        
        # Search for current content
        search_results = self.search_itential_docs(query, max_pages=3)
        
        if not search_results:
            return {
                'answer': f"Could not find current information about '{query}' on docs.itential.com. The site might be temporarily unavailable.",
                'sources': [],
                'confidence': 0.0,
                'method': 'web_search_failed',
                'response_time': time.time() - start_time
            }
        
        # Combine content from multiple pages
        combined_content = ""
        sources = []
        
        for result in search_results[:2]:  # Use top 2 results
            combined_content += f"From {result['title']}:\n{result['content']}\n\n"
            sources.append(result['url'])
        
        # Generate answer using local LLM
        if self.llm:
            prompt = f"""Based on the following current documentation from docs.itential.com, answer this question: {query}

Current Documentation:
{combined_content[:3000]}

Provide a direct, accurate answer based only on the documentation above. Include specific version numbers and requirements when available."""

            try:
                response = await self.llm.ainvoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
            except:
                # Fallback to sync call
                response = self.llm.invoke(prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
        else:
            # Fallback: extract answer from content
            answer = f"Based on current documentation: {combined_content[:500]}..."
        
        return {
            'answer': answer,
            'sources': sources,
            'confidence': 0.9,  # High confidence for current data
            'method': 'web_search',
            'response_time': time.time() - start_time,
            'pages_searched': len(search_results)
        }

    async def local_rag_response(self, query: str) -> Dict[str, Any]:
        """Generate response using local RAG system."""
        if not self.has_local_rag:
            return await self.web_search_response(query)
        
        try:
            if hasattr(self.local_rag, 'generate_fast_response'):
                result = self.local_rag.generate_fast_response(query, k=5)
            else:
                result = self.local_rag.generate_response(query, top_k=5)
            
            result['method'] = 'local_rag'
            return result
            
        except Exception as e:
            print(f"Local RAG failed: {e}, falling back to web search")
            return await self.web_search_response(query)

    async def smart_response(self, query: str) -> Dict[str, Any]:
        """Smart response that chooses the best method."""
        start_time = time.time()
        
        print(f"🤖 Processing query: {query}")
        
        # Decide search method
        use_web = self.should_search_web(query)
        
        if use_web:
            print("🌐 Using web search for current information")
            result = await self.web_search_response(query)
        else:
            print("⚡ Using local RAG for fast response")
            result = await self.local_rag_response(query)
        
        result['total_time'] = time.time() - start_time
        result['routing_decision'] = 'web' if use_web else 'local'
        
        return result

# Streamlit Interface for Web-Searching Chatbot
def create_web_searching_ui():
    """Create Streamlit interface with web search capabilities."""
    import streamlit as st
    
    st.set_page_config(
        page_title="Web-Searching Itential Assistant",
        page_icon="🌐",
        layout="wide"
    )
    
    st.title("🌐 Web-Searching Itential Assistant")
    st.markdown("*Combines local RAG with real-time web search*")
    
    # Initialize chatbot
    if 'web_chatbot' not in st.session_state:
        with st.spinner("Loading web-searching assistant..."):
            st.session_state.web_chatbot = WebSearchingChatbot()
    
    # Query mode selection
    col1, col2 = st.columns(2)
    
    with col1:
        search_mode = st.selectbox(
            "Search Mode",
            ["Auto (Smart Routing)", "Force Web Search", "Force Local RAG"],
            help="Auto chooses best method, Web gets current info, Local is faster"
        )
    
    with col2:
        show_debug = st.checkbox("Show Debug Info", True)
    
    # Chat interface
    if prompt := st.chat_input("Ask about Itential documentation..."):
        
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                
                chatbot = st.session_state.web_chatbot
                
                # Handle different modes
                if search_mode == "Force Web Search":
                    result = asyncio.run(chatbot.web_search_response(prompt))
                elif search_mode == "Force Local RAG":
                    result = asyncio.run(chatbot.local_rag_response(prompt))
                else:
                    result = asyncio.run(chatbot.smart_response(prompt))
                
                # Display answer
                st.write(result['answer'])
                
                # Show debug info
                if show_debug:
                    st.markdown("---")
                    
                    cols = st.columns(4)
                    
                    with cols[0]:
                        method = result.get('method', 'unknown')
                        st.metric("Method", method)
                    
                    with cols[1]:
                        total_time = result.get('total_time', result.get('response_time', 0))
                        st.metric("Response Time", f"{total_time:.2f}s")
                    
                    with cols[2]:
                        confidence = result.get('confidence', 0)
                        st.metric("Confidence", f"{confidence:.1%}")
                    
                    with cols[3]:
                        if 'pages_searched' in result:
                            st.metric("Pages Searched", result['pages_searched'])
                        elif 'sources' in result:
                            st.metric("Sources", len(result['sources']))
                
                # Show sources
                sources = result.get('sources', [])
                if sources:
                    st.markdown("**Sources:**")
                    for i, source in enumerate(sources[:3], 1):
                        st.markdown(f"{i}. [{source}]({source})")

async def main():
    """Demo the web-searching chatbot."""
    chatbot = WebSearchingChatbot()
    
    test_queries = [
        "What Node.js version is required for IAP 2023.2?",
        "What is the latest version of IAP available?",
        "Current system requirements for IAG",
        "Recent updates to Itential Platform"
    ]
    
    print("🌐 Web-Searching Chatbot Demo")
    print("=" * 40)
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        result = await chatbot.smart_response(query)
        
        print(f"📊 Method: {result.get('method', 'unknown')}")
        print(f"⏱️ Time: {result.get('total_time', 0):.2f}s")
        print(f"🎯 Confidence: {result.get('confidence', 0):.1%}")
        print(f"📝 Answer: {result['answer'][:150]}...")
        print("-" * 40)

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Demo in terminal")
    print("2. Launch Streamlit interface")
    
    choice = input("Enter choice (1/2): ").strip()
    
    if choice == "1":
        asyncio.run(main())
    elif choice == "2":
        create_web_searching_ui()
    else:
        print("Invalid choice. Running demo...")
        asyncio.run(main())