#!/usr/bin/env python3
"""
Speed-Optimized Chatbot for Itential Documentation
Optimized for fast local performance with Ollama models.
"""

import os
import re
import json
import time
import asyncio
from typing import List, Dict, Optional, Any
from collections import defaultdict
import logging
from pathlib import Path
from functools import lru_cache
import hashlib

# Universal imports with backwards compatibility
try:
    from compatibility_imports import (
        get_ollama_llm, get_ollama_embeddings, 
        get_chroma, get_documents, get_prompts
    )
    ChatOllama = get_ollama_llm()
    OllamaEmbeddings = get_ollama_embeddings()
    Chroma = get_chroma()
    Document = get_documents()
    ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate = get_prompts()
except ImportError:
    # Fallback imports
    try:
        from langchain_community.chat_models import ChatOllama
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain.chat_models import ChatOllama
        from langchain.embeddings import OllamaEmbeddings
        from langchain.vectorstores import Chroma
    
    try:
        from langchain_core.documents import Document
        from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
    except ImportError:
        try:
            from langchain.docstore.document import Document
            from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
        except ImportError:
            from langchain.schema import Document
            from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SpeedOptimizedChatbot:
    """
    High-performance chatbot optimized for fast local responses with Ollama.
    Features: Fast models, aggressive caching, streaming responses, optimized search.
    """
    
    def __init__(self, 
                 qa_db_path: str = "./qa_enhanced_chroma_db",
                 fallback_db_path: str = "./super_enhanced_chroma_db",
                 model: str = "mistral:7b",  # Faster than llama3.1:8b
                 embedding_model: str = "nomic-embed-text",
                 enable_streaming: bool = True,
                 cache_size: int = 1000):
        """
        Initialize speed-optimized chatbot.
        
        Args:
            qa_db_path: Path to Q&A optimized vector database
            fallback_db_path: Path to fallback database
            model: Fast Ollama model (mistral:7b, phi3:mini)
            embedding_model: Embedding model
            enable_streaming: Enable response streaming for perceived speed
            cache_size: Size of response cache
        """
        
        logger.info("[INIT] Initializing Speed-Optimized Chatbot...")
        start_time = time.time()
        
        # Configuration
        self.qa_db_path = qa_db_path
        self.fallback_db_path = fallback_db_path
        self.model_name = model
        self.embedding_model = embedding_model
        self.enable_streaming = enable_streaming
        self.cache_size = cache_size
        
        # Performance tracking
        self.response_times = []
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize components with speed optimizations
        self._initialize_fast_components()
        
        # Setup optimized caching
        self._setup_caching()
        
        # Verify and benchmark
        self._verify_and_benchmark()
        
        init_time = time.time() - start_time
        logger.info(f"[SUCCESS] Speed-Optimized Chatbot ready in {init_time:.2f}s")
    
    def _initialize_fast_components(self):
        """Initialize all components with speed optimizations."""
        
        # Fast embeddings
        try:
            self.embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                # Remove deprecated parameters that cause validation errors
                # request_timeout=30.0,  # Not supported in newer versions
                # num_thread=os.cpu_count()  # Not supported in newer versions
            )
            logger.info(f"[OK] Fast embeddings initialized: {self.embedding_model}")
        except Exception as e:
            logger.error(f"[ERROR] Embeddings initialization failed: {e}")
            raise
        
        # Load vector stores with preference for Q&A database
        self._initialize_vector_stores()
        
        # Fast LLM with optimized settings
        try:
            self.llm = ChatOllama(
                model=self.model_name,
                # Speed optimizations
                temperature=0.1,        # Lower temperature for consistency
                top_k=20,              # Reduced from 40 for speed
                top_p=0.8,             # Reduced for faster generation
                num_ctx=4096,          # Smaller context for speed
                num_predict=512,       # Limit response length for speed
                repeat_penalty=1.1,
                # Remove deprecated parameters
                # num_thread=os.cpu_count(),  # Not supported in newer versions
                # use_mmap=True,              # Not supported in newer versions
                # use_mlock=True,             # Not supported in newer versions
                # Request timeout - use different parameter name
                timeout=60.0  # Use timeout instead of request_timeout
            )
            logger.info(f"[OK] Fast LLM initialized: {self.model_name}")
        except Exception as e:
            logger.error(f"[ERROR] LLM initialization failed: {e}")
            raise
        
        # Speed-optimized system prompt
        self.system_prompt = self._get_fast_system_prompt()
    
    def _initialize_vector_stores(self):
        """Initialize vector stores with preference for Q&A optimized database."""
        self.vector_stores = []
        self.primary_store = None
        
        # Try Q&A optimized database first, then fallbacks
        database_options = [
            (self.qa_db_path, "qa_optimized"),
            (self.fallback_db_path, "fallback"),
            ("./super_enhanced_chroma_db", "super_enhanced"),
            ("./enhanced_chroma_db", "enhanced"),
            ("./chroma_db", "original")
        ]
        
        for db_path, db_type in database_options:
            if os.path.exists(db_path):
                try:
                    store = Chroma(
                        persist_directory=db_path,
                        embedding_function=self.embeddings
                    )
                    
                    # Quick test
                    test_results = store.similarity_search("test", k=1)
                    
                    self.vector_stores.append((db_type, store))
                    if not self.primary_store:
                        self.primary_store = store
                    
                    logger.info(f"[OK] {db_type.replace('_', ' ').title()} database loaded: {db_path}")
                    break  # Use first working database for speed
                        
                except Exception as e:
                    logger.warning(f"[WARNING] Failed to load {db_type} database: {e}")
                    continue
        
        if not self.vector_stores:
            raise RuntimeError("[ERROR] No vector databases found!")
        
        logger.info(f"[INFO] Using {self.vector_stores[0][0]} database for optimal speed")
    
    def _get_fast_system_prompt(self) -> str:
        """Optimized system prompt for fast, accurate responses."""
        return """You are a fast and accurate Itential documentation assistant. Provide direct, concise answers based on the context.

SPEED RULES:
1. Start with the direct answer immediately
2. Be concise but complete
3. Include specific details when available (versions, steps, examples)
4. Skip unnecessary explanations
5. Use bullet points or numbered lists for clarity

ACCURACY RULES:
1. ONLY use information from the provided context
2. Quote exact versions, requirements, or specifications
3. If context is incomplete, state what's missing
4. Cite source URLs when relevant

RESPONSE FORMAT:
- Direct answer first
- Key details in bullets/numbers
- Code examples in code blocks
- Source citation at end

Keep responses focused and actionable for Itential users."""

    def _setup_caching(self):
        """Setup aggressive caching for maximum speed."""
        
        # Response cache with LRU eviction
        self.response_cache = {}
        self.cache_order = []
        
        # Embedding cache for common queries
        self.embedding_cache = {}
        
        # Search result cache
        self.search_cache = {}
        
        logger.info(f"[CACHE] Caching system initialized (size: {self.cache_size})")

    @lru_cache(maxsize=1000)
    def _cached_embedding_search(self, query_hash: str, k: int) -> str:
        """Cached embedding search for repeated queries."""
        # This is a placeholder - actual implementation would use the hash
        # to lookup pre-computed results
        return f"cached_search_{query_hash}_{k}"

    def _get_query_hash(self, query: str, k: int) -> str:
        """Generate hash for query caching."""
        return hashlib.md5(f"{query.lower().strip()}_{k}".encode()).hexdigest()[:16]

    def fast_search(self, query: str, k: int = 5) -> List[Dict]:
        """Ultra-fast search with aggressive caching and optimizations."""
        start_time = time.time()
        
        # Check cache first
        query_hash = self._get_query_hash(query, k)
        if query_hash in self.search_cache:
            self.cache_hits += 1
            search_time = time.time() - start_time
            logger.debug(f"[CACHE] Cache hit for query in {search_time:.3f}s")
            return self.search_cache[query_hash]
        
        self.cache_misses += 1
        
        try:
            # Fast vector search with reduced k for speed
            search_k = min(k * 2, 10)  # Limit search scope
            results = self.primary_store.similarity_search_with_score(query, k=search_k)
            
            # Fast result processing
            processed_results = []
            for doc, distance in results[:k]:  # Only process what we need
                # Quick score calculation
                score = max(0, (2.0 - distance)) * 10
                
                processed_results.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', ''),
                    'title': doc.metadata.get('title', ''),
                    'score': score,
                    'metadata': doc.metadata,
                    'method': 'fast_vector'
                })
            
            # Cache results
            self._update_search_cache(query_hash, processed_results)
            
            search_time = time.time() - start_time
            logger.debug(f"[SEARCH] Fast search completed in {search_time:.3f}s")
            
            return processed_results
            
        except Exception as e:
            logger.error(f"[ERROR] Fast search failed: {e}")
            return []

    def _update_search_cache(self, query_hash: str, results: List[Dict]):
        """Update search cache with LRU eviction."""
        # Simple LRU cache management
        if len(self.search_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_hash = list(self.search_cache.keys())[0]
            del self.search_cache[oldest_hash]
        
        self.search_cache[query_hash] = results

    def generate_fast_response(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Generate ultra-fast response with all optimizations enabled."""
        start_time = time.time()
        
        try:
            # Input validation
            if not query or not isinstance(query, str):
                return {
                    'answer': "Please provide a valid question about Itential documentation.",
                    'sources': [],
                    'confidence': 0.0,
                    'response_time': time.time() - start_time
                }
            
            # Check response cache
            cache_key = self._get_query_hash(query, k)
            if cache_key in self.response_cache:
                self.cache_hits += 1
                cached_response = self.response_cache[cache_key].copy()
                cached_response['response_time'] = time.time() - start_time
                cached_response['cached'] = True
                logger.info(f"[SPEED] Cached response in {cached_response['response_time']:.3f}s")
                return cached_response
            
            self.cache_misses += 1
            logger.info(f"[PROCESSING] Processing: {query}")
            
            # Fast search
            search_start = time.time()
            context_results = self.fast_search(query, k)
            search_time = time.time() - search_start
            
            if not context_results:
                return {
                    'answer': f"No relevant information found for: '{query}'. Try rephrasing with more specific terms.",
                    'sources': [],
                    'confidence': 0.0,
                    'response_time': time.time() - start_time,
                    'search_time': search_time
                }
            
            # Fast confidence calculation
            confidence = self._fast_confidence_calc(context_results)
            
            # Optimized context formatting
            formatted_context = self._fast_format_context(context_results)
            sources = [r.get('source', '') for r in context_results if r.get('source')][:3]
            
            # Fast prompt creation
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                HumanMessagePromptTemplate.from_template(
                    "Context: {context}\n\nQuestion: {query}\n\nProvide a direct, concise answer:"
                )
            ])
            
            # Generate response
            llm_start = time.time()
            messages = prompt.format_messages(context=formatted_context, query=query)
            
            if self.enable_streaming:
                # Streaming response for perceived speed
                response_chunks = []
                for chunk in self.llm.stream(messages):
                    if hasattr(chunk, 'content'):
                        response_chunks.append(chunk.content)
                
                response_text = ''.join(response_chunks)
            else:
                # Standard response
                llm_response = self.llm.invoke(messages)
                response_text = llm_response.content
            
            llm_time = time.time() - llm_start
            
            # Fast response formatting
            final_answer = self._fast_format_response(response_text, confidence)
            
            # Prepare response data
            response_data = {
                'answer': final_answer,
                'sources': sources,
                'confidence': confidence,
                'response_time': time.time() - start_time,
                'search_time': search_time,
                'llm_time': llm_time,
                'context_count': len(context_results),
                'cached': False
            }
            
            # Cache successful responses
            if confidence > 0.3:
                self._update_response_cache(cache_key, response_data)
            
            # Track performance
            total_time = response_data['response_time']
            self.response_times.append(total_time)
            
            logger.info(f"[SPEED] Response generated in {total_time:.3f}s (search: {search_time:.3f}s, llm: {llm_time:.3f}s)")
            return response_data
            
        except Exception as e:
            logger.error(f"[ERROR] Fast response generation failed: {e}")
            return {
                'answer': f"Error processing request. Please try again.",
                'sources': [],
                'confidence': 0.0,
                'response_time': time.time() - start_time,
                'error': str(e)
            }

    def _fast_confidence_calc(self, results: List[Dict]) -> float:
        """Fast confidence calculation."""
        if not results:
            return 0.0
        
        scores = [r.get('score', 0) for r in results[:3]]
        avg_score = sum(scores) / len(scores) if scores else 0
        return min(1.0, avg_score / 15.0)  # Simplified calculation

    def _fast_format_context(self, results: List[Dict]) -> str:
        """Fast context formatting with minimal processing."""
        if not results:
            return "No context available."
        
        context_parts = []
        for i, result in enumerate(results[:3], 1):  # Limit to top 3 for speed
            content = result.get('content', '')[:800]  # Truncate for speed
            title = result.get('title', 'Untitled')[:60]
            
            context_parts.append(f"Source {i}: {title}\nContent: {content}\n---")
        
        return "\n".join(context_parts)

    def _fast_format_response(self, response: str, confidence: float) -> str:
        """Fast response formatting."""
        try:
            # Minimal formatting for speed
            response = response.strip()
            
            # Add confidence note only for low confidence
            if confidence < 0.4:
                response += "\n\n*Note: Please verify with official documentation.*"
            
            return response
        except Exception:
            return response

    def _update_response_cache(self, cache_key: str, response_data: Dict):
        """Update response cache with LRU eviction."""
        if len(self.response_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = list(self.response_cache.keys())[0]
            del self.response_cache[oldest_key]
        
        # Store without timing data to avoid stale timestamps
        cache_data = response_data.copy()
        cache_data.pop('response_time', None)
        cache_data.pop('search_time', None)
        cache_data.pop('llm_time', None)
        
        self.response_cache[cache_key] = cache_data

    def generate_response(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Backwards compatible method."""
        return self.generate_fast_response(query, top_k)

    def generate_enhanced_response(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Backwards compatible method."""
        return self.generate_fast_response(query, top_k)

    def _verify_and_benchmark(self):
        """Verify setup and run performance benchmark."""
        try:
            logger.info("[TEST] Running performance benchmark...")
            
            # Test queries for benchmarking
            test_queries = [
                "What is IAP?",
                "How to install Itential?",
                "Node.js requirements",
                "User management",
                "API endpoints"
            ]
            
            benchmark_times = []
            for query in test_queries:
                start_time = time.time()
                result = self.generate_fast_response(query, k=3)
                response_time = time.time() - start_time
                benchmark_times.append(response_time)
                
                if result.get('confidence', 0) > 0:
                    logger.info(f"[OK] '{query}' -> {response_time:.3f}s")
                else:
                    logger.warning(f"[WARNING] '{query}' -> {response_time:.3f}s (low confidence)")
            
            avg_time = sum(benchmark_times) / len(benchmark_times)
            logger.info(f"[STATS] Average response time: {avg_time:.3f}s")
            
            # Performance assessment
            if avg_time < 1.0:
                logger.info("[PERFORMANCE] Excellent performance!")
            elif avg_time < 2.0:
                logger.info("[PERFORMANCE] Good performance!")
            elif avg_time < 3.0:
                logger.info("[PERFORMANCE] Acceptable performance")
            else:
                logger.warning("[PERFORMANCE] Consider using faster model (phi3:mini)")
            
        except Exception as e:
            logger.error(f"[ERROR] Benchmark failed: {e}")

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        if self.response_times:
            avg_time = sum(self.response_times) / len(self.response_times)
            min_time = min(self.response_times)
            max_time = max(self.response_times)
        else:
            avg_time = min_time = max_time = 0.0
        
        total_requests = self.cache_hits + self.cache_misses
        cache_hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'model': self.model_name,
            'total_requests': total_requests,
            'cache_hit_rate': f"{cache_hit_rate:.1f}%",
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'avg_response_time': f"{avg_time:.3f}s",
            'min_response_time': f"{min_time:.3f}s",
            'max_response_time': f"{max_time:.3f}s",
            'response_count': len(self.response_times),
            'cache_size': len(self.response_cache),
            'search_cache_size': len(self.search_cache)
        }

    def clear_cache(self):
        """Clear all caches."""
        self.response_cache.clear()
        self.search_cache.clear()
        self.embedding_cache.clear()
        logger.info("[CACHE] All caches cleared")

    def optimize_for_speed(self):
        """Apply additional speed optimizations."""
        logger.info("[OPTIMIZE] Applying additional speed optimizations...")
        
        # Reduce context size for faster processing
        if hasattr(self.llm, 'num_ctx'):
            self.llm.num_ctx = 2048
        
        # Reduce prediction length
        if hasattr(self.llm, 'num_predict'):
            self.llm.num_predict = 256
        
        # Enable aggressive caching
        self.cache_size = min(2000, self.cache_size * 2)
        
        logger.info("[OK] Speed optimizations applied")


# Model-specific optimized chatbots
class MistralChatbot(SpeedOptimizedChatbot):
    """Mistral-optimized chatbot for balanced speed and quality."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('model', 'mistral:7b')
        super().__init__(**kwargs)


class Phi3Chatbot(SpeedOptimizedChatbot):
    """Phi3-optimized chatbot for maximum speed."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('model', 'phi3:mini')
        super().__init__(**kwargs)


class LlamaChatbot(SpeedOptimizedChatbot):
    """Llama-optimized chatbot for best quality."""
    
    def __init__(self, **kwargs):
        kwargs.setdefault('model', 'llama3.1:8b')
        super().__init__(**kwargs)


# Factory function for easy model selection
def create_fast_chatbot(model_preference: str = "balanced", **kwargs) -> SpeedOptimizedChatbot:
    """
    Create optimized chatbot based on performance preference.
    
    Args:
        model_preference: "speed", "balanced", or "quality"
        **kwargs: Additional configuration options
    
    Returns:
        Optimized chatbot instance
    """
    
    if model_preference == "speed":
        return Phi3Chatbot(**kwargs)
    elif model_preference == "balanced":
        return MistralChatbot(**kwargs)
    elif model_preference == "quality":
        return LlamaChatbot(**kwargs)
    else:
        # Default to balanced
        return MistralChatbot(**kwargs)


# Backwards compatibility
class ImprovedChatbot(MistralChatbot):
    """Backwards compatibility with speed optimizations."""
    pass

class ProductionChatbot(MistralChatbot):
    """Production chatbot with speed optimizations."""
    pass


if __name__ == "__main__":
    print("[SPEED] Speed-Optimized Itential Chatbot")
    print("=" * 50)
    
    try:
        # Test different model configurations
        models_to_test = [
            ("speed", "phi3:mini"),
            ("balanced", "mistral:7b"),
            ("quality", "llama3.1:8b")
        ]
        
        print("\n[TEST] Testing different model configurations...")
        
        for preference, model_name in models_to_test:
            try:
                print(f"\n--- Testing {preference.upper()} mode ({model_name}) ---")
                
                # Create chatbot
                chatbot = create_fast_chatbot(preference)
                
                # Test query
                test_query = "What version of Node.js is required for IAP 2023.2?"
                result = chatbot.generate_fast_response(test_query)
                
                # Show results
                print(f"Response time: {result.get('response_time', 0):.3f}s")
                print(f"Confidence: {result.get('confidence', 0):.1%}")
                print(f"Answer: {result['answer'][:100]}...")
                
                # Show performance stats
                stats = chatbot.get_performance_stats()
                print(f"Cache hit rate: {stats['cache_hit_rate']}")
                
            except Exception as e:
                print(f"[ERROR] {preference} mode failed: {e}")
                continue
        
        print("\n[RECOMMENDATIONS] Recommendations:")
        print("   [SPEED] For fastest responses: phi3:mini")
        print("   [BALANCED] For balanced performance: mistral:7b")
        print("   [QUALITY] For best quality: llama3.1:8b")
        
        print("\n[USAGE] Usage:")
        print("   # Create speed-optimized chatbot")
        print("   chatbot = create_fast_chatbot('balanced')")
        print("   response = chatbot.generate_fast_response('your question')")
        
    except Exception as e:
        print(f"[ERROR] Testing failed: {e}")
        print("\nPlease ensure:")
        print("1. Ollama is running: ollama serve")
        print("2. Models are available: ollama pull mistral:7b")
        print("3. Vector database exists")