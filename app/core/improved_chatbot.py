#!/usr/bin/env python3
"""
Complete Speed-Optimized Chatbot for Itential Documentation
Optimized for fast local performance with Ollama models.
Includes streaming, caching, error handling, and performance monitoring.
"""

import os
import re
import json
import time
import asyncio
from typing import List, Dict, Optional, Any, Tuple
from collections import defaultdict, deque
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
                 qa_db_path: str = "./technical_optimized_chroma_db",
                 fallback_db_path: str = "./qa_enhanced_chroma_db",
                 model: str = "mistral:7b",  # Faster than llama3.1:8b
                 embedding_model: str = "nomic-embed-text",
                 enable_streaming: bool = True,
                 cache_size: int = 1000):
        """
        Initialize speed-optimized chatbot.
        
        Args:
            qa_db_path: Primary vector database path
            fallback_db_path: Fallback database path
            model: Ollama model for text generation
            embedding_model: Embedding model name
            enable_streaming: Enable streaming responses
            cache_size: Maximum cache size
        """
        self.qa_db_path = qa_db_path
        self.fallback_db_path = fallback_db_path
        self.model_name = model
        self.embedding_model = embedding_model
        self.enable_streaming = enable_streaming
        self.cache_size = cache_size
        
        # Performance tracking
        self.response_times = deque(maxlen=1000)
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Multi-level caching
        self.response_cache = {}  # Full response cache
        self.search_cache = {}    # Search results cache
        self.embedding_cache = {} # Embedding cache
        
        # System prompt optimized for speed and accuracy
        self.system_prompt = """You are a fast, accurate assistant for Itential documentation.
Provide direct, concise answers based strictly on the context provided.
Focus on technical details like versions, requirements, and configurations.
If information is not in the context, say so briefly.
Keep responses under 200 words unless more detail is specifically requested."""

        # Initialize components
        self._initialize_components()
        
        # Verify setup
        self._verify_and_benchmark()

    def _initialize_components(self):
        """Initialize all chatbot components with error handling."""
        try:
            logger.info("[INIT] Initializing speed-optimized chatbot...")
            
            # Initialize embeddings
            self.embeddings = OllamaEmbeddings(
                model=self.embedding_model,
                show_progress=False  # Reduce output for speed
            )
            
            # Try primary database first
            db_path = self._find_database()
            logger.info(f"[INIT] Using database: {db_path}")
            
            # Initialize vector store
            self.vector_store = Chroma(
                persist_directory=db_path,
                embedding_function=self.embeddings
            )
            
            # Initialize LLM with speed optimizations
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=0.1,  # Low temperature for consistency
                num_ctx=4096,     # Moderate context window
                num_predict=512,  # Limit prediction length
                num_thread=4      # Use multiple threads
            )
            
            logger.info("[INIT] All components initialized successfully")
            
        except Exception as e:
            logger.error(f"[INIT] Initialization failed: {e}")
            raise

    def _find_database(self) -> str:
        """Find available database with fallback options."""
        # Check paths relative to current directory and app directory
        search_paths = [
            self.qa_db_path,
            self.fallback_db_path,
            f"../{self.qa_db_path}",
            f"../{self.fallback_db_path}",
            "./super_enhanced_chroma_db",
            "../super_enhanced_chroma_db"
        ]
        
        for path in search_paths:
            if Path(path).exists() and Path(path).is_dir():
                logger.info(f"[DB] Found database at: {path}")
                return path
        
        # If no database found, create a fallback message
        logger.warning("[DB] No existing database found")
        raise FileNotFoundError(f"No vector database found in any of: {search_paths}")

    def _get_query_hash(self, query: str, k: int = 5) -> str:
        """Generate cache key for query."""
        cache_string = f"{query.lower().strip()}_{k}_{self.model_name}"
        return hashlib.md5(cache_string.encode()).hexdigest()

    def fast_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Ultra-fast vector search with caching."""
        # Check search cache
        query_hash = self._get_query_hash(query, k)
        if query_hash in self.search_cache:
            return self.search_cache[query_hash]
        
        try:
            # Perform similarity search
            docs = self.vector_store.similarity_search_with_score(query, k=k)
            
            # Format results
            results = []
            for doc, score in docs:
                results.append({
                    'content': doc.page_content,
                    'source': doc.metadata.get('source', ''),
                    'title': doc.metadata.get('title', ''),
                    'score': score
                })
            
            # Cache results with LRU eviction
            self._update_search_cache(query_hash, results)
            
            return results
            
        except Exception as e:
            logger.error(f"[SEARCH] Search failed: {e}")
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
        return min(1.0, max(0.0, 1.0 - avg_score / 2.0))  # Convert distance to confidence

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

    def stream_response(self, query: str, k: int = 5):
        """Stream response in real-time for UI integration."""
        try:
            # Get search results first
            context_results = self.fast_search(query, k)
            
            if not context_results:
                yield "No relevant information found."
                return
            
            # Format context
            formatted_context = self._fast_format_context(context_results)
            
            # Create prompt
            prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(self.system_prompt),
                HumanMessagePromptTemplate.from_template(
                    "Context: {context}\n\nQuestion: {query}\n\nProvide a direct, concise answer:"
                )
            ])
            
            messages = prompt.format_messages(context=formatted_context, query=query)
            
            # Stream response
            for chunk in self.llm.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            yield f"Error generating response: {e}"

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

    def get_database_info(self) -> Dict[str, Any]:
        """Get information about the vector database."""
        try:
            # Get collection info
            collection = self.vector_store._collection
            count = collection.count()
            
            return {
                'database_path': self.qa_db_path,
                'document_count': count,
                'embedding_model': self.embedding_model,
                'collection_name': collection.name if hasattr(collection, 'name') else 'unknown'
            }
        except Exception as e:
            return {
                'database_path': self.qa_db_path,
                'error': str(e)
            }


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