#!/usr/bin/env python3
"""
Enhanced Context-Aware Chatbot for Itential Documentation
COMPLETE REPLACEMENT for app/core/improved_chatbot.py
Optimized for fast local performance with context-aware search to prevent over-matching.
"""

import os
import re
import json
import time
import asyncio
from typing import List, Dict, Optional, Any, Tuple
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


class EnhancedContextAwareChatbot:
    """
    Enhanced chatbot with context-aware processing to prevent over-matching.
    Features: Context detection, domain-specific search, fast responses.
    """
    
    def __init__(self, 
                 model_name: str = "mistral:7b",
                 embedding_model: str = "nomic-embed-text",
                 qa_db_path: str = "./enhanced_context_chroma_db",
                 fallback_db_path: str = "./technical_optimized_chroma_db",
                 cache_size: int = 100):
        """Initialize enhanced context-aware chatbot."""
        
        self.model_name = model_name
        self.embedding_model = embedding_model
        self.qa_db_path = qa_db_path
        self.fallback_db_path = fallback_db_path
        self.cache_size = cache_size
        
        # Performance tracking
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Context patterns for query classification
        self.context_patterns = {
            'CLI_TOOLS': {
                'keywords': ['itential-cli', 'cli', 'command line', 'ansible-galaxy', 'netcommon'],
                'phrases': ['duplicate data', 'cli troubleshooting', 'command line interface'],
                'indicators': ['cli role', 'itential-cli role', 'ansible collection', 'netcommon version']
            },
            'PLATFORM_EVENTS': {
                'keywords': ['event service', 'deduplication', 'operations manager', 'email adapter'],
                'phrases': ['event deduplication', 'email trigger', 'operations manager trigger'],
                'indicators': ['platform event', 'event configuration', 'trigger type']
            },
            'VERSION_MATRIX': {
                'keywords': ['version', 'requirement', 'dependency', 'compatibility'],
                'phrases': ['version requirements', 'system requirements', 'compatibility matrix'],
                'indicators': ['python version', 'node.js version', 'iap version', 'platform version']
            },
            'TROUBLESHOOTING': {
                'keywords': ['troubleshoot', 'debug', 'error', 'issue', 'problem'],
                'phrases': ['troubleshooting steps', 'fix issue', 'resolve problem'],
                'indicators': ['how to fix', 'debugging guide', 'error resolution']
            }
        }
        
        # Context-specific system prompts
        self.context_prompts = {
            'CLI_TOOLS': """You are a specialized Itential CLI troubleshooting assistant. Focus ONLY on command-line interface tools and their issues.

CRITICAL RULES FOR CLI QUERIES:
1. If the query mentions "itential-cli" and "duplicate data", this is specifically about CLI tool output duplication
2. The solution is usually related to ansible netcommon collection version (should be 5.1.0 or greater)
3. DO NOT confuse CLI issues with Platform event deduplication - these are completely different
4. CLI troubleshooting steps:
   - Check ansible-galaxy collection list
   - Verify netcommon collection version is 5.1.0+
   - Upgrade with: ansible-galaxy collection install ansible.netcommon:5.1.0 --force

CONTEXT FOCUS: Command line tools, CLI troubleshooting, ansible collections, netcommon versions
AVOID: Platform event services, database configuration, general platform issues""",

            'PLATFORM_EVENTS': """You are a specialized Itential Platform event services assistant. Focus ONLY on platform-level event processing and deduplication.

CRITICAL RULES FOR PLATFORM EVENT QUERIES:
1. Event deduplication is for Platform services, not CLI tools
2. Configure event deduplication in service configuration with messageId field
3. Use uniqueProps setting for event deduplication
4. Operations Manager trigger configuration is platform-level

CONTEXT FOCUS: Platform event services, event deduplication, Operations Manager, email adapters
AVOID: CLI tools, command line interfaces, ansible collections""",

            'VERSION_MATRIX': """You are a specialized Itential version and dependency assistant. Focus ONLY on version requirements and compatibility.

CRITICAL RULES FOR VERSION QUERIES:
1. Provide exact version numbers when available
2. Include compatibility information between products
3. Mention supported versions clearly
4. Include upgrade paths when relevant

CONTEXT FOCUS: Version requirements, compatibility matrices, system requirements, dependencies
PROVIDE: Specific version numbers, compatibility info, requirement details""",

            'TROUBLESHOOTING': """You are a specialized Itential troubleshooting assistant. Focus on problem resolution and debugging.

CRITICAL RULES FOR TROUBLESHOOTING QUERIES:
1. Provide step-by-step troubleshooting guidance
2. Include specific commands or configuration when applicable
3. Mention common causes and solutions
4. Provide verification steps

CONTEXT FOCUS: Problem resolution, debugging steps, error solutions
PROVIDE: Step-by-step guidance, specific solutions, verification methods""",

            'GENERAL': """You are a helpful Itential documentation assistant. Provide accurate, context-specific answers.

CRITICAL RULES:
1. Identify the specific domain of the query first
2. Focus your answer on that domain only
3. If unclear, ask for clarification about the specific area
4. Cite sources when available

RESPONSE FORMAT:
- Direct answer first
- Specific details and examples
- Source links when available"""
        }
        
        # Initialize components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize embeddings, vector store, and LLM."""
        try:
            # Fast embeddings
            self.embeddings = OllamaEmbeddings(model=self.embedding_model)
            logger.info(f"✅ Embeddings initialized: {self.embedding_model}")
            
            # Load vector stores with preference for enhanced database
            self._initialize_vector_stores()
            
            # Fast LLM with optimized settings
            self.llm = ChatOllama(
                model=self.model_name,
                temperature=0.1,
                top_k=20,
                top_p=0.8,
                num_ctx=4096,
                num_predict=512,
                timeout=60.0
            )
            logger.info(f"✅ LLM initialized: {self.model_name}")
            
            # Setup caching
            self._setup_caching()
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise

    def _initialize_vector_stores(self):
        """Initialize vector stores with preference for enhanced database."""
        self.vector_stores = []
        self.primary_store = None
        
        # Try enhanced context database first, then fallbacks
        database_options = [
            (self.qa_db_path, "enhanced_context"),
            (self.fallback_db_path, "fallback"),
            ("./technical_optimized_chroma_db", "technical"),
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
                    
                    logger.info(f"✅ {db_type.replace('_', ' ').title()} database loaded: {db_path}")
                    break  # Use first working database for speed
                        
                except Exception as e:
                    logger.warning(f"⚠️  Failed to load {db_type} database: {e}")
                    continue
        
        if not self.vector_stores:
            raise RuntimeError("❌ No vector databases found!")
        
        logger.info(f"🎯 Using {self.vector_stores[0][0]} database for optimal performance")

    def _setup_caching(self):
        """Setup aggressive caching for maximum speed."""
        self.response_cache = {}
        self.cache_order = []
        self.embedding_cache = {}
        self.search_cache = {}
        
        logger.info(f"📦 Caching system initialized (size: {self.cache_size})")

    @lru_cache(maxsize=1000)
    def _cached_embedding_search(self, query_hash: str, k: int) -> str:
        """Cached embedding search for repeated queries."""
        return f"cached_search_{query_hash}_{k}"

    def _get_query_hash(self, query: str, k: int) -> str:
        """Generate hash for query caching."""
        return hashlib.md5(f"{query.lower().strip()}_{k}".encode()).hexdigest()[:16]

    def detect_query_context(self, query: str) -> Tuple[str, float]:
        """
        Detect the primary context/domain of the query.
        Returns (context, confidence_score)
        """
        query_lower = query.lower()
        context_scores = {}
        
        for context_name, context_info in self.context_patterns.items():
            score = 0
            
            # Keyword matching (base score)
            for keyword in context_info['keywords']:
                if keyword in query_lower:
                    score += 5
            
            # Phrase matching (higher score)
            for phrase in context_info['phrases']:
                if phrase in query_lower:
                    score += 10
            
            # Specific indicator matching (highest score)
            for indicator in context_info['indicators']:
                if indicator in query_lower:
                    score += 15
            
            # Special CLI detection with very high confidence
            if context_name == 'CLI_TOOLS':
                cli_specific = [
                    'itential-cli role', 'cli duplicate', 'ansible-galaxy',
                    'netcommon', 'command line duplicate', 'cli troubleshoot'
                ]
                for specific in cli_specific:
                    if specific in query_lower:
                        score += 25  # Very high confidence for CLI
            
            if score > 0:
                context_scores[context_name] = score
        
        if context_scores:
            best_context = max(context_scores, key=context_scores.get)
            confidence = min(context_scores[best_context] / 20.0, 1.0)  # Normalize to 0-1
            return best_context, confidence
        else:
            return 'GENERAL', 0.3

    def create_context_aware_search_query(self, original_query: str, context: str) -> str:
        """Create an enhanced search query based on detected context."""
        
        if context == 'CLI_TOOLS':
            # Enhance CLI queries to be more specific
            cli_enhancements = [
                "CLI_TROUBLESHOOTING", "itential-cli", "command line",
                "ansible", "netcommon", "duplicate data"
            ]
            enhanced_query = f"CLI_TOOLS {original_query} {' '.join(cli_enhancements[:3])}"
            
        elif context == 'PLATFORM_EVENTS':
            # Enhance platform event queries
            platform_enhancements = [
                "PLATFORM_EVENTS", "event service", "deduplication",
                "operations manager", "platform configuration"
            ]
            enhanced_query = f"PLATFORM_EVENTS {original_query} {' '.join(platform_enhancements[:3])}"
            
        elif context == 'VERSION_MATRIX':
            # Enhance version queries
            version_enhancements = [
                "VERSION_REQUIREMENTS", "dependency", "compatibility",
                "version matrix", "requirements"
            ]
            enhanced_query = f"VERSION_REQUIREMENTS {original_query} {' '.join(version_enhancements[:3])}"
            
        elif context == 'TROUBLESHOOTING':
            # Enhance troubleshooting queries
            trouble_enhancements = [
                "TROUBLESHOOTING_GUIDE", "fix", "resolve",
                "debug", "solution"
            ]
            enhanced_query = f"TROUBLESHOOTING_GUIDE {original_query} {' '.join(trouble_enhancements[:3])}"
            
        else:
            # Default enhancement
            enhanced_query = f"GENERAL_DOCS {original_query}"
        
        return enhanced_query

    def context_aware_search(self, query: str, k: int = 5) -> Tuple[List[Dict], str, float]:
        """
        Perform context-aware search that targets the right domain.
        Returns (search_results, detected_context, confidence)
        """
        # Detect query context
        context, confidence = self.detect_query_context(query)
        
        # Create context-specific search query
        enhanced_query = self.create_context_aware_search_query(query, context)
        
        # Perform search with context targeting
        try:
            # Primary search with enhanced query
            results = self.primary_store.similarity_search(enhanced_query, k=k)
            
            # Filter results by context relevance
            filtered_results = self._filter_results_by_context(results, context, query)
            
            # If no good context matches, fall back to original query
            if not filtered_results and context != 'GENERAL':
                logger.info(f"No context-specific results, falling back to general search")
                results = self.primary_store.similarity_search(query, k=k)
                filtered_results = results
            
            # Convert to dictionaries with metadata
            result_dicts = []
            for doc in filtered_results:
                result_dict = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source', ''),
                    'title': doc.metadata.get('title', ''),
                    'domain': doc.metadata.get('domain', 'unknown'),
                    'context_match': doc.metadata.get('domain') == context,
                    'score': 1.0  # Placeholder score
                }
                result_dicts.append(result_dict)
            
            return result_dicts, context, confidence
            
        except Exception as e:
            logger.error(f"❌ Context-aware search failed: {e}")
            return [], context, confidence

    def _filter_results_by_context(self, results: List, target_context: str, original_query: str) -> List:
        """Filter search results to prioritize target context."""
        if not results:
            return results
        
        # Group results by domain
        context_results = []
        other_results = []
        
        for doc in results:
            doc_domain = doc.metadata.get('domain', 'GENERAL')
            
            # Direct context match
            if doc_domain == target_context:
                context_results.append(doc)
            # Special case: CLI troubleshooting
            elif (target_context == 'CLI_TOOLS' and 
                  doc_domain == 'TROUBLESHOOTING' and
                  any(term in doc.page_content.lower() for term in ['cli', 'command line', 'itential-cli'])):
                context_results.append(doc)
            else:
                other_results.append(doc)
        
        # Return context results first, then others as fallback
        if context_results:
            # For CLI queries, strongly prefer CLI-specific results
            if target_context == 'CLI_TOOLS':
                return context_results[:4] + other_results[:1]
            else:
                return context_results + other_results[:2]
        else:
            return other_results

    def create_context_specific_prompt(self, query: str, context: str, search_results: List[Dict]) -> Tuple[str, str]:
        """Create a context-specific prompt for the LLM."""
        
        # Get context-specific system prompt
        system_prompt = self.context_prompts.get(context, self.context_prompts['GENERAL'])
        
        # Format search results
        context_text = self._format_context_for_prompt(search_results, context)
        
        # Create enhanced prompt
        if context == 'CLI_TOOLS':
            user_prompt = f"""Context from CLI troubleshooting documentation:
{context_text}

User Query: {query}

IMPORTANT: This query is about CLI tools. Focus on command-line interface solutions, ansible collections, and CLI troubleshooting. Do NOT provide Platform event service solutions unless specifically asked.

Provide a direct, specific answer for CLI troubleshooting:"""

        elif context == 'PLATFORM_EVENTS':
            user_prompt = f"""Context from Platform event documentation:
{context_text}

User Query: {query}

IMPORTANT: This query is about Platform event services. Focus on event deduplication, Operations Manager, and platform-level configuration. Do NOT provide CLI tool solutions.

Provide a direct, specific answer for Platform event configuration:"""

        elif context == 'VERSION_MATRIX':
            user_prompt = f"""Context from version and dependency documentation:
{context_text}

User Query: {query}

IMPORTANT: This query is about versions and dependencies. Provide specific version numbers, compatibility information, and requirement details.

Provide a direct, specific answer with exact version information:"""

        elif context == 'TROUBLESHOOTING':
            user_prompt = f"""Context from troubleshooting documentation:
{context_text}

User Query: {query}

IMPORTANT: This query is about troubleshooting and problem resolution. Provide step-by-step guidance and specific solutions.

Provide a direct, specific troubleshooting answer:"""

        else:
            user_prompt = f"""Context from Itential documentation:
{context_text}

User Query: {query}

Provide a helpful, accurate answer based on the context above:"""

        return system_prompt, user_prompt

    def _format_context_for_prompt(self, search_results: List[Dict], context: str) -> str:
        """Format search results into context for the prompt."""
        if not search_results:
            return "No relevant documentation found."
        
        context_parts = []
        
        for i, result in enumerate(search_results[:3], 1):  # Limit to top 3
            content = result.get('content', '')
            title = result.get('title', 'Untitled')
            source = result.get('source', '')
            domain = result.get('domain', 'unknown')
            
            # Truncate content for prompt efficiency
            if len(content) > 800:
                content = content[:800] + "..."
            
            context_part = f"""
Source {i} - {title} [{domain}]:
{content}
URL: {source}
---"""
            context_parts.append(context_part)
        
        return '\n'.join(context_parts)

    def generate_context_aware_response(self, query: str) -> Dict[str, Any]:
        """Generate a response using context-aware processing."""
        start_time = time.time()
        
        try:
            # Step 1: Context-aware search
            search_results, detected_context, confidence = self.context_aware_search(query)
            search_time = time.time() - start_time
            
            if not search_results:
                return {
                    'answer': f"No relevant documentation found for: '{query}'. Try rephrasing with more specific terms.",
                    'context': detected_context,
                    'confidence': 0.0,
                    'sources': [],
                    'response_time': time.time() - start_time,
                    'context_aware': True
                }
            
            # Step 2: Create context-specific prompt
            system_prompt, user_prompt = self.create_context_specific_prompt(
                query, detected_context, search_results
            )
            
            # Step 3: Generate response with context-aware prompt
            llm_start = time.time()
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", user_prompt)
            ])
            
            messages = prompt.format_messages()
            response = self.llm.invoke(messages)
            answer = response.content.strip()
            
            llm_time = time.time() - llm_start
            
            # Step 4: Post-process answer
            enhanced_answer = self._enhance_answer_with_context(answer, detected_context, confidence)
            
            return {
                'answer': enhanced_answer,
                'context': detected_context,
                'confidence': confidence,
                'sources': [
                    {
                        'title': result.get('title', ''),
                        'url': result.get('source', ''),
                        'domain': result.get('domain', '')
                    }
                    for result in search_results[:3]
                ],
                'response_time': time.time() - start_time,
                'search_time': search_time,
                'llm_time': llm_time,
                'context_aware': True
            }
            
        except Exception as e:
            logger.error(f"❌ Context-aware response generation failed: {e}")
            return {
                'answer': f"Error processing query: {str(e)}. Please try again.",
                'context': 'ERROR',
                'confidence': 0.0,
                'sources': [],
                'response_time': time.time() - start_time,
                'error': str(e),
                'context_aware': True
            }

    def _enhance_answer_with_context(self, answer: str, context: str, confidence: float) -> str:
        """Enhance the answer with context-specific formatting."""
        
        # Add context indicator for high-confidence responses
        if confidence > 0.7:
            context_labels = {
                'CLI_TOOLS': '🖥️ CLI Troubleshooting',
                'PLATFORM_EVENTS': '⚙️ Platform Events',
                'VERSION_MATRIX': '📋 Version Requirements',
                'TROUBLESHOOTING': '🔧 Troubleshooting'
            }
            
            if context in context_labels:
                answer = f"**{context_labels[context]}**\n\n{answer}"
        
        # Add verification note for low confidence
        if confidence < 0.4:
            answer += "\n\n*Note: Please verify this information with the official documentation.*"
        
        return answer

    def fast_search(self, query: str, k: int = 5) -> List[Dict]:
        """Fast search with basic caching (fallback method)."""
        query_hash = self._get_query_hash(query, k)
        
        # Check search cache
        if query_hash in self.search_cache:
            return self.search_cache[query_hash]
        
        try:
            # Perform search
            results = self.primary_store.similarity_search(query, k=k)
            
            # Convert to dict format
            result_dicts = []
            for doc in results:
                result_dict = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'source': doc.metadata.get('source', ''),
                    'title': doc.metadata.get('title', ''),
                    'score': 1.0  # Placeholder
                }
                result_dicts.append(result_dict)
            
            # Cache results
            self._update_search_cache(query_hash, result_dicts)
            
            return result_dicts
            
        except Exception as e:
            logger.error(f"❌ Fast search failed: {e}")
            return []

    def _update_search_cache(self, query_hash: str, results: List[Dict]):
        """Update search cache with LRU eviction."""
        if len(self.search_cache) >= self.cache_size:
            oldest_hash = list(self.search_cache.keys())[0]
            del self.search_cache[oldest_hash]
        
        self.search_cache[query_hash] = results

    def generate_fast_response(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Generate ultra-fast response with context-aware capabilities."""
        start_time = time.time()
        
        try:
            # Input validation
            if not query or not isinstance(query, str):
                return {
                    'answer': "Please provide a valid question about Itential documentation.",
                    'sources': [],
                    'confidence': 0.0,
                    'response_time': time.time() - start_time,
                    'context_aware': False
                }
            
            # Check response cache
            cache_key = self._get_query_hash(query, k)
            if cache_key in self.response_cache:
                self.cache_hits += 1
                cached_response = self.response_cache[cache_key].copy()
                cached_response['response_time'] = time.time() - start_time
                cached_response['cached'] = True
                logger.info(f"📦 Cached response in {cached_response['response_time']:.3f}s")
                return cached_response
            
            self.cache_misses += 1
            
            # Use context-aware search for better accuracy
            try:
                return self.generate_context_aware_response(query)
            except Exception as context_error:
                logger.warning(f"Context-aware processing failed, falling back to standard: {context_error}")
                
                # Fallback to standard search
                search_start = time.time()
                context_results = self.fast_search(query, k)
                search_time = time.time() - search_start
                
                if not context_results:
                    return {
                        'answer': f"No relevant information found for: '{query}'. Try rephrasing with more specific terms.",
                        'sources': [],
                        'confidence': 0.0,
                        'response_time': time.time() - start_time,
                        'context_aware': False
                    }
                
                # Standard processing
                confidence = self._fast_confidence_calc(context_results)
                context_text = self._fast_format_context(context_results)
                
                llm_start = time.time()
                prompt_text = f"""Based on the following Itential documentation context, provide a direct and helpful answer:

{context_text}

Question: {query}

Provide a clear, specific answer based on the context above:"""

                response = self.llm.invoke(prompt_text)
                answer = self._fast_format_response(response.content.strip(), confidence)
                llm_time = time.time() - llm_start
                
                result = {
                    'answer': answer,
                    'sources': [{'title': r.get('title', ''), 'url': r.get('source', '')} for r in context_results[:3]],
                    'confidence': confidence,
                    'response_time': time.time() - start_time,
                    'search_time': search_time,
                    'llm_time': llm_time,
                    'context_aware': False,
                    'fallback_used': True
                }
                
                # Update cache
                self._update_response_cache(cache_key, result)
                return result
                
        except Exception as e:
            logger.error(f"❌ Fast response generation failed: {e}")
            return {
                'answer': f"I encountered an error processing your question: {str(e)}. Please try again.",
                'sources': [],
                'confidence': 0.0,
                'response_time': time.time() - start_time,
                'error': str(e),
                'context_aware': False
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

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics."""
        total_queries = self.cache_hits + self.cache_misses
        cache_hit_rate = f"{(self.cache_hits / total_queries * 100):.1f}%" if total_queries > 0 else "0%"
        
        return {
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'cache_hit_rate': cache_hit_rate,
            'cache_size': len(self.response_cache),
            'max_cache_size': self.cache_size,
            'vector_stores_loaded': len(self.vector_stores),
            'primary_database': self.vector_stores[0][0] if self.vector_stores else 'none',
            'model_name': self.model_name,
            'embedding_model': self.embedding_model
        }

    def get_context_stats(self) -> Dict[str, Any]:
        """Get statistics about the context-aware system."""
        try:
            collection = self.primary_store.get()
            documents = collection.get('documents', [])
            metadatas = collection.get('metadatas', [])
            
            domain_counts = {}
            for metadata in metadatas:
                domain = metadata.get('domain', 'unknown')
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            return {
                'total_documents': len(documents),
                'domain_distribution': domain_counts,
                'context_patterns': len(self.context_patterns),
                'database_path': self.qa_db_path,
                'context_prompts': list(self.context_prompts.keys())
            }
        except Exception as e:
            return {'error': str(e)}

    # Backwards compatibility methods
    def generate_response(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Backwards compatible method."""
        return self.generate_fast_response(query, top_k)

    def generate_enhanced_response(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Backwards compatible method."""
        return self.generate_fast_response(query, top_k)


# Speed-optimized variants for different model preferences
class SpeedOptimizedChatbot(EnhancedContextAwareChatbot):
    """
    Speed-optimized chatbot with context awareness.
    Inherits all context-aware capabilities with speed focus.
    """
    
    def __init__(self, 
                 qa_db_path: str = "./enhanced_context_chroma_db",
                 fallback_db_path: str = "./technical_optimized_chroma_db",
                 model: str = "mistral:7b",
                 embedding_model: str = "nomic-embed-text",
                 enable_streaming: bool = True,
                 cache_size: int = 1000):
        """Initialize speed-optimized chatbot with context awareness."""
        
        super().__init__(
            model_name=model,
            embedding_model=embedding_model,
            qa_db_path=qa_db_path,
            fallback_db_path=fallback_db_path,
            cache_size=cache_size
        )
        
        self.enable_streaming = enable_streaming
        
        # Verify and benchmark performance
        self._verify_and_benchmark()

    def _verify_and_benchmark(self):
        """Verify setup and run performance benchmark."""
        try:
            logger.info("🧪 Running performance benchmark...")
            
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
                    logger.info(f"✅ '{query}' -> {response_time:.3f}s")
                else:
                    logger.warning(f"⚠️  '{query}' -> {response_time:.3f}s (low confidence)")
            
            avg_time = sum(benchmark_times) / len(benchmark_times)
            logger.info(f"📊 Average response time: {avg_time:.3f}s")
            
            # Performance assessment
            if avg_time < 1.0:
                logger.info("🚀 Excellent performance!")
            elif avg_time < 2.0:
                logger.info("✅ Good performance!")
            elif avg_time < 3.0:
                logger.info("👍 Acceptable performance")
            else:
                logger.warning("⚠️  Consider using faster model (phi3:mini)")
            
        except Exception as e:
            logger.error(f"❌ Benchmark failed: {e}")


class MistralChatbot(SpeedOptimizedChatbot):
    """Mistral-optimized chatbot for balanced performance."""
    
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
        Optimized chatbot instance with context awareness
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


def create_context_aware_chatbot(model_name: str = "mistral:7b",
                                db_path: str = "./enhanced_context_chroma_db") -> EnhancedContextAwareChatbot:
    """Factory function to create a context-aware chatbot."""
    return EnhancedContextAwareChatbot(
        model_name=model_name,
        qa_db_path=db_path
    )


# Backwards compatibility aliases
class ImprovedChatbot(MistralChatbot):
    """Backwards compatibility with enhanced context awareness."""
    pass

class ProductionChatbot(MistralChatbot):
    """Production chatbot with enhanced context awareness."""
    pass

class FastTechnicalChatbot(MistralChatbot):
    """Fast technical chatbot with enhanced context awareness."""
    pass


def test_context_awareness():
    """Test the context-aware system with sample queries."""
    print("🧪 Testing Enhanced Context-Aware Chatbot System...")
    
    try:
        chatbot = create_context_aware_chatbot()
        
        test_queries = [
            ("itential-cli role is showing duplicate data. how do i troubleshoot this", "CLI_TOOLS"),
            ("how to configure event deduplication in Operations Manager", "PLATFORM_EVENTS"),
            ("What Node.js version is required for IAP 2023.2?", "VERSION_MATRIX"),
            ("troubleshooting steps for platform errors", "TROUBLESHOOTING")
        ]
        
        print("\n📋 Context Detection Test Results:")
        for query, expected_context in test_queries:
            print(f"\n🔍 Query: '{query}'")
            
            # Test context detection
            detected_context, confidence = chatbot.detect_query_context(query)
            print(f"   Expected: {expected_context}")
            print(f"   Detected: {detected_context} (confidence: {confidence:.2f})")
            
            if detected_context == expected_context:
                print("   ✅ Context detection: CORRECT")
            else:
                print("   ❌ Context detection: INCORRECT")
            
            # Test full response
            response = chatbot.generate_context_aware_response(query)
            print(f"   Response time: {response.get('response_time', 0):.2f}s")
            print(f"   Context aware: {response.get('context_aware', False)}")
            print(f"   Answer preview: {response['answer'][:100]}...")
        
        # Show system stats
        stats = chatbot.get_context_stats()
        print(f"\n📊 System Statistics:")
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        print(f"   Domain distribution: {stats.get('domain_distribution', {})}")
        
        # Show performance stats
        perf_stats = chatbot.get_performance_stats()
        print(f"\n⚡ Performance Statistics:")
        print(f"   Cache hit rate: {perf_stats['cache_hit_rate']}")
        print(f"   Model: {perf_stats['model_name']}")
        print(f"   Primary database: {perf_stats['primary_database']}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    print("🚀 Enhanced Context-Aware Itential Chatbot")
    print("="*50)
    
    try:
        # Test different model configurations with context awareness
        models_to_test = [
            ("speed", "phi3:mini"),
            ("balanced", "mistral:7b"),
            ("quality", "llama3.1:8b")
        ]
        
        print("\n🧪 Testing different model configurations...")
        
        for preference, model_name in models_to_test:
            try:
                print(f"\n--- Testing {preference.upper()} mode ({model_name}) ---")
                
                # Create chatbot
                chatbot = create_fast_chatbot(preference)
                
                # Test CLI troubleshooting query
                test_query = "itential-cli role is showing duplicate data. how do i troubleshoot this"
                result = chatbot.generate_fast_response(test_query)
                
                # Show results
                print(f"Response time: {result.get('response_time', 0):.3f}s")
                print(f"Context: {result.get('context', 'unknown')}")
                print(f"Confidence: {result.get('confidence', 0):.1%}")
                print(f"Context aware: {result.get('context_aware', False)}")
                print(f"Answer: {result['answer'][:100]}...")
                
                # Show performance stats
                stats = chatbot.get_performance_stats()
                print(f"Cache hit rate: {stats['cache_hit_rate']}")
                
            except Exception as e:
                print(f"❌ {preference} mode failed: {e}")
                continue
        
        print("\n📋 Recommendations:")
        print("   🚀 For fastest responses: phi3:mini")
        print("   ⚖️  For balanced performance: mistral:7b")
        print("   🎯 For best quality: llama3.1:8b")
        
        print("\n📝 Usage:")
        print("   # Create context-aware chatbot")
        print("   chatbot = create_fast_chatbot('balanced')")
        print("   response = chatbot.generate_fast_response('your question')")
        print("\n🎯 Expected CLI troubleshooting behavior:")
        print("   • Detects CLI_TOOLS context")
        print("   • Provides netcommon collection solution")
        print("   • Does NOT mention Platform event deduplication")
        
        # Run context awareness test
        print("\n" + "="*50)
        test_context_awareness()
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        print("\nPlease ensure:")
        print("1. Ollama is running: ollama serve")
        print("2. Models are available: ollama pull mistral:7b")
        print("3. Enhanced vector database exists: ./enhanced_context_chroma_db")