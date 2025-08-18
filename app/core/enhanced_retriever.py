import re
import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import logging
import json

# Universal imports with backwards compatibility
try:
    from compatibility_imports import get_chroma, get_documents
    Chroma = get_chroma()
    Document = get_documents()
except ImportError:
    # Fallback imports
    try:
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain.vectorstores import Chroma
    
    try:
        from langchain_core.documents import Document
    except ImportError:
        try:
            from langchain.docstore.document import Document
        except ImportError:
            from langchain.schema import Document

logger = logging.getLogger(__name__)

class DependencyAwareRetriever:
    """Enhanced retriever specifically designed for technical documentation with dependencies."""
    
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        self.keyword_index = defaultdict(set)
        self.version_patterns = {}
        self.dependency_cache = {}
        
        # Pre-compiled regex patterns for better performance and safety
        self._compile_regex_patterns()
        
        # Build indexes with error handling
        self._build_specialized_indexes()

    def _compile_regex_patterns(self):
        """Compile regex patterns with proper error handling."""
        try:
            # Technology detection patterns
            self.tech_pattern = re.compile(
                r'\b(?:python|node(?:js)?|npm|pip|java|mongodb|redis|vault|rabbitmq|ansible|terraform)\b',
                re.IGNORECASE
            )
            
            # Version detection patterns - more robust
            self.version_pattern = re.compile(
                r'\b(\d+\.\d+(?:\.\d+)?)\b'
            )
            
            # Version range patterns - fixed escaping
            self.version_range_pattern = re.compile(
                r'([><=]+)\s*(\d+\.\d+(?:\.\d+)?)'
            )
            
            # Release version patterns
            self.release_pattern = re.compile(
                r'\b(20\d{2}\.\d+)\b'
            )
            
            # Dependency-specific patterns - safer regex
            self.python_version_pattern = re.compile(
                r'python\s*(?:[:\-=]|\s+version)?\s*([><=]*\s*\d+\.\d+(?:\.\d+)?)',
                re.IGNORECASE
            )
            
            self.node_version_pattern = re.compile(
                r'node(?:js|\.js)?\s*(?:[:\-=]|\s+version)?\s*([><=]*\s*\d+\.\d+(?:\.\d+)?)',
                re.IGNORECASE
            )
            
            self.platform_version_pattern = re.compile(
                r'(?:iap|platform)\s*(?:[:\-=]|\s+version)?\s*(20\d{2}\.\d+|\d+\.\d+)',
                re.IGNORECASE
            )
            
            # Multi-word technical phrases
            self.phrase_pattern = re.compile(
                r'(?:python\s+version|node\s+version|system\s+requirements|software\s+dependencies)',
                re.IGNORECASE
            )
            
            logger.info("✅ Regex patterns compiled successfully")
            
        except re.error as e:
            logger.error(f"❌ Failed to compile regex patterns: {e}")
            # Fallback to simple string matching
            self._use_fallback_patterns()

    def _use_fallback_patterns(self):
        """Use simple string matching as fallback."""
        self.tech_pattern = None
        self.version_pattern = None
        self.version_range_pattern = None
        self.release_pattern = None
        self.python_version_pattern = None
        self.node_version_pattern = None
        self.platform_version_pattern = None
        self.phrase_pattern = None
        logger.warning("⚠️  Using fallback string matching due to regex compilation failure")

    def _build_specialized_indexes(self):
        """Build specialized indexes for better technical content retrieval."""
        try:
            # Get all documents from vector store with error handling
            collection = self.vector_store.get()
            if not collection:
                logger.warning("⚠️  Vector store collection is empty")
                return
                
            documents = collection.get('documents', [])
            metadatas = collection.get('metadatas', [])
            
            if not documents:
                logger.warning("⚠️  No documents found in vector store")
                return
            
            logger.info(f"🔍 Building specialized indexes from {len(documents)} documents...")
            
            processed_count = 0
            error_count = 0
            
            for doc_id, (content, metadata) in enumerate(zip(documents, metadatas)):
                try:
                    if not content or not isinstance(content, str):
                        continue
                    
                    # Safely process content
                    content_safe = self._sanitize_content(content)
                    if not content_safe:
                        continue
                    
                    # Build keyword index with technical terms
                    keywords = self._extract_technical_keywords(content_safe)
                    for keyword in keywords:
                        if keyword:  # Ensure keyword is not empty
                            self.keyword_index[keyword].add(doc_id)
                    
                    # Build version pattern index
                    versions = self._extract_version_patterns(content_safe)
                    for version in versions:
                        if version:  # Ensure version is not empty
                            if version not in self.version_patterns:
                                self.version_patterns[version] = set()
                            self.version_patterns[version].add(doc_id)
                    
                    # Cache dependency information
                    if self._is_dependency_content(content_safe, metadata or {}):
                        dep_info = self._extract_dependency_info(content_safe)
                        if dep_info:  # Only cache if we extracted valid info
                            self.dependency_cache[doc_id] = dep_info
                    
                    processed_count += 1
                    
                except Exception as e:
                    error_count += 1
                    logger.debug(f"Error processing document {doc_id}: {e}")
                    continue
            
            logger.info(f"✅ Built indexes: {len(self.keyword_index)} keywords, "
                       f"{len(self.version_patterns)} versions, "
                       f"{len(self.dependency_cache)} dependency documents")
            logger.info(f"📊 Processed: {processed_count}, Errors: {error_count}")
                       
        except Exception as e:
            logger.error(f"❌ Failed to build specialized indexes: {e}")
            # Initialize empty indexes to prevent further errors
            self.keyword_index = defaultdict(set)
            self.version_patterns = {}
            self.dependency_cache = {}

    def _sanitize_content(self, content: str) -> str:
        """Safely sanitize content to prevent parsing errors."""
        try:
            if not isinstance(content, str):
                return ""
            
            # Remove any potential problematic characters
            # Replace smart quotes and other unicode issues
            content = content.replace('"', '"').replace('"', '"')
            content = content.replace(''', "'").replace(''', "'")
            
            # Remove null bytes and other control characters
            content = content.replace('\x00', '')
            
            # Normalize whitespace
            content = re.sub(r'\s+', ' ', content)
            
            return content.strip()
            
        except Exception as e:
            logger.debug(f"Error sanitizing content: {e}")
            return ""

    def _extract_technical_keywords(self, text: str) -> List[str]:
        """Extract technical keywords and phrases from text with error handling."""
        try:
            if not text:
                return []
            
            text_lower = text.lower()
            keywords = []
            
            # Technology terms - use regex if available, otherwise fallback
            if self.tech_pattern:
                try:
                    tech_terms = self.tech_pattern.findall(text_lower)
                    keywords.extend(tech_terms)
                except Exception as e:
                    logger.debug(f"Tech pattern error: {e}")
                    # Fallback to simple string search
                    tech_fallback = ['python', 'nodejs', 'node', 'mongodb', 'java', 'npm', 'pip']
                    for term in tech_fallback:
                        if term in text_lower:
                            keywords.append(term)
            else:
                # Simple string matching fallback
                tech_terms = ['python', 'nodejs', 'node', 'mongodb', 'java', 'npm', 'pip', 'redis', 'vault']
                for term in tech_terms:
                    if term in text_lower:
                        keywords.append(term)
            
            # Version-related terms
            version_terms = ['version', 'requirement', 'dependency', 'prerequisite', 'supported', 'compatible', 'required']
            for term in version_terms:
                if term in text_lower:
                    keywords.append(term)
            
            # Product terms
            product_terms = ['iap', 'itential', 'platform', 'gateway', 'iag']
            for term in product_terms:
                if term in text_lower:
                    keywords.append(term)
            
            # Release versions - use regex if available
            if self.release_pattern:
                try:
                    releases = self.release_pattern.findall(text_lower)
                    keywords.extend(releases)
                except Exception as e:
                    logger.debug(f"Release pattern error: {e}")
            else:
                # Simple fallback for common releases
                release_fallback = ['2023.1', '2023.2', '2022.1']
                for release in release_fallback:
                    if release in text_lower:
                        keywords.append(release)
            
            # Multi-word technical phrases - use regex if available
            if self.phrase_pattern:
                try:
                    phrases = self.phrase_pattern.findall(text_lower)
                    keywords.extend(phrases)
                except Exception as e:
                    logger.debug(f"Phrase pattern error: {e}")
            
            return list(set(filter(None, keywords)))  # Remove duplicates and empty strings
            
        except Exception as e:
            logger.debug(f"Error extracting technical keywords: {e}")
            return []

    def _extract_version_patterns(self, text: str) -> List[str]:
        """Extract version numbers and patterns with error handling."""
        try:
            if not text:
                return []
            
            patterns = []
            
            # Semantic versions - use regex if available
            if self.version_pattern:
                try:
                    semantic = self.version_pattern.findall(text)
                    patterns.extend(semantic)
                except Exception as e:
                    logger.debug(f"Version pattern error: {e}")
            
            # Range patterns - use regex if available
            if self.version_range_pattern:
                try:
                    ranges = self.version_range_pattern.findall(text)
                    # Convert tuples to strings
                    range_strings = [f"{op}{version}" for op, version in ranges]
                    patterns.extend(range_strings)
                except Exception as e:
                    logger.debug(f"Version range pattern error: {e}")
            
            # Release patterns - use regex if available
            if self.release_pattern:
                try:
                    releases = self.release_pattern.findall(text)
                    patterns.extend(releases)
                except Exception as e:
                    logger.debug(f"Release pattern error: {e}")
            
            return list(set(filter(None, patterns)))  # Remove duplicates and empty strings
            
        except Exception as e:
            logger.debug(f"Error extracting version patterns: {e}")
            return []

    def _is_dependency_content(self, content: str, metadata: Dict) -> bool:
        """Check if content contains dependency information with error handling."""
        try:
            if not content or not isinstance(content, str):
                return False
            
            indicators = [
                'dependency', 'dependencies', 'requirement', 'requirements',
                'version', 'prerequisite', 'supported', 'compatible'
            ]
            
            content_lower = content.lower()
            title_lower = metadata.get('title', '').lower() if metadata else ''
            
            has_indicator = any(ind in content_lower for ind in indicators)
            has_tech = any(tech in content_lower for tech in ['python', 'node', 'java', 'mongodb'])
            
            return has_indicator and has_tech
            
        except Exception as e:
            logger.debug(f"Error checking dependency content: {e}")
            return False

    def _extract_dependency_info(self, content: str) -> Dict:
        """Extract structured dependency information with comprehensive error handling and better table parsing."""
        try:
            if not content or not isinstance(content, str):
                return {}
            
            info = {
                'python_versions': [],
                'node_versions': [],
                'platform_versions': [],
                'requirements': [],
                'table_data': []  # Store raw table data for better analysis
            }
            
            # First, try to detect and parse table structures
            table_info = self._parse_dependency_tables(content)
            if table_info:
                info['table_data'] = table_info
            
            # Safely extract Python versions
            try:
                if self.python_version_pattern:
                    python_matches = self.python_version_pattern.findall(content)
                    info['python_versions'] = [match.strip() for match in python_matches if match.strip()]
                else:
                    # Fallback: simple string search
                    if 'python' in content.lower():
                        info['python_versions'] = ['found']
            except Exception as e:
                logger.debug(f"Error extracting Python versions: {e}")
            
            # Safely extract Node.js versions - IMPROVED LOGIC
            try:
                if self.node_version_pattern:
                    node_matches = self.node_version_pattern.findall(content)
                    # Filter out obvious non-Node.js versions (like Vault versions)
                    filtered_matches = []
                    for match in node_matches:
                        version_num = match.strip()
                        # Skip versions that are clearly not Node.js (like 1.15.4 which is typical for Vault)
                        if version_num and not self._is_likely_vault_version(version_num, content):
                            filtered_matches.append(version_num)
                    info['node_versions'] = filtered_matches
                else:
                    # Fallback: simple string search
                    if any(term in content.lower() for term in ['node', 'nodejs']):
                        info['node_versions'] = ['found']
            except Exception as e:
                logger.debug(f"Error extracting Node versions: {e}")
            
            # Safely extract Platform versions
            try:
                if self.platform_version_pattern:
                    platform_matches = self.platform_version_pattern.findall(content)
                    info['platform_versions'] = [match.strip() for match in platform_matches if match.strip()]
                else:
                    # Fallback: simple string search
                    if any(term in content.lower() for term in ['iap', 'platform']):
                        info['platform_versions'] = ['found']
            except Exception as e:
                logger.debug(f"Error extracting Platform versions: {e}")
            
            return info
            
        except Exception as e:
            logger.debug(f"Error extracting dependency info: {e}")
            return {}
    
    def _is_likely_vault_version(self, version: str, content: str) -> bool:
        """Check if a version number is likely a Vault version rather than Node.js."""
        try:
            # Vault versions are typically 1.x.x, while Node.js is typically 16.x.x, 18.x.x, 20.x.x
            if version.startswith('1.') and 'vault' in content.lower():
                return True
            
            # Check if the version appears in a vault context
            vault_context_pattern = re.compile(
                rf'vault\s*[:\-\|]?\s*{re.escape(version)}',
                re.IGNORECASE
            )
            if vault_context_pattern.search(content):
                return True
                
            return False
            
        except Exception as e:
            logger.debug(f"Error checking vault version: {e}")
            return False
    
    def _parse_dependency_tables(self, content: str) -> List[Dict]:
        """Parse dependency tables to better understand column structure."""
        try:
            if not content:
                return []
            
            tables = []
            
            # Look for table-like structures
            lines = content.split('\n')
            current_table = []
            in_table = False
            
            for line in lines:
                line = line.strip()
                if not line:
                    if current_table and in_table:
                        # End of table
                        table_info = self._analyze_table_structure(current_table)
                        if table_info:
                            tables.append(table_info)
                        current_table = []
                        in_table = False
                    continue
                
                # Detect table headers or separators
                if any(indicator in line.lower() for indicator in ['platform', 'version', 'node', 'python', 'vault']):
                    if '|' in line or '\t' in line or '  ' in line:
                        in_table = True
                        current_table.append(line)
                elif in_table:
                    current_table.append(line)
            
            # Process final table if exists
            if current_table and in_table:
                table_info = self._analyze_table_structure(current_table)
                if table_info:
                    tables.append(table_info)
            
            return tables
            
        except Exception as e:
            logger.debug(f"Error parsing dependency tables: {e}")
            return []
    
    def _analyze_table_structure(self, table_lines: List[str]) -> Optional[Dict]:
        """Analyze table structure to identify columns correctly."""
        try:
            if not table_lines:
                return None
            
            # Try to identify headers and data
            headers = []
            data_rows = []
            
            for line in table_lines:
                # Split by common separators
                if '|' in line:
                    cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                elif '\t' in line:
                    cells = [cell.strip() for cell in line.split('\t') if cell.strip()]
                else:
                    # Split by multiple spaces
                    cells = [cell.strip() for cell in re.split(r'\s{2,}', line) if cell.strip()]
                
                if cells:
                    # Check if this looks like a header row
                    if any(header_term in ' '.join(cells).lower() 
                          for header_term in ['platform', 'version', 'node', 'python', 'vault']):
                        if not headers:  # First header row found
                            headers = cells
                        continue
                    else:
                        data_rows.append(cells)
            
            if headers and data_rows:
                return {
                    'headers': headers,
                    'data': data_rows,
                    'type': 'dependency_table'
                }
            
            return None
            
        except Exception as e:
            logger.debug(f"Error analyzing table structure: {e}")
            return None

    def _calculate_relevance_score(self, query: str, doc_content: str, metadata: Dict, doc_id: int) -> float:
        """Calculate enhanced relevance score for dependency queries with error handling."""
        try:
            if not query or not doc_content:
                return 0.0
            
            score = 0.0
            query_lower = query.lower()
            content_lower = doc_content.lower()
            
            # Extract query components safely
            query_keywords = self._extract_technical_keywords(query)
            query_versions = self._extract_version_patterns(query)
            
            # Base keyword matching
            for keyword in query_keywords:
                if keyword and keyword in content_lower:
                    score += 2.0
                    
                    # Boost for exact matches in title
                    title = metadata.get('title', '') if metadata else ''
                    if keyword in title.lower():
                        score += 3.0
            
            # Version-specific matching
            for version in query_versions:
                if version and version in content_lower:
                    score += 5.0  # High boost for version matches
            
            # Special handling for dependency queries
            dependency_terms = ['python version', 'node version', 'requirements', 'dependencies']
            if any(term in query_lower for term in dependency_terms):
                
                # Check if this is a dependency document
                if doc_id in self.dependency_cache:
                    score += 10.0  # Major boost for dependency docs
                    
                    dep_info = self.dependency_cache[doc_id]
                    
                    # Specific technology matching
                    if 'python' in query_lower and dep_info.get('python_versions'):
                        score += 15.0
                    if 'node' in query_lower and dep_info.get('node_versions'):
                        score += 15.0
                    
                    # Platform version matching
                    platform_versions = dep_info.get('platform_versions', [])
                    for platform_version in platform_versions:
                        if platform_version and platform_version in query_lower:
                            score += 20.0
            
            # Content type boosting - safely access metadata
            if metadata:
                content_type = metadata.get('content_type', '')
                if content_type == 'table':
                    score += 8.0
                elif content_type == 'table_row':
                    score += 5.0
                elif content_type == 'summary':
                    score += 3.0
                
                # Priority boosting
                priority = metadata.get('priority', '')
                if priority == 'high':
                    score += 7.0
                elif priority == 'medium':
                    score += 3.0
                
                # Source URL boosting
                source = metadata.get('source', '')
                if source and ('dependencies' in source or 'requirements' in source):
                    score += 10.0
            
            return max(0.0, score)  # Ensure non-negative score
            
        except Exception as e:
            logger.debug(f"Error calculating relevance score: {e}")
            return 0.0

    def hybrid_search(self, query: str, k: int = 5) -> List[Dict]:
        """Perform hybrid search combining vector similarity and specialized indexes with comprehensive error handling."""
        try:
            if not query or not isinstance(query, str):
                logger.warning("Invalid query provided to hybrid_search")
                return []
            
            # Vector similarity search with error handling
            vector_results = []
            try:
                vector_results = self.vector_store.similarity_search_with_score(query, k=k*3)
            except Exception as e:
                logger.warning(f"Vector search failed: {e}")
                # Continue with keyword search only
            
            # Get all documents for keyword search with error handling
            documents = []
            metadatas = []
            try:
                collection = self.vector_store.get()
                if collection:
                    documents = collection.get('documents', [])
                    metadatas = collection.get('metadatas', [])
            except Exception as e:
                logger.warning(f"Failed to get documents for keyword search: {e}")
            
            # Keyword-based search using specialized indexes
            keyword_candidates = set()
            
            try:
                query_keywords = self._extract_technical_keywords(query)
                query_versions = self._extract_version_patterns(query)
                
                # Find candidates based on keywords
                for keyword in query_keywords:
                    if keyword and keyword in self.keyword_index:
                        keyword_candidates.update(self.keyword_index[keyword])
                
                # Find candidates based on versions
                for version in query_versions:
                    if version and version in self.version_patterns:
                        keyword_candidates.update(self.version_patterns[version])
                        
            except Exception as e:
                logger.warning(f"Keyword search failed: {e}")
            
            # Score all candidates
            all_candidates = {}
            
            # Add vector results with error handling
            for doc, distance in vector_results:
                try:
                    content = doc.page_content
                    metadata = doc.metadata or {}
                    
                    # Create a safe document ID
                    content_preview = content[:100] if content else ""
                    doc_id = hash(content_preview)
                    
                    # Convert distance to similarity score
                    vector_score = max(0, (2.0 - distance)) * 0.5
                    relevance_score = self._calculate_relevance_score(query, content, metadata, doc_id)
                    
                    combined_score = vector_score + relevance_score
                    
                    all_candidates[doc_id] = {
                        'document': doc,
                        'score': combined_score,
                        'content': content,
                        'metadata': metadata,
                        'source_type': 'vector'
                    }
                    
                except Exception as e:
                    logger.debug(f"Error processing vector result: {e}")
                    continue
            
            # Add keyword candidates with error handling
            for doc_id in keyword_candidates:
                try:
                    if doc_id < len(documents):
                        content = documents[doc_id]
                        metadata = metadatas[doc_id] if doc_id < len(metadatas) else {}
                        
                        if not content:
                            continue
                        
                        relevance_score = self._calculate_relevance_score(query, content, metadata, doc_id)
                        
                        content_preview = content[:100] if content else ""
                        hash_id = hash(content_preview)
                        
                        if hash_id in all_candidates:
                            # Boost existing candidates
                            all_candidates[hash_id]['score'] += relevance_score * 0.3
                        else:
                            # Add new candidate
                            try:
                                doc = Document(page_content=content, metadata=metadata)
                                all_candidates[hash_id] = {
                                    'document': doc,
                                    'score': relevance_score * 0.7,
                                    'content': content,
                                    'metadata': metadata,
                                    'source_type': 'keyword'
                                }
                            except Exception as e:
                                logger.debug(f"Error creating document: {e}")
                                continue
                                
                except Exception as e:
                    logger.debug(f"Error processing keyword candidate {doc_id}: {e}")
                    continue
            
            # Sort by score and return top results
            try:
                sorted_results = sorted(
                    all_candidates.values(),
                    key=lambda x: x.get('score', 0),
                    reverse=True
                )[:k]
            except Exception as e:
                logger.error(f"Error sorting results: {e}")
                sorted_results = list(all_candidates.values())[:k]
            
            # Format results with error handling
            formatted_results = []
            for result in sorted_results:
                try:
                    metadata = result.get('metadata', {})
                    formatted_result = {
                        'content': result.get('content', ''),
                        'source': metadata.get('source', '') if metadata else '',
                        'title': metadata.get('title', '') if metadata else '',
                        'score': result.get('score', 0),
                        'metadata': metadata,
                        'content_type': metadata.get('content_type', 'unknown') if metadata else 'unknown'
                    }
                    formatted_results.append(formatted_result)
                except Exception as e:
                    logger.debug(f"Error formatting result: {e}")
                    continue
            
            # Log results for debugging
            logger.info(f"🔍 Query: '{query}' returned {len(formatted_results)} results")
            for i, result in enumerate(formatted_results[:3], 1):
                title = result.get('title', '')[:50]
                score = result.get('score', 0)
                content_type = result.get('content_type', 'unknown')
                logger.info(f"  {i}. Score: {score:.2f}, Type: {content_type}, Title: {title}...")
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ Critical error in hybrid search: {e}")
            # Return empty results rather than crashing
            return []

    def search_dependencies(self, query: str, k: int = 5) -> List[Dict]:
        """Specialized search for dependency-related queries with error handling."""
        try:
            if not query or not isinstance(query, str):
                logger.warning("Invalid query provided to search_dependencies")
                return []
            
            # Pre-filter for dependency documents
            dependency_docs = []
            
            try:
                collection = self.vector_store.get()
                if not collection:
                    logger.warning("No collection available for dependency search")
                    return self.hybrid_search(query, k)  # Fallback
                
                documents = collection.get('documents', [])
                metadatas = collection.get('metadatas', [])
                
                if not documents:
                    logger.warning("No documents available for dependency search")
                    return []
                
                for doc_id, (content, metadata) in enumerate(zip(documents, metadatas)):
                    try:
                        if not content:
                            continue
                        
                        metadata = metadata or {}
                        
                        # Check if this is a dependency document
                        is_cached_dep = doc_id in self.dependency_cache
                        is_dep_content = self._is_dependency_content(content, metadata)
                        
                        if is_cached_dep or is_dep_content:
                            score = self._calculate_relevance_score(query, content, metadata, doc_id)
                            
                            if score > 5.0:  # Only include highly relevant docs
                                dependency_docs.append({
                                    'content': content,
                                    'source': metadata.get('source', ''),
                                    'title': metadata.get('title', ''),
                                    'score': score,
                                    'metadata': metadata
                                })
                                
                    except Exception as e:
                        logger.debug(f"Error processing dependency document {doc_id}: {e}")
                        continue
                
                # Sort and return
                dependency_docs.sort(key=lambda x: x.get('score', 0), reverse=True)
                return dependency_docs[:k]
                
            except Exception as e:
                logger.error(f"Error in dependency search: {e}")
                return self.hybrid_search(query, k)  # Fallback to regular search
                
        except Exception as e:
            logger.error(f"❌ Critical error in dependency search: {e}")
            return []  # Return empty results rather than crashing