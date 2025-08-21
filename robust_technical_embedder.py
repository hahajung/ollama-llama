#!/usr/bin/env python3
"""
Enhanced Context-Aware Technical Documentation Embedder
COMPLETE REPLACEMENT for robust_technical_embedder.py
Creates optimized vector database with domain-specific context to prevent over-matching.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import time
from collections import defaultdict

# Universal imports
try:
    from compatibility_imports import (
        get_ollama_embeddings, get_chroma, 
        get_text_splitter, get_documents
    )
    OllamaEmbeddings = get_ollama_embeddings()
    Chroma = get_chroma()
    RecursiveCharacterTextSplitter = get_text_splitter()
    Document = get_documents()
except ImportError:
    try:
        from langchain_community.embeddings import OllamaEmbeddings
        from langchain_community.vectorstores import Chroma
    except ImportError:
        from langchain.embeddings import OllamaEmbeddings
        from langchain.vectorstores import Chroma
    
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    
    try:
        from langchain_core.documents import Document
    except ImportError:
        try:
            from langchain.docstore.document import Document
        except ImportError:
            from langchain.schema import Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedTechnicalDocumentationEmbedder:
    """Enhanced embedder with context-aware processing to prevent over-matching."""
    
    def __init__(self, data_file: str = "complete_technical_docs.jsonl"):
        self.data_file = Path(data_file)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Enhanced domain categorization
        self.domain_categories = {
            'CLI_TOOLS': {
                'keywords': ['itential-cli', 'cli', 'command line', 'ansible-galaxy', 'netcommon'],
                'context_prefix': 'CLI_TROUBLESHOOTING',
                'priority': 'CRITICAL'
            },
            'PLATFORM_EVENTS': {
                'keywords': ['event service', 'deduplication', 'operations manager', 'email adapter'],
                'context_prefix': 'PLATFORM_EVENTS',
                'priority': 'HIGH'
            },
            'TROUBLESHOOTING': {
                'keywords': ['troubleshooting', 'error', 'issue', 'problem', 'duplicate'],
                'context_prefix': 'TROUBLESHOOTING_GUIDE',
                'priority': 'HIGH'
            },
            'VERSION_DEPENDENCIES': {
                'keywords': ['version', 'compatibility', 'requirements', 'dependencies'],
                'context_prefix': 'VERSION_REQUIREMENTS',
                'priority': 'HIGH'
            },
            'INSTALLATION_CONFIG': {
                'keywords': ['installation', 'setup', 'deployment', 'configuration'],
                'context_prefix': 'INSTALLATION_GUIDE',
                'priority': 'MEDIUM'
            }
        }
        
        # Enhanced question templates with domain specificity
        self.enhanced_question_templates = {
            'cli_specific': [
                "CLI: How to fix {problem} in itential-cli?",
                "CLI: {tool} {issue} troubleshooting steps",
                "CLI: itential-cli {error} resolution",
                "CLI_COMMAND: {command} duplicate data fix",
                "CLI_NETCOMMON: {issue} in ansible netcommon {version}",
                "CLI_TROUBLESHOOT: itential-cli role showing duplicate data",
                "ANSIBLE: netcommon collection version {version} issues"
            ],
            'platform_specific': [
                "PLATFORM: {service} {issue} in Itential Platform",
                "PLATFORM_CONFIG: {component} configuration for {issue}",
                "PLATFORM_EVENTS: {trigger} deduplication setup",
                "EVENT_SERVICE: email adapter trigger configuration",
                "OPERATIONS_MANAGER: trigger type configuration"
            ],
            'version_specific': [
                "VERSION: What {dependency} version for {product} {version}?",
                "COMPATIBILITY: {tool} {version} requirements",
                "DEPENDENCY: {component} version matrix for {product}",
                "REQUIREMENTS: System requirements for {product} {version}"
            ],
            'troubleshooting_specific': [
                "TROUBLESHOOT: How to resolve {issue}?",
                "PROBLEM: {error} in {component}",
                "SOLUTION: Fix for {problem} in {system}",
                "DEBUG: {issue} debugging steps"
            ]
        }
        
        # Track processing stats
        self.processing_stats = {
            'cli_documents': 0,
            'platform_documents': 0,
            'version_documents': 0,
            'troubleshooting_documents': 0,
            'general_documents': 0,
            'total_qa_pairs': 0
        }

    def detect_document_domain(self, doc_data: Dict[str, Any]) -> Tuple[str, str, float]:
        """
        Detect document domain with enhanced accuracy.
        Returns (domain, context_prefix, confidence)
        """
        # Get document content and metadata
        title = doc_data.get('title', '').lower()
        url = doc_data.get('url', '').lower()
        content = doc_data.get('raw_text', '').lower()
        category = doc_data.get('category', '').upper()
        
        # Use category from scraper if available
        if category in self.domain_categories:
            domain_info = self.domain_categories[category]
            return category, domain_info['context_prefix'], 0.9
        
        # Fallback to content analysis
        domain_scores = {}
        
        for domain_name, domain_info in self.domain_categories.items():
            score = 0
            
            # URL keyword matching
            for keyword in domain_info['keywords']:
                if keyword in url:
                    score += 15
            
            # Title keyword matching
            for keyword in domain_info['keywords']:
                if keyword in title:
                    score += 12
            
            # Content keyword matching
            for keyword in domain_info['keywords']:
                count = content.count(keyword)
                score += min(count * 3, 15)
            
            # Special CLI detection (prevent over-matching)
            if domain_name == 'CLI_TOOLS':
                cli_strong_indicators = [
                    'itential-cli role', 'ansible-galaxy collection', 
                    'netcommon version', 'duplicate return data'
                ]
                for indicator in cli_strong_indicators:
                    if indicator in content:
                        score += 25
                        
            # Special Platform Events detection (separate from CLI)
            elif domain_name == 'PLATFORM_EVENTS':
                platform_indicators = [
                    'event service configuration', 'operations manager trigger',
                    'email adapter', 'platform deduplication'
                ]
                for indicator in platform_indicators:
                    if indicator in content:
                        score += 20
                        
                # Penalty if clearly CLI content
                if 'itential-cli' in content or 'command line' in content:
                    score -= 15
            
            if score > 0:
                domain_scores[domain_name] = score
        
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            confidence = min(domain_scores[best_domain] / 30.0, 1.0)
            domain_info = self.domain_categories[best_domain]
            return best_domain, domain_info['context_prefix'], confidence
        else:
            return 'GENERAL', 'GENERAL_DOCS', 0.3

    def create_context_aware_chunks(self, doc_data: Dict[str, Any]) -> List[Document]:
        """Create context-aware chunks with domain-specific prefixes."""
        documents = []
        
        # Extract basic info
        title = doc_data.get('title', 'Untitled')
        url = doc_data.get('url', '')
        raw_text = doc_data.get('raw_text', '')
        
        if not raw_text or len(raw_text) < 50:
            return documents
        
        # Detect domain
        domain, context_prefix, confidence = self.detect_document_domain(doc_data)
        
        # Create domain-specific chunks
        if domain == 'CLI_TOOLS':
            chunk_size = 400  # Smaller chunks for CLI content
            chunk_overlap = 100
        elif domain == 'PLATFORM_EVENTS':
            chunk_size = 500  # Medium chunks for platform content
            chunk_overlap = 125
        else:
            chunk_size = 800  # Larger chunks for general content
            chunk_overlap = 200
        
        # Create text splitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        # Split content
        chunks = splitter.split_text(raw_text)
        
        # Process each chunk with enhanced context
        for i, chunk in enumerate(chunks):
            # Create context-enhanced content
            enhanced_content = self._enhance_chunk_with_domain_context(
                chunk, context_prefix, domain, title, url, i
            )
            
            # Enhanced metadata
            metadata = {
                'source': url,
                'title': title,
                'chunk_id': f"{domain}_{i}",
                'domain': domain,
                'context_prefix': context_prefix,
                'confidence': confidence,
                'original_category': doc_data.get('category', 'unknown'),
                'priority_score': doc_data.get('priority_score', 0),
                'is_critical': doc_data.get('is_critical', False),
                'chunk_index': i,
                'total_chunks': len(chunks),
                'chunk_type': 'context_aware'
            }
            
            # Add domain-specific metadata
            if domain == 'CLI_TOOLS':
                metadata.update({
                    'tool_type': 'cli',
                    'search_priority': 'critical',
                    'troubleshooting_category': 'command_line'
                })
            elif domain == 'PLATFORM_EVENTS':
                metadata.update({
                    'component_type': 'platform',
                    'service_category': 'events',
                    'search_priority': 'high'
                })
            elif domain == 'VERSION_DEPENDENCIES':
                metadata.update({
                    'info_type': 'version',
                    'search_priority': 'high'
                })
            
            documents.append(Document(
                page_content=enhanced_content,
                metadata=metadata
            ))
        
        return documents

    def _enhance_chunk_with_domain_context(self, chunk: str, context_prefix: str, 
                                         domain: str, title: str, url: str, chunk_index: int) -> str:
        """Enhance chunk content with domain-specific context."""
        # Start with domain context
        enhanced_parts = [f"[{context_prefix}]"]
        
        # Add document context
        enhanced_parts.append(f"DOCUMENT: {title}")
        enhanced_parts.append(f"SOURCE: {url}")
        
        # Add domain-specific context
        if domain == 'CLI_TOOLS':
            enhanced_parts.append("CONTEXT: Command Line Interface Tools and Troubleshooting")
            
            # Extract CLI-specific keywords
            cli_keywords = ['itential-cli', 'ansible-galaxy', 'netcommon', 'duplicate data', 'collection version']
            found_keywords = [kw for kw in cli_keywords if kw.lower() in chunk.lower()]
            if found_keywords:
                enhanced_parts.append(f"CLI_KEYWORDS: {', '.join(found_keywords)}")
            
            # Add CLI troubleshooting indicators
            if 'duplicate' in chunk.lower() and 'data' in chunk.lower():
                enhanced_parts.append("ISSUE_TYPE: Duplicate Data Problem")
            if 'netcommon' in chunk.lower():
                enhanced_parts.append("COMPONENT: Ansible Netcommon Collection")
                
        elif domain == 'PLATFORM_EVENTS':
            enhanced_parts.append("CONTEXT: Platform Event Services and Configuration")
            
            # Extract platform-specific keywords
            platform_keywords = ['event service', 'deduplication', 'operations manager', 'email adapter']
            found_keywords = [kw for kw in platform_keywords if kw.lower() in chunk.lower()]
            if found_keywords:
                enhanced_parts.append(f"PLATFORM_KEYWORDS: {', '.join(found_keywords)}")
                
        elif domain == 'TROUBLESHOOTING':
            enhanced_parts.append("CONTEXT: Problem Resolution and Troubleshooting")
            
            # Extract problem indicators
            problem_patterns = [
                r'duplicate\s+\w+', r'error\s+\w+', r'issue\s+with\s+\w+', 
                r'problem\s+\w+', r'fix\s+\w+', r'resolve\s+\w+'
            ]
            problems_found = []
            for pattern in problem_patterns:
                matches = re.findall(pattern, chunk, re.IGNORECASE)
                problems_found.extend(matches[:2])
            
            if problems_found:
                enhanced_parts.append(f"PROBLEM_INDICATORS: {', '.join(problems_found[:3])}")
                
        elif domain == 'VERSION_DEPENDENCIES':
            enhanced_parts.append("CONTEXT: Version Requirements and Dependencies")
            
            # Extract version information
            version_patterns = [
                r'(?:python|node\.?js|mongodb|redis)\s+(?:version\s+)?(\d+\.\d+)',
                r'(?:iap|platform)\s+(\d{4}\.\d+)',
                r'version\s+(\d+\.\d+\.\d+)'
            ]
            versions_found = []
            for pattern in version_patterns:
                matches = re.findall(pattern, chunk, re.IGNORECASE)
                versions_found.extend(matches[:3])
            
            if versions_found:
                enhanced_parts.append(f"VERSIONS_MENTIONED: {', '.join(versions_found[:5])}")
        
        # Add the actual content
        enhanced_parts.append("CONTENT:")
        enhanced_parts.append(chunk)
        
        return '\n'.join(enhanced_parts)

    def create_enhanced_qa_pairs(self, doc_data: Dict[str, Any]) -> List[Document]:
        """Create enhanced Q&A pairs with domain-specific targeting."""
        documents = []
        
        title = doc_data.get('title', 'Untitled')
        url = doc_data.get('url', '')
        raw_text = doc_data.get('raw_text', '')
        
        if not raw_text or len(raw_text) < 100:
            return documents
        
        # Detect domain
        domain, context_prefix, confidence = self.detect_document_domain(doc_data)
        
        # Create domain-specific Q&A pairs
        if domain == 'CLI_TOOLS':
            qa_pairs = self._create_cli_qa_pairs(doc_data, context_prefix)
        elif domain == 'PLATFORM_EVENTS':
            qa_pairs = self._create_platform_qa_pairs(doc_data, context_prefix)
        elif domain == 'VERSION_DEPENDENCIES':
            qa_pairs = self._create_version_qa_pairs(doc_data, context_prefix)
        elif domain == 'TROUBLESHOOTING':
            qa_pairs = self._create_troubleshooting_qa_pairs(doc_data, context_prefix)
        else:
            qa_pairs = self._create_general_qa_pairs(doc_data, context_prefix)
        
        # Convert to documents
        for question, answer in qa_pairs:
            metadata = {
                'source': url,
                'title': title,
                'domain': domain,
                'context_prefix': context_prefix,
                'confidence': confidence,
                'qa_type': 'enhanced_domain_specific',
                'is_synthetic': True
            }
            
            # Enhanced Q&A format with clear domain marking
            content = f"[{context_prefix}_QA]\nQUESTION: {question}\nANSWER: {answer}"
            
            documents.append(Document(
                page_content=content,
                metadata=metadata
            ))
        
        return documents

    def _create_cli_qa_pairs(self, doc_data: Dict[str, Any], context_prefix: str) -> List[Tuple[str, str]]:
        """Create CLI-specific Q&A pairs with enhanced targeting."""
        qa_pairs = []
        content = doc_data.get('raw_text', '')
        title = doc_data.get('title', '')
        
        # CLI-specific pattern extraction
        cli_patterns = {
            'duplicate_data': r'duplicate.*?data.*?(?:in|of).*?(?:itential-cli|return|cli)',
            'netcommon_version': r'netcommon.*?(?:collection|version).*?(\d+\.\d+\.\d+)',
            'ansible_galaxy': r'ansible-galaxy.*?collection.*?(?:install|list|update|upgrade)',
            'cli_troubleshooting': r'(?:troubleshoot|debug|fix).*?(?:cli|command\s+line)'
        }
        
        for pattern_name, pattern in cli_patterns.items():
            if re.search(pattern, content, re.IGNORECASE | re.DOTALL):
                if pattern_name == 'duplicate_data':
                    qa_pairs.extend([
                        ("itential-cli role is showing duplicate data. how do i troubleshoot this?", 
                         f"CLI Troubleshooting: {content[:600]}..."),
                        ("CLI duplicate data troubleshooting steps",
                         f"For CLI duplicate data issues: {content[:500]}..."),
                        ("How to fix duplicate return data in itential-cli commands?",
                         f"CLI issue resolution: {content[:500]}..."),
                        ("itential-cli duplicate data netcommon collection",
                         f"Netcommon collection fix: {content[:500]}...")
                    ])
                
                elif pattern_name == 'netcommon_version':
                    versions = re.findall(r'\d+\.\d+\.\d+', content)
                    for version in versions[:2]:
                        qa_pairs.extend([
                            (f"ansible netcommon collection version {version} duplicate data",
                             f"Netcommon {version} information: {content[:400]}..."),
                            (f"What netcommon version fixes duplicate data? {version}",
                             f"Version {version} requirements: {content[:400]}...")
                        ])
                
                elif pattern_name == 'ansible_galaxy':
                    qa_pairs.extend([
                        ("ansible-galaxy collection install netcommon fix duplicate",
                         f"Collection installation: {content[:400]}..."),
                        ("How to upgrade ansible netcommon collection?",
                         f"Upgrade instructions: {content[:400]}...")
                    ])
        
        # Update processing stats
        self.processing_stats['cli_documents'] += 1
        
        return qa_pairs[:12]  # Limit but allow more for CLI content

    def _create_platform_qa_pairs(self, doc_data: Dict[str, Any], context_prefix: str) -> List[Tuple[str, str]]:
        """Create Platform-specific Q&A pairs."""
        qa_pairs = []
        content = doc_data.get('raw_text', '')
        title = doc_data.get('title', '')
        
        # Platform-specific patterns
        platform_patterns = {
            'event_deduplication': r'event.*?deduplication.*?(?:active|configuration)',
            'email_adapter': r'email.*?adapter.*?(?:trigger|duplicate)',
            'operations_manager': r'operations.*?manager.*?(?:trigger|event)',
            'unique_props': r'uniqueProps.*?(?:setting|field|messageId)'
        }
        
        for pattern_name, pattern in platform_patterns.items():
            if re.search(pattern, content, re.IGNORECASE):
                if pattern_name == 'event_deduplication':
                    qa_pairs.extend([
                        ("How to configure event deduplication in Itential Platform?",
                         f"Platform event deduplication: {content[:500]}..."),
                        ("Platform event service duplicate prevention",
                         f"Event service configuration: {content[:500]}..."),
                        ("PLATFORM: event deduplication configuration steps",
                         f"Based on {title}: {content[:500]}...")
                    ])
                elif pattern_name == 'email_adapter':
                    qa_pairs.extend([
                        ("PLATFORM: email adapter trigger duplicate jobs",
                         f"Email adapter configuration: {content[:400]}..."),
                        ("Operations Manager email trigger deduplication",
                         f"Email trigger setup: {content[:400]}...")
                    ])
        
        self.processing_stats['platform_documents'] += 1
        return qa_pairs[:8]

    def _create_version_qa_pairs(self, doc_data: Dict[str, Any], context_prefix: str) -> List[Tuple[str, str]]:
        """Create version-specific Q&A pairs."""
        qa_pairs = []
        content = doc_data.get('raw_text', '')
        title = doc_data.get('title', '')
        
        # Extract version information
        version_patterns = [
            (r'(?:IAP|iap)\s*([0-9]{4}\.[0-9]+)', 'IAP'),
            (r'(?:python|Python)\s*([0-9]+\.[0-9]+)', 'Python'),
            (r'(?:node|Node)\.?js\s*([0-9]+\.[0-9]+)', 'Node.js'),
            (r'(?:mongodb|MongoDB)\s*([0-9]+\.[0-9]+)', 'MongoDB')
        ]
        
        for pattern, product in version_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for version in matches[:2]:
                qa_pairs.extend([
                    (f"VERSION: What {product} version {version} requirements?",
                     f"{product} {version} requirements: {content[:400]}..."),
                    (f"COMPATIBILITY: {product} {version} system requirements",
                     f"Version {version} compatibility: {content[:400]}...")
                ])
        
        self.processing_stats['version_documents'] += 1
        return qa_pairs[:6]

    def _create_troubleshooting_qa_pairs(self, doc_data: Dict[str, Any], context_prefix: str) -> List[Tuple[str, str]]:
        """Create troubleshooting-specific Q&A pairs."""
        qa_pairs = []
        content = doc_data.get('raw_text', '')
        title = doc_data.get('title', '')
        
        # Extract troubleshooting content
        trouble_patterns = [
            r'(?:solution|fix|resolution)[:\-]?\s*([^.\n]+)',
            r'(?:step\s+\d+|first|then|next)[:\-]?\s*([^.\n]+)',
            r'(?:troubleshoot|debug)[:\-]?\s*([^.\n]+)'
        ]
        
        steps_found = []
        for pattern in trouble_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            steps_found.extend(matches[:3])
        
        if steps_found:
            qa_pairs.extend([
                (f"TROUBLESHOOT: How to resolve issues in {title}?",
                 f"Troubleshooting steps: {' | '.join(steps_found[:3])}"),
                (f"PROBLEM: Debugging steps for {title}",
                 f"Debug process: {content[:400]}...")
            ])
        
        self.processing_stats['troubleshooting_documents'] += 1
        return qa_pairs[:4]

    def _create_general_qa_pairs(self, doc_data: Dict[str, Any], context_prefix: str) -> List[Tuple[str, str]]:
        """Create general Q&A pairs."""
        qa_pairs = []
        content = doc_data.get('raw_text', '')
        title = doc_data.get('title', '')
        
        if len(content) > 200:
            qa_pairs.extend([
                (f"What is explained in {title}?", f"Overview: {content[:400]}..."),
                (f"Key information about {title}", f"Summary: {content[:400]}...")
            ])
        
        self.processing_stats['general_documents'] += 1
        return qa_pairs[:3]

    def create_optimized_vector_store(self) -> Optional[Chroma]:
        """Create optimized vector store with enhanced context awareness."""
        try:
            if not self.data_file.exists():
                raise FileNotFoundError(f"Data file {self.data_file} not found. Run enhanced scraper first.")

            logger.info("🔄 Loading documents for enhanced context-aware processing...")
            documents: List[Dict[str, Any]] = []
            
            with open(self.data_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            doc_data = json.loads(line)
                            documents.append(doc_data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping malformed JSON on line {line_num}: {e}")

            logger.info(f"📄 Loaded {len(documents)} documents")

            # Process documents with enhanced context awareness
            all_enhanced_documents: List[Document] = []
            domain_stats = defaultdict(int)
            
            for doc_idx, doc_data in enumerate(documents):
                try:
                    # Create context-aware chunks
                    chunk_docs = self.create_context_aware_chunks(doc_data)
                    all_enhanced_documents.extend(chunk_docs)
                    
                    # Create enhanced Q&A pairs
                    qa_docs = self.create_enhanced_qa_pairs(doc_data)
                    all_enhanced_documents.extend(qa_docs)
                    
                    # Track domain distribution
                    if chunk_docs:
                        domain = chunk_docs[0].metadata.get('domain', 'GENERAL')
                        domain_stats[domain] += len(chunk_docs) + len(qa_docs)
                        self.processing_stats['total_qa_pairs'] += len(qa_docs)
                    
                    if (doc_idx + 1) % 20 == 0:
                        logger.info(f"📊 Processed {doc_idx + 1}/{len(documents)} documents")
                        
                except Exception as e:
                    logger.warning(f"Error processing document {doc_idx}: {e}")
                    continue

            logger.info("📈 Enhanced domain distribution:")
            for domain, count in domain_stats.items():
                logger.info(f"   {domain}: {count} documents")
            
            logger.info("📊 Processing statistics:")
            for stat_name, count in self.processing_stats.items():
                logger.info(f"   {stat_name}: {count}")

            # Create vector store in prioritized batches
            logger.info("🚀 Creating enhanced context-aware vector database...")
            
            # Prioritize CLI and troubleshooting content
            cli_docs = [doc for doc in all_enhanced_documents 
                       if doc.metadata.get('domain') == 'CLI_TOOLS']
            platform_docs = [doc for doc in all_enhanced_documents 
                            if doc.metadata.get('domain') == 'PLATFORM_EVENTS']
            other_docs = [doc for doc in all_enhanced_documents 
                         if doc.metadata.get('domain') not in ['CLI_TOOLS', 'PLATFORM_EVENTS']]
            
            # Process in priority order
            all_docs_ordered = cli_docs + platform_docs + other_docs
            
            logger.info(f"📋 Processing order: CLI({len(cli_docs)}) + Platform({len(platform_docs)}) + Other({len(other_docs)})")
            
            batch_size = 25
            vector_store: Optional[Chroma] = None
            
            for i in range(0, len(all_docs_ordered), batch_size):
                batch = all_docs_ordered[i:i + batch_size]
                
                try:
                    if vector_store is None:
                        vector_store = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            persist_directory="./enhanced_context_chroma_db"
                        )
                        logger.info(f"✅ Created enhanced vector store with first batch of {len(batch)} documents")
                    else:
                        vector_store.add_documents(batch)
                        
                    logger.info(f"📊 Processed {min(i + batch_size, len(all_docs_ordered))}/{len(all_docs_ordered)} documents")
                    time.sleep(0.3)  # Brief pause between batches
                    
                except Exception as e:
                    logger.error(f"❌ Error processing batch {i//batch_size + 1}: {e}")
                    continue

            logger.info("🎉 Enhanced context-aware vector database created successfully!")
            
            # Test the enhanced system
            self._test_enhanced_context_system(vector_store)
            
            return vector_store

        except Exception as e:
            logger.error(f"❌ Error creating enhanced vector store: {str(e)}")
            raise

    def _test_enhanced_context_system(self, vector_store: Chroma):
        """Test the enhanced context-aware system."""
        logger.info("🧪 Testing enhanced context-aware system...")
        
        test_queries = [
            ("itential-cli role is showing duplicate data. how do i troubleshoot this", "CLI_TOOLS"),
            ("how to configure event deduplication in Operations Manager", "PLATFORM_EVENTS"),
            ("What Node.js version is required for IAP 2023.2?", "VERSION_DEPENDENCIES"),
            ("troubleshooting duplicate issues", "TROUBLESHOOTING")
        ]
        
        for query, expected_domain in test_queries:
            try:
                results = vector_store.similarity_search(query, k=5)
                if results:
                    found_domains = [doc.metadata.get('domain', 'UNKNOWN') for doc in results]
                    context_prefixes = [doc.metadata.get('context_prefix', 'NONE') for doc in results]
                    
                    logger.info(f"🔍 Query: '{query[:50]}...'")
                    logger.info(f"   Expected domain: {expected_domain}")
                    logger.info(f"   Found domains: {found_domains[:3]}")
                    logger.info(f"   Context prefixes: {context_prefixes[:3]}")
                    
                    # Check if we got the right domain in top results
                    if expected_domain in found_domains[:3]:
                        logger.info("   ✅ Enhanced context targeting: SUCCESS")
                    else:
                        logger.warning("   ⚠️  Enhanced context targeting: NEEDS_IMPROVEMENT")
                        
                    # Check for CLI troubleshooting specifically
                    if "cli" in query.lower() and "CLI_TROUBLESHOOTING" in context_prefixes[:2]:
                        logger.info("   ✅ CLI troubleshooting detection: SUCCESS")
                        
                else:
                    logger.warning(f"   ❌ No results for query: {query}")
                    
            except Exception as e:
                logger.error(f"   ❌ Test failed for query '{query}': {e}")

    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get detailed processing statistics."""
        return {
            'processing_stats': self.processing_stats,
            'domain_categories': list(self.domain_categories.keys()),
            'data_file': str(self.data_file),
            'embeddings_model': 'nomic-embed-text'
        }

def main():
    """Main function to create enhanced context-aware vector store."""
    embedder = EnhancedTechnicalDocumentationEmbedder("complete_technical_docs.jsonl")
    
    # Remove existing database
    import shutil
    db_path = Path("./enhanced_context_chroma_db")
    if db_path.exists():
        shutil.rmtree(db_path)
        logger.info("🗑️  Removed existing enhanced context vector database")
    
    # Create enhanced vector store
    vector_store = embedder.create_optimized_vector_store()
    
    if vector_store:
        logger.info("🎉 Enhanced Context-Aware RAG system ready!")
        logger.info("📂 Database path: ./enhanced_context_chroma_db")
        logger.info("")
        logger.info("📋 Update your chatbot configuration:")
        logger.info("   qa_db_path='./enhanced_context_chroma_db'")
        logger.info("")
        logger.info("🧪 Test CLI troubleshooting query:")
        logger.info("   'itential-cli role is showing duplicate data. how do i troubleshoot this'")
        logger.info("")
        logger.info("✅ Expected behavior:")
        logger.info("   • Should detect CLI_TOOLS domain")
        logger.info("   • Should NOT confuse with Platform event deduplication")
        logger.info("   • Should provide netcommon collection version solution")
        
        # Show processing statistics
        stats = embedder.get_processing_statistics()
        logger.info("\n📊 Final Processing Statistics:")
        for stat_name, count in stats['processing_stats'].items():
            logger.info(f"   {stat_name}: {count}")

if __name__ == "__main__":
    main()