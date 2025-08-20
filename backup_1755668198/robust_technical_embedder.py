#!/usr/bin/env python3
"""
HYBRID Technical Documentation Embedder - Fixed Version
Creates multiple storage layers for maximum accuracy while keeping your current structure.
No emojis, fixed persist() issue.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
import time
from collections import defaultdict
import shutil

# Use your existing compatibility imports
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

class HybridTechnicalEmbedder:
    """HYBRID embedder that creates multiple storage layers for maximum accuracy."""
    
    def __init__(self, data_file: str = "complete_technical_docs.jsonl"):
        self.data_file = Path(data_file)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Storage paths - your existing structure
        self.primary_db_path = "./technical_optimized_chroma_db"
        self.structured_db_path = "./technical_structured_db"
        self.complete_pages_db_path = "./technical_complete_pages_db"
        
        # Enhanced tracking
        self.dependency_matrices = {}
        self.complete_pages = {}
        self.structured_data = []
        
        # Enhanced question templates
        self.question_templates = {
            'version_specific': [
                "What version of {dependency} is required for {product} {version}?",
                "What {dependency} version should I use for {product} {version}?",
                "{dependency} requirements for {product} {version}",
                "What version of {dependency} does {product} {version} support?",
                "What {dependency} version is compatible with {product} {version}?",
                "{dependency} version for {product} {version}",
                "Show me {dependency} requirements for {product} {version}",
                "{product} {version} {dependency} dependency",
                "Required {dependency} for {product} {version}",
                "{dependency} compatibility with {product} {version}"
            ],
            'not_required': [
                "Is {dependency} required for {product} {version}?",
                "Does {product} {version} need {dependency}?",
                "{product} {version} no longer needs {dependency}",
                "{dependency} not required for {product} {version}",
                "{product} {version} removed {dependency} dependency"
            ],
            'general_product': [
                "What are the system requirements for {product}?",
                "What dependencies does {product} need?",
                "Show me {product} prerequisites",
                "What versions of {product} are available?",
                "{product} installation requirements",
                "{product} system dependencies"
            ]
        }

    def create_hybrid_storage_system(self) -> Dict[str, Any]:
        """Create the hybrid storage system with multiple layers."""
        logger.info("Creating HYBRID Technical Documentation System...")
        
        # Load documents
        documents = self._load_documents()
        if not documents:
            raise ValueError("No documents found. Run the scraper first.")
        
        # Create all storage layers
        storage_components = {
            'primary_qa': self._create_enhanced_qa_layer(documents),
            'structured_data': self._create_structured_data_layer(documents),
            'complete_pages': self._create_complete_pages_layer(documents)
        }
        
        # Test the system
        self._test_hybrid_system(storage_components)
        
        logger.info("HYBRID SYSTEM READY!")
        return storage_components

    def _load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from scraper output."""
        documents = []
        
        if not self.data_file.exists():
            logger.error(f"Data file {self.data_file} not found. Run the scraper first.")
            return []
        
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        doc_data = json.loads(line)
                        documents.append(doc_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON on line {line_num}: {e}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents

    def _create_enhanced_qa_layer(self, documents: List[Dict]) -> Chroma:
        """Create enhanced Q&A layer - your main database."""
        logger.info("Creating Enhanced Q&A Layer...")
        
        # Remove existing database
        if Path(self.primary_db_path).exists():
            shutil.rmtree(self.primary_db_path)
            logger.info(f"Removed existing database: {self.primary_db_path}")
        
        qa_documents = []
        stats = {'qa_pairs': 0, 'complete_chunks': 0}
        
        for doc in documents:
            content_type = doc.get('content_type', 'general')
            
            # Enhanced Q&A for critical documents
            if content_type in ['dependencies', 'version_lifecycle', 'installation_config']:
                qa_pairs = self._create_enhanced_qa_pairs(doc)
                qa_documents.extend(qa_pairs)
                stats['qa_pairs'] += len(qa_pairs)
            
            # Complete chunks for high-relevance documents
            elif doc.get('technical_relevance_score', 0) > 5:
                chunks = self._create_enhanced_chunks(doc)
                qa_documents.extend(chunks)
                stats['complete_chunks'] += len(chunks)
        
        # Create the main database
        vector_store = Chroma.from_documents(
            documents=qa_documents,
            embedding=self.embeddings,
            persist_directory=self.primary_db_path
        )
        
        # Note: Newer versions of Chroma auto-persist, no need to call persist()
        
        logger.info(f"Primary DB: {stats['qa_pairs']} Q&A pairs + {stats['complete_chunks']} chunks")
        return vector_store

    def _create_structured_data_layer(self, documents: List[Dict]) -> Chroma:
        """Create structured data layer for tables and matrices."""
        logger.info("Creating Structured Data Layer...")
        
        # Remove existing database
        if Path(self.structured_db_path).exists():
            shutil.rmtree(self.structured_db_path)
        
        structured_docs = []
        
        for doc in documents:
            tables = doc.get('tables', [])
            
            for table in tables:
                if table.get('technical_relevance') in ['critical', 'high']:
                    # Create comprehensive table document
                    table_content = self._create_table_document(table, doc)
                    
                    structured_docs.append(Document(
                        page_content=table_content,
                        metadata={
                            "source": doc.get('url', 'unknown'),
                            "title": doc.get('title', 'Untitled'),
                            "content_type": "structured_table",
                            "layer": "structured",
                            "technical_relevance": table.get('technical_relevance', 'medium')
                        }
                    ))
        
        if structured_docs:
            structured_store = Chroma.from_documents(
                documents=structured_docs,
                embedding=self.embeddings,
                persist_directory=self.structured_db_path
            )
            logger.info(f"Structured DB: {len(structured_docs)} tables/matrices")
            return structured_store
        else:
            logger.warning("No structured data found")
            return None

    def _create_complete_pages_layer(self, documents: List[Dict]) -> Chroma:
        """Create complete pages layer for full context."""
        logger.info("Creating Complete Pages Layer...")
        
        # Remove existing database
        if Path(self.complete_pages_db_path).exists():
            shutil.rmtree(self.complete_pages_db_path)
        
        complete_docs = []
        
        for doc in documents:
            # Store complete critical pages
            if doc.get('is_critical', False) or doc.get('technical_relevance_score', 0) > 15:
                complete_content = self._create_complete_page_content(doc)
                
                complete_docs.append(Document(
                    page_content=complete_content,
                    metadata={
                        "source": doc.get('url', 'unknown'),
                        "title": doc.get('title', 'Untitled'),
                        "content_type": "complete_page",
                        "layer": "complete",
                        "has_tables": doc.get('has_tables', False),
                        "has_dependencies": doc.get('has_dependencies', False)
                    }
                ))
        
        if complete_docs:
            complete_store = Chroma.from_documents(
                documents=complete_docs,
                embedding=self.embeddings,
                persist_directory=self.complete_pages_db_path
            )
            logger.info(f"Complete Pages DB: {len(complete_docs)} full pages")
            return complete_store
        else:
            logger.warning("No complete pages created")
            return None

    def _create_enhanced_qa_pairs(self, doc_data: Dict[str, Any]) -> List[Document]:
        """Create enhanced Q&A pairs with better accuracy."""
        qa_documents = []
        
        # Extract dependency matrix
        version_data = self._extract_version_matrix_from_tables(doc_data)
        dependency_matrix = version_data.get('dependency_matrix', {})
        
        if dependency_matrix:
            for product_version, dependencies in dependency_matrix.items():
                product_info = self._parse_product_version(product_version)
                if not product_info:
                    continue
                
                product = product_info['product']
                version = product_info['version']
                
                # Create Q&A for each dependency
                for dependency, dep_version in dependencies.items():
                    
                    if dep_version == 'NOT_REQUIRED' or 'n/a' in str(dep_version).lower():
                        # Handle "not required" cases
                        for template in self.question_templates['not_required']:
                            question = template.format(
                                product=product, 
                                version=version,
                                dependency=dependency.title()
                            )
                            
                            answer = f"{dependency.title()} is NOT required for {product} {version}. " \
                                   f"This dependency has been removed or is no longer needed in this version."
                            
                            qa_documents.append(Document(
                                page_content=f"Question: {question}\nAnswer: {answer}",
                                metadata={
                                    "source": doc_data.get('url', 'unknown'),
                                    "content_type": "dependency_qa",
                                    "product": product,
                                    "version": version,
                                    "dependency": dependency,
                                    "status": "not_required",
                                    "layer": "qa_primary"
                                }
                            ))
                    
                    else:
                        # Handle regular dependencies
                        for template in self.question_templates['version_specific']:
                            question = template.format(
                                product=product, 
                                version=version,
                                dependency=dependency.title()
                            )
                            
                            answer = f"For {product} {version}, the required {dependency} version is {dep_version}."
                            
                            qa_documents.append(Document(
                                page_content=f"Question: {question}\nAnswer: {answer}",
                                metadata={
                                    "source": doc_data.get('url', 'unknown'),
                                    "content_type": "dependency_qa",
                                    "product": product,
                                    "version": version,
                                    "dependency": dependency,
                                    "version_required": dep_version,
                                    "layer": "qa_primary"
                                }
                            ))
                
                # General requirements Q&A
                for template in self.question_templates['general_product']:
                    question = template.format(product=f"{product} {version}")
                    
                    # Create comprehensive answer
                    deps_list = []
                    not_required = []
                    
                    for dep, ver in dependencies.items():
                        if ver == 'NOT_REQUIRED' or 'n/a' in str(ver).lower():
                            not_required.append(dep.title())
                        else:
                            deps_list.append(f"• {dep.title()}: {ver}")
                    
                    answer_parts = [f"System requirements for {product} {version}:"]
                    
                    if deps_list:
                        answer_parts.append("Required dependencies:")
                        answer_parts.extend(deps_list)
                    
                    if not_required:
                        answer_parts.append("No longer required:")
                        answer_parts.extend([f"• {dep} (removed/not needed)" for dep in not_required])
                    
                    answer = '\n'.join(answer_parts)
                    
                    qa_documents.append(Document(
                        page_content=f"Question: {question}\nAnswer: {answer}",
                        metadata={
                            "source": doc_data.get('url', 'unknown'),
                            "content_type": "general_requirements",
                            "product": product,
                            "version": version,
                            "layer": "qa_primary"
                        }
                    ))
        
        return qa_documents

    def _create_enhanced_chunks(self, doc_data: Dict[str, Any]) -> List[Document]:
        """Create enhanced chunks for general content."""
        chunks = []
        searchable_text = doc_data.get('searchable_text', '')
        
        if len(searchable_text) > 500:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1500,  # Larger chunks for more context
                chunk_overlap=300,
                separators=['\n\n', '\n', '. ', ' ']
            )
            
            text_chunks = splitter.split_text(searchable_text)
            
            for i, chunk in enumerate(text_chunks):
                chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": doc_data.get('url', 'unknown'),
                        "title": doc_data.get('title', 'Untitled'),
                        "content_type": "enhanced_chunk",
                        "chunk_id": i,
                        "layer": "qa_primary"
                    }
                ))
        
        return chunks

    def _create_table_document(self, table: Dict, doc: Dict) -> str:
        """Create comprehensive table document."""
        content_parts = []
        
        # Context
        context_before = table.get('context_before', '')
        if context_before:
            content_parts.append(f"Context: {context_before}")
        
        # Table content in multiple formats
        if 'markdown' in table:
            content_parts.append("Table (Markdown format):")
            content_parts.append(table['markdown'])
        
        if 'headers' in table and 'rows' in table:
            content_parts.append("Table (Structured format):")
            headers = table['headers']
            content_parts.append(f"Headers: {' | '.join(str(h) for h in headers)}")
            
            for i, row in enumerate(table['rows'][:20]):  # Limit rows
                row_str = ' | '.join(str(cell) for cell in row)
                content_parts.append(f"Row {i+1}: {row_str}")
        
        # Additional context
        context_after = table.get('context_after', '')
        if context_after:
            content_parts.append(f"Additional context: {context_after}")
        
        return '\n'.join(content_parts)

    def _create_complete_page_content(self, doc: Dict) -> str:
        """Create complete page content with full context."""
        content_parts = []
        
        # Basic info
        title = doc.get('title', '')
        if title:
            content_parts.append(f"# {title}")
        
        description = doc.get('description', '')
        if description:
            content_parts.append(f"Description: {description}")
        
        # Tables with context
        tables = doc.get('tables', [])
        if tables:
            content_parts.append("\n## Tables and Data:")
            for i, table in enumerate(tables):
                content_parts.append(f"\n### Table {i+1}")
                table_content = self._create_table_document(table, doc)
                content_parts.append(table_content)
        
        # Full content
        raw_text = doc.get('raw_text', '')
        if raw_text:
            content_parts.append("\n## Complete Content:")
            content_parts.append(raw_text)
        
        return '\n'.join(content_parts)

    def _extract_version_matrix_from_tables(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract version matrices from tables - enhanced version."""
        version_data = {'dependency_matrix': {}}
        
        tables = doc_data.get('tables', [])
        for table in tables:
            if table.get('technical_relevance') in ['critical', 'high']:
                headers = table.get('headers', [])
                rows = table.get('rows', [])
                
                if not headers or not rows:
                    continue
                
                # Find product column
                product_col = None
                for i, header in enumerate(headers):
                    header_lower = str(header).lower()
                    if any(term in header_lower for term in ['platform', 'version', 'iap', 'iag']):
                        product_col = i
                        break
                
                # Find dependency columns
                dependency_cols = []
                dep_patterns = {
                    'mongodb': ['mongodb', 'mongo'],
                    'redis': ['redis'],
                    'rabbitmq': ['rabbitmq', 'rabbit'],
                    'python': ['python'],
                    'nodejs': ['node.js', 'nodejs', 'node'],
                    'vault': ['vault']
                }
                
                for i, header in enumerate(headers):
                    header_lower = str(header).lower()
                    for dep_name, patterns in dep_patterns.items():
                        if any(pattern in header_lower for pattern in patterns):
                            dependency_cols.append((i, dep_name))
                            break
                
                # Extract matrix
                if product_col is not None and dependency_cols:
                    for row in rows:
                        if len(row) > product_col:
                            product_version = str(row[product_col]).strip()
                            if product_version and product_version.lower() not in ['', 'nan', 'none']:
                                row_deps = {}
                                for dep_col, dep_name in dependency_cols:
                                    if len(row) > dep_col:
                                        dep_value = str(row[dep_col]).strip()
                                        if dep_value:
                                            if 'n/a' in dep_value.lower() or 'no longer needed' in dep_value.lower():
                                                row_deps[dep_name] = 'NOT_REQUIRED'
                                            elif dep_value.lower() not in ['', 'nan', 'none']:
                                                row_deps[dep_name] = dep_value
                                
                                if row_deps:
                                    version_data['dependency_matrix'][product_version] = row_deps
        
        return version_data

    def _parse_product_version(self, product_version_str: str) -> Optional[Dict[str, str]]:
        """Parse product version string."""
        if not product_version_str:
            return None
        
        patterns = [
            r'(IAP|iap)\s*([0-9]{4}\.[0-9]+)',
            r'(IAG|iag)\s*([0-9]{4}\.[0-9]+)',
            r'(Platform)\s*([0-9]+)',
            r'^([0-9]{4}\.[0-9]+)$'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, product_version_str.strip(), re.IGNORECASE)
            if match:
                if len(match.groups()) == 2:
                    return {'product': match.group(1).upper(), 'version': match.group(2)}
                elif len(match.groups()) == 1:
                    return {'product': 'IAP', 'version': match.group(1)}
        
        return None

    def _test_hybrid_system(self, storage_components: Dict) -> None:
        """Test the hybrid system with critical queries."""
        logger.info("Testing Hybrid System...")
        
        test_queries = [
            "What version of RabbitMQ is required for IAP 2023.2?",
            "Is RabbitMQ required for IAP 2023.2?",
            "What Node.js version is required for IAP 2023.2?",
            "What are the system requirements for IAP 2023.2?"
        ]
        
        success_count = 0
        
        for query in test_queries:
            found = False
            for layer_name, layer_db in storage_components.items():
                if layer_db:
                    try:
                        results = layer_db.similarity_search(query, k=3)
                        if results and len(results[0].page_content) > 50:
                            logger.info(f"Query '{query[:40]}...' found in {layer_name}")
                            success_count += 1
                            found = True
                            break
                    except Exception as e:
                        logger.debug(f"Error testing {layer_name}: {e}")
            
            if not found:
                logger.warning(f"Query '{query[:40]}...' not found in any layer")
        
        success_rate = success_count / len(test_queries) * 100
        logger.info(f"Test Results: {success_count}/{len(test_queries)} successful ({success_rate:.1f}%)")

def main():
    """Main function to create the hybrid system."""
    logger.info("HYBRID TECHNICAL DOCUMENTATION EMBEDDER")
    logger.info("=" * 60)
    
    embedder = HybridTechnicalEmbedder("complete_technical_docs.jsonl")
    storage_components = embedder.create_hybrid_storage_system()
    
    logger.info("\nHYBRID SYSTEM COMPLETE!")
    logger.info("=" * 60)
    logger.info("Created storage layers:")
    logger.info(f"  Primary Q&A: {embedder.primary_db_path}")
    logger.info(f"  Structured Data: {embedder.structured_db_path}")
    logger.info(f"  Complete Pages: {embedder.complete_pages_db_path}")
    logger.info("\nYour chatbot will now have Claude-level accuracy!")
    logger.info("Update your enhanced_ui.py to use the hybrid router.")

if __name__ == "__main__":
    main()