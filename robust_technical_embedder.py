#!/usr/bin/env python3
"""
Technical Documentation Embedder
Creates optimized vector database for fast technical queries about versions and dependencies.
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

class TechnicalDocumentationEmbedder:
    """Specialized embedder for technical documentation with focus on versions and dependencies."""
    
    def __init__(self, data_file: str = "complete_technical_docs.jsonl"):
        self.data_file = Path(data_file)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Track extracted technical data
        self.version_matrix: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.dependency_relationships: Dict[str, Dict] = {}
        self.product_info: Dict[str, Dict] = {}
        
        # Technical question templates for comprehensive coverage
        self.question_templates = {
            'version_specific': [
                "What version of {dependency} is required for {product} {version}?",
                "What {dependency} version should I use for {product} {version}?",
                "{dependency} requirements for {product} {version}",
                "What version of {dependency} does {product} {version} support?",
                "What {dependency} version is compatible with {product} {version}?",
                "{dependency} version for {product} {version}",
                "Show me {dependency} requirements for {product} {version}"
            ],
            'general_product': [
                "What are the system requirements for {product}?",
                "What dependencies does {product} need?",
                "Show me {product} prerequisites",
                "What versions of {product} are available?",
                "List {product} requirements",
                "Dependencies for {product}",
                "{product} installation requirements"
            ],
            'dependency_focused': [
                "What {dependency} versions are supported?",
                "Show me all {dependency} requirements",
                "What products use {dependency}?",
                "Which {product} versions support {dependency}?",
                "What are the {dependency} version requirements?"
            ],
            'comparison': [
                "Compare {dependency} requirements across {product} versions",
                "What changed in {dependency} requirements for {product}?",
                "Differences between {product} {version1} and {version2} {dependency} requirements"
            ]
        }

    def extract_version_matrix_from_tables(self, doc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract comprehensive version matrix from tables."""
        version_data = {
            'versions_found': {},
            'dependency_matrix': {},
            'tables_processed': 0
        }
        
        tables = doc_data.get('tables', [])
        
        for table in tables:
            if table.get('technical_relevance') in ['critical', 'high']:
                version_data['tables_processed'] += 1
                
                # Extract from markdown table
                markdown = table.get('markdown', '')
                if markdown:
                    parsed_data = self._parse_dependency_table(markdown)
                    if parsed_data:
                        version_data['dependency_matrix'].update(parsed_data)
                
                # Extract from table metadata
                extracted_versions = table.get('extracted_versions', [])
                for version_info in extracted_versions:
                    product = version_info.get('product', '').lower()
                    version = version_info.get('version', '')
                    if product and version:
                        if product not in version_data['versions_found']:
                            version_data['versions_found'][product] = set()
                        version_data['versions_found'][product].add(version)
        
        return version_data

    def _parse_dependency_table(self, markdown_table: str) -> Optional[Dict[str, Dict]]:
        """Parse dependency table to extract version requirements."""
        try:
            lines = markdown_table.strip().split('\n')
            if len(lines) < 3:  # Need header, separator, and at least one data row
                return None
            
            # Parse header
            headers = [h.strip().lower() for h in lines[0].split('|') if h.strip()]
            if not headers:
                return None
            
            # Find column indices for important data
            version_col = None
            product_col = None
            dependency_cols = {}
            
            for i, header in enumerate(headers):
                header_clean = header.replace(':', '').strip()
                if any(term in header_clean for term in ['version', 'release', 'iap', 'iag', 'platform']):
                    if not version_col:  # Take first version column
                        version_col = i
                        product_col = i
                elif any(dep in header_clean for dep in ['python', 'node', 'mongo', 'redis', 'rabbit']):
                    dependency_cols[header_clean] = i
            
            if version_col is None:
                return None
            
            # Parse data rows
            dependency_matrix = {}
            
            for line in lines[2:]:  # Skip header and separator
                if '|' not in line:
                    continue
                    
                cells = [cell.strip() for cell in line.split('|') if cell.strip()]
                if len(cells) <= max(version_col, max(dependency_cols.values()) if dependency_cols else 0):
                    continue
                
                product_version = cells[version_col] if version_col < len(cells) else ''
                if not product_version or product_version in ['---', '']:
                    continue
                
                # Clean product version
                product_version = self._clean_version_string(product_version)
                
                if product_version:
                    dependency_matrix[product_version] = {}
                    
                    # Extract dependency versions
                    for dep_name, col_idx in dependency_cols.items():
                        if col_idx < len(cells):
                            dep_version = self._clean_version_string(cells[col_idx])
                            if dep_version:
                                dependency_matrix[product_version][dep_name] = dep_version
            
            return dependency_matrix if dependency_matrix else None
            
        except Exception as e:
            logger.warning(f"Error parsing dependency table: {e}")
            return None

    def _clean_version_string(self, version_str: str) -> str:
        """Clean and normalize version strings."""
        if not version_str or version_str.strip() in ['---', '', 'N/A', 'TBD']:
            return ''
        
        # Remove common prefixes/suffixes
        cleaned = version_str.strip()
        cleaned = re.sub(r'^[>=<~\^]+', '', cleaned)  # Remove version operators
        cleaned = re.sub(r'\s*\([^)]*\)', '', cleaned)  # Remove parenthetical notes
        cleaned = cleaned.replace('*', '').strip()
        
        return cleaned if cleaned else ''

    def create_comprehensive_qa_pairs(self, doc_data: Dict[str, Any]) -> List[Document]:
        """Create comprehensive Q&A pairs for all technical scenarios."""
        qa_documents: List[Document] = []
        
        url = doc_data.get('url', 'unknown')
        title = doc_data.get('title', 'Untitled')
        content_type = doc_data.get('content_type', 'general')
        
        # Extract version matrix
        version_data = self.extract_version_matrix_from_tables(doc_data)
        dependency_matrix = version_data.get('dependency_matrix', {})
        
        if dependency_matrix:
            logger.info(f"Processing dependency matrix with {len(dependency_matrix)} product versions")
            
            # Create Q&A pairs for each product version and dependency combination
            for product_version, dependencies in dependency_matrix.items():
                
                # Parse product and version
                product_info = self._parse_product_version(product_version)
                if not product_info:
                    continue
                
                product = product_info['product']
                version = product_info['version']
                
                # Create version-specific Q&A pairs for each dependency
                for dependency, dep_version in dependencies.items():
                    if not dep_version:
                        continue
                    
                    # Generate multiple question variations
                    for template in self.question_templates['version_specific']:
                        question = template.format(
                            dependency=dependency.title(),
                            product=product,
                            version=version
                        )
                        
                        # Create comprehensive answer
                        answer = self._create_dependency_answer(
                            product, version, dependency, dep_version, dependency_matrix
                        )
                        
                        qa_content = f"Question: {question}\n\nAnswer: {answer}"
                        
                        qa_documents.append(Document(
                            page_content=qa_content,
                            metadata={
                                "source": url,
                                "title": title,
                                "content_type": "technical_qa",
                                "question": question,
                                "product": product,
                                "product_version": version,
                                "dependency": dependency,
                                "dependency_version": dep_version,
                                "priority": "critical",
                                "qa_type": "version_specific"
                            }
                        ))
                
                # Create general product questions
                for template in self.question_templates['general_product']:
                    question = template.format(product=f"{product} {version}")
                    
                    answer = self._create_general_requirements_answer(
                        product, version, dependencies, dependency_matrix
                    )
                    
                    qa_content = f"Question: {question}\n\nAnswer: {answer}"
                    
                    qa_documents.append(Document(
                        page_content=qa_content,
                        metadata={
                            "source": url,
                            "title": title,
                            "content_type": "technical_qa",
                            "question": question,
                            "product": product,
                            "product_version": version,
                            "priority": "high",
                            "qa_type": "general_requirements"
                        }
                    ))
        
        # Handle version lifecycle content
        if content_type == 'version_lifecycle':
            extracted_versions = doc_data.get('extracted_versions', {})
            
            for product, versions in extracted_versions.items():
                version_list = list(versions) if isinstance(versions, (list, set)) else [versions]
                
                # Create version listing Q&A pairs
                version_questions = [
                    f"What versions of {product.upper()} are available?",
                    f"List all {product.upper()} versions",
                    f"Show me {product.upper()} releases",
                    f"What {product.upper()} versions are supported?",
                    f"Which {product.upper()} versions can I use?"
                ]
                
                for question in version_questions:
                    answer = f"Available {product.upper()} versions include: {', '.join(sorted(version_list))}. " \
                            f"For specific version details and support information, refer to the official documentation."
                    
                    qa_content = f"Question: {question}\n\nAnswer: {answer}"
                    
                    qa_documents.append(Document(
                        page_content=qa_content,
                        metadata={
                            "source": url,
                            "title": title,
                            "content_type": "version_qa",
                            "question": question,
                            "product": product,
                            "available_versions": version_list,
                            "priority": "critical",
                            "qa_type": "version_listing"
                        }
                    ))
        
        return qa_documents

    def _parse_product_version(self, product_version_str: str) -> Optional[Dict[str, str]]:
        """Parse product and version from combined string."""
        # Common patterns
        patterns = [
            r'(IAP|iap)\s*([0-9]{4}\.[0-9]+(?:\.[0-9]+)?)',
            r'(IAG|iag)\s*([0-9]{4}\.[0-9]+(?:\.[0-9]+)?)',
            r'(Platform|platform)\s*([0-9]+(?:\.[0-9]+)?)',
            r'([A-Za-z\s]+)\s+([0-9]+\.[0-9]+(?:\.[0-9]+)?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, product_version_str, re.IGNORECASE)
            if match:
                product = match.group(1).strip().upper()
                version = match.group(2).strip()
                return {'product': product, 'version': version}
        
        return None

    def _create_dependency_answer(self, product: str, version: str, dependency: str, 
                                 dep_version: str, full_matrix: Dict) -> str:
        """Create comprehensive dependency-specific answer."""
        answer_parts = [
            f"For {product} {version}, the required {dependency} version is {dep_version}."
        ]
        
        # Add context from other versions if available
        other_versions = []
        for prod_ver, deps in full_matrix.items():
            if dependency in deps and prod_ver != f"{product} {version}":
                other_versions.append(f"{prod_ver}: {deps[dependency]}")
        
        if other_versions:
            answer_parts.append(f"\nFor comparison, other versions:")
            answer_parts.extend([f"• {ver}" for ver in other_versions[:3]])
        
        # Add installation/configuration hint
        if dependency.lower() in ['python', 'node.js', 'nodejs']:
            answer_parts.append(f"\nEnsure you have {dependency} {dep_version} or compatible version installed before proceeding with {product} {version} installation.")
        elif dependency.lower() in ['mongodb', 'redis', 'rabbitmq']:
            answer_parts.append(f"\nThis {dependency} version should be configured and running before starting {product} {version}.")
        
        return '\n'.join(answer_parts)

    def _create_general_requirements_answer(self, product: str, version: str, 
                                          dependencies: Dict, full_matrix: Dict) -> str:
        """Create general requirements answer."""
        answer_parts = [
            f"System requirements for {product} {version}:"
        ]
        
        # List all dependencies
        dep_list = []
        for dep_name, dep_version in dependencies.items():
            dep_list.append(f"• {dep_name.title()}: {dep_version}")
        
        if dep_list:
            answer_parts.extend(dep_list)
        else:
            answer_parts.append("• No specific dependency versions documented in this source")
        
        answer_parts.append(f"\nEnsure all dependencies are installed and configured before deploying {product} {version}.")
        
        return '\n'.join(answer_parts)

    def create_optimized_vector_store(self) -> Optional[Chroma]:
        """Create optimized vector store for fast technical queries."""
        try:
            if not self.data_file.exists():
                raise FileNotFoundError(f"Data file {self.data_file} not found. Run technical scraper first.")

            logger.info("Loading comprehensive technical documentation...")
            documents: List[Dict[str, Any]] = []
            
            with open(self.data_file, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if line.strip():
                        try:
                            doc_data = json.loads(line)
                            documents.append(doc_data)
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping malformed JSON on line {line_num}: {e}")

            logger.info(f"Loaded {len(documents)} technical documents")

            # Process documents to create comprehensive Q&A pairs
            logger.info("Creating comprehensive technical Q&A pairs...")
            all_qa_documents: List[Document] = []
            
            technical_stats = {
                'dependency_docs': 0,
                'version_docs': 0, 
                'qa_pairs_created': 0,
                'products_found': set(),
                'dependencies_found': set()
            }
            
            for doc_idx, doc_data in enumerate(documents):
                try:
                    content_type = doc_data.get('content_type', 'general')
                    
                    # Process technical documents
                    if content_type in ['dependencies', 'version_lifecycle', 'installation_config']:
                        qa_docs = self.create_comprehensive_qa_pairs(doc_data)
                        all_qa_documents.extend(qa_docs)
                        
                        # Track statistics
                        if content_type == 'dependencies':
                            technical_stats['dependency_docs'] += 1
                        elif content_type == 'version_lifecycle':
                            technical_stats['version_docs'] += 1
                        
                        technical_stats['qa_pairs_created'] += len(qa_docs)
                        
                        # Track products and dependencies found
                        extracted_versions = doc_data.get('extracted_versions', {})
                        for product in extracted_versions.keys():
                            technical_stats['products_found'].add(product)
                        
                        dependency_info = doc_data.get('dependency_info', {})
                        for dep in dependency_info.keys():
                            technical_stats['dependencies_found'].add(dep)
                        
                        if len(qa_docs) > 0:
                            logger.info(f"Created {len(qa_docs)} Q&A pairs from {doc_data.get('title', 'Unknown')[:50]}")
                    
                    # Also create general content chunks for comprehensive coverage
                    elif doc_data.get('technical_relevance_score', 0) > 5.0:
                        general_chunks = self._create_general_technical_chunks(doc_data)
                        all_qa_documents.extend(general_chunks)
                    
                    if len(all_qa_documents) % 100 == 0:
                        logger.info(f"Processed {len(all_qa_documents)} Q&A pairs so far...")
                        
                except Exception as e:
                    logger.error(f"Error processing document {doc_idx}: {e}")
                    continue

            if not all_qa_documents:
                raise ValueError("No Q&A pairs created from technical documents")

            # Log comprehensive statistics
            logger.info("Technical Processing Summary:")
            logger.info(f"  📊 Total Q&A pairs created: {technical_stats['qa_pairs_created']}")
            logger.info(f"  📋 Dependency documents: {technical_stats['dependency_docs']}")
            logger.info(f"  📅 Version documents: {technical_stats['version_docs']}")
            logger.info(f"  🏷️ Products found: {sorted(technical_stats['products_found'])}")
            logger.info(f"  🔧 Dependencies found: {sorted(technical_stats['dependencies_found'])}")

            # Prioritize Q&A pairs for optimal retrieval
            critical_qa = [doc for doc in all_qa_documents 
                          if doc.metadata.get('priority') == 'critical']
            high_qa = [doc for doc in all_qa_documents 
                      if doc.metadata.get('priority') == 'high']
            other_qa = [doc for doc in all_qa_documents 
                       if doc.metadata.get('priority') not in ['critical', 'high']]
            
            logger.info(f"Q&A Priority Distribution:")
            logger.info(f"  🚨 Critical: {len(critical_qa)} pairs")
            logger.info(f"  ⚡ High: {len(high_qa)} pairs")
            logger.info(f"  📄 Other: {len(other_qa)} pairs")

            # Create optimized vector store
            logger.info("Creating optimized technical vector database...")
            
            batch_size = 25
            vector_store: Optional[Chroma] = None
            
            # Process in priority order
            all_qa_ordered = critical_qa + high_qa + other_qa
            
            for i in range(0, len(all_qa_ordered), batch_size):
                batch = all_qa_ordered[i:i + batch_size]
                
                try:
                    if vector_store is None:
                        vector_store = Chroma.from_documents(
                            documents=batch,
                            embedding=self.embeddings,
                            persist_directory="./technical_optimized_chroma_db"
                        )
                        logger.info(f"Created technical vector store with first batch of {len(batch)} Q&A pairs")
                    else:
                        vector_store.add_documents(batch)
                        
                    logger.info(f"Processed {min(i + batch_size, len(all_qa_ordered))}/{len(all_qa_ordered)} Q&A pairs")
                    time.sleep(0.3)  # Brief pause between batches
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    continue

            logger.info("✅ Technical optimized vector database created successfully!")
            
            # Comprehensive testing
            logger.info("🧪 Testing technical query performance...")
            test_results = self._test_technical_queries(vector_store)
            
            return vector_store

        except Exception as e:
            logger.error(f"Error creating technical vector store: {str(e)}")
            raise

    def _create_general_technical_chunks(self, doc_data: Dict[str, Any]) -> List[Document]:
        """Create general technical chunks for comprehensive coverage."""
        chunks = []
        
        url = doc_data.get('url', 'unknown')
        title = doc_data.get('title', 'Untitled')
        searchable_text = doc_data.get('searchable_text', '')
        
        if len(searchable_text) > 1000:
            # Split into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=800,
                chunk_overlap=100
            )
            
            text_chunks = splitter.split_text(searchable_text)
            
            for i, chunk in enumerate(text_chunks):
                chunks.append(Document(
                    page_content=chunk,
                    metadata={
                        "source": url,
                        "title": title,
                        "content_type": "technical_content",
                        "chunk_id": i,
                        "priority": "medium"
                    }
                ))
        
        return chunks

    def _test_technical_queries(self, vector_store: Chroma) -> Dict[str, Any]:
        """Test the vector store with technical queries."""
        test_queries = [
            # Version-specific queries
            "What version of Node.js is required for IAP 2023.2?",
            "Python requirements for IAP 2023.1",
            "What MongoDB version does IAP 2023.2 need?",
            "Redis version for IAG 2023.1",
            
            # General product queries
            "What are the system requirements for IAP 2023.2?",
            "Dependencies for Platform 6",
            "Show me IAP 2023.1 prerequisites",
            
            # Version listing queries
            "What versions of IAP are available?",
            "List all IAG versions",
            "Show me Platform versions"
        ]
        
        results = {}
        
        for query in test_queries:
            try:
                search_results = vector_store.similarity_search_with_score(query, k=3)
                
                if search_results:
                    best_result = search_results[0]
                    doc, score = best_result
                    
                    # Check if it's a relevant technical result
                    metadata = doc.metadata
                    is_technical = metadata.get('content_type') in ['technical_qa', 'version_qa']
                    has_product = 'product' in metadata
                    has_version = 'product_version' in metadata or 'available_versions' in metadata
                    
                    results[query] = {
                        'score': score,
                        'is_technical': is_technical,
                        'has_product_info': has_product,
                        'has_version_info': has_version,
                        'content_preview': doc.page_content[:100] + "...",
                        'metadata': {k: v for k, v in metadata.items() if k in ['product', 'product_version', 'dependency']}
                    }
                    
                    logger.info(f"✅ '{query}' -> Score: {score:.3f}, Technical: {is_technical}, Product: {has_product}")
                else:
                    results[query] = {'score': 0, 'found': False}
                    logger.warning(f"❌ '{query}' -> No results found")
                    
            except Exception as e:
                logger.error(f"Error testing query '{query}': {e}")
                results[query] = {'error': str(e)}
        
        # Summary statistics
        successful_queries = sum(1 for r in results.values() if r.get('score', 0) > 0)
        technical_queries = sum(1 for r in results.values() if r.get('is_technical', False))
        
        logger.info(f"🧪 Test Results Summary:")
        logger.info(f"  ✅ Successful queries: {successful_queries}/{len(test_queries)}")
        logger.info(f"  🎯 Technical responses: {technical_queries}/{len(test_queries)}")
        
        return results

def main() -> None:
    """Main function to create technical optimized vector store."""
    embedder = TechnicalDocumentationEmbedder("complete_technical_docs.jsonl")
    
    # Remove existing database
    import shutil
    db_path = Path("./technical_optimized_chroma_db")
    if db_path.exists():
        shutil.rmtree(db_path)
        logger.info("Removed existing technical vector database")
    
    # Create optimized vector store
    vector_store = embedder.create_optimized_vector_store()
    
    if vector_store:
        logger.info("🎉 Technical optimized RAG system ready!")
        logger.info("Database path: ./technical_optimized_chroma_db")
        logger.info("")
        logger.info("📋 Update your chatbot configuration:")
        logger.info("   qa_db_path='./technical_optimized_chroma_db'")
        logger.info("")
        logger.info("🧪 Test queries:")
        logger.info("   • 'What Node.js version is required for IAP 2023.2?'")
        logger.info("   • 'Python requirements for IAP 2023.1'")
        logger.info("   • 'What versions of IAP are available?'")
        logger.info("   • 'MongoDB version for Platform 6'")

if __name__ == "__main__":
    main()