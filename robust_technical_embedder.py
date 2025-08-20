#!/usr/bin/env python3
"""
Comprehensive Technical Documentation Embedder
Captures the full breadth of Itential technical content including:
- Database Migration Scripts with examples
- Adapter configurations (LDAP, Email, Local AAA, etc.)
- Automation Studio features (Search Object Attributes, Form Data)
- ServiceNow Application Components
- Network configurations and troubleshooting
- Code examples and JSON configurations
- Step-by-step procedures
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

class ComprehensiveTechnicalEmbedder:
    """Comprehensive embedder that captures ALL technical content from Itential documentation."""
    
    def __init__(self, data_file: str = "complete_technical_docs.jsonl"):
        self.data_file = Path(data_file)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.db_path = "./technical_optimized_chroma_db"
        
        # Comprehensive content tracking
        self.content_categories = {
            'dependencies': [],
            'migration_scripts': [],
            'adapter_configs': [], 
            'automation_studio': [],
            'servicenow_components': [],
            'network_configs': [],
            'code_examples': [],
            'procedures': [],
            'troubleshooting': [],
            'api_references': []
        }
        
        # Enhanced extraction patterns
        self.content_patterns = {
            'migration_script': [
                'migration script', 'migratePropertiesToDatabase', 'database migration',
                'node_modules/@itential', 'cd /opt/pronghorn', 'properties.json'
            ],
            'adapter_config': [
                'adapter configuration', 'service config', 'ldap adapter', 'email adapter',
                'local aaa', 'adapter properties', 'brokers', 'authentication'
            ],
            'automation_studio': [
                'automation studio', 'search object attributes', 'form data', 'workflow',
                'formData', 'query task', 'automation canvas', 'job variable'
            ],
            'servicenow': [
                'servicenow', 'application components', 'service now', 'snow',
                'incident', 'service catalog', 'cmdb'
            ],
            'network_adapter': [
                'network adapter', 'automation gateway', 'connectivity check',
                'troubleshooting adapter', 'endpoint configuration'
            ],
            'code_example': [
                '{\n', '#!/', 'function', 'const ', 'var ', 'npm', 'git clone',
                'docker', 'kubectl', 'ansible'
            ],
            'procedure': [
                'step 1', 'step 2', 'navigate to', 'click', 'select', 'configure',
                'install', 'restart', 'verify'
            ]
        }

    def create_comprehensive_system(self) -> Any:
        """Create comprehensive system that captures all technical content."""
        logger.info("Creating Comprehensive Technical Documentation System...")
        
        # Load and analyze documents
        documents = self._load_documents()
        if not documents:
            raise ValueError("No documents found. Run the scraper first.")
        
        # Categorize all content
        self._categorize_all_content(documents)
        
        # Create comprehensive document chunks
        all_documents = self._create_comprehensive_chunks(documents)
        
        # Create vector store
        if Path(self.db_path).exists():
            shutil.rmtree(self.db_path)
            logger.info(f"Removed existing database: {self.db_path}")
        
        vector_store = Chroma.from_documents(
            documents=all_documents,
            embedding=self.embeddings,
            persist_directory=self.db_path
        )
        
        # Test comprehensive capabilities
        self._test_comprehensive_system(vector_store)
        
        logger.info("Comprehensive Technical System Ready!")
        return vector_store

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
        
        logger.info(f"Loaded {len(documents)} documents for comprehensive processing")
        return documents

    def _categorize_all_content(self, documents: List[Dict]) -> None:
        """Categorize all content by type for comprehensive coverage."""
        logger.info("Categorizing all technical content...")
        
        for doc in documents:
            content = (doc.get('searchable_text', '') + ' ' + 
                      doc.get('raw_text', '') + ' ' + 
                      doc.get('title', '')).lower()
            
            # Check each content pattern
            for category, patterns in self.content_patterns.items():
                if any(pattern in content for pattern in patterns):
                    self.content_categories[self._map_to_category(category)].append(doc)
                    break
            else:
                # Default categorization by content type
                content_type = doc.get('content_type', 'general')
                if content_type == 'dependencies':
                    self.content_categories['dependencies'].append(doc)
                elif 'code' in content or 'script' in content:
                    self.content_categories['code_examples'].append(doc)
                else:
                    self.content_categories['procedures'].append(doc)
        
        # Log categorization results
        for category, docs in self.content_categories.items():
            logger.info(f"  {category}: {len(docs)} documents")

    def _map_to_category(self, pattern_type: str) -> str:
        """Map pattern types to categories."""
        mapping = {
            'migration_script': 'migration_scripts',
            'adapter_config': 'adapter_configs',
            'automation_studio': 'automation_studio',
            'servicenow': 'servicenow_components',
            'network_adapter': 'network_configs',
            'code_example': 'code_examples',
            'procedure': 'procedures'
        }
        return mapping.get(pattern_type, 'procedures')

    def _create_comprehensive_chunks(self, documents: List[Dict]) -> List[Document]:
        """Create comprehensive chunks covering all content types."""
        all_documents = []
        
        # 1. Dependency tables and version info
        all_documents.extend(self._create_dependency_chunks())
        
        # 2. Migration scripts with examples
        all_documents.extend(self._create_migration_script_chunks())
        
        # 3. Adapter configurations
        all_documents.extend(self._create_adapter_config_chunks())
        
        # 4. Automation Studio content
        all_documents.extend(self._create_automation_studio_chunks())
        
        # 5. ServiceNow components
        all_documents.extend(self._create_servicenow_chunks())
        
        # 6. Code examples and configurations
        all_documents.extend(self._create_code_example_chunks())
        
        # 7. Procedures and troubleshooting
        all_documents.extend(self._create_procedure_chunks())
        
        # 8. General technical content
        all_documents.extend(self._create_general_technical_chunks(documents))
        
        logger.info(f"Created {len(all_documents)} comprehensive document chunks")
        return all_documents

    def _create_dependency_chunks(self) -> List[Document]:
        """Create dependency-specific chunks."""
        chunks = []
        dependency_docs = self.content_categories['dependencies']
        
        for doc in dependency_docs:
            # Extract dependency tables
            tables = doc.get('tables', [])
            for table in tables:
                if self._is_dependency_table(table):
                    chunks.extend(self._process_dependency_table(table, doc))
        
        logger.info(f"Created {len(chunks)} dependency chunks")
        return chunks

    def _create_migration_script_chunks(self) -> List[Document]:
        """Create migration script chunks with examples."""
        chunks = []
        migration_docs = self.content_categories['migration_scripts']
        
        for doc in migration_docs:
            content = doc.get('searchable_text', '') or doc.get('raw_text', '')
            title = doc.get('title', '')
            
            # Extract migration script content
            if 'migration' in content.lower() and 'script' in content.lower():
                
                # Create comprehensive migration chunk
                migration_content = f"""
Database Migration Script Information from {title}:

{content}

Migration Script Examples and Commands:
- Script location: /opt/pronghorn/current/node_modules/@itential/pronghorn-core/migration_scripts
- Main script: migratePropertiesToDatabase.js
- Command format: node migratePropertiesToDatabase.js --userInputs [parameters]
- Backup creation: properties_b4b03d30-ad00-4f61-bd9e-7953968ef8c4.json format

This script migrates properties.json configuration into MongoDB database.
After migration, properties.json only contains MongoDB connection properties.
All other configuration parameters are stored in the default 'pronghorn' database.
"""
                
                chunks.append(Document(
                    page_content=migration_content,
                    metadata={
                        "source": doc.get('url', 'unknown'),
                        "title": title,
                        "content_type": "migration_script",
                        "category": "database_migration",
                        "priority": "critical"
                    }
                ))
                
                # Create command-specific chunks
                code_blocks = doc.get('code_blocks', [])
                for code_block in code_blocks:
                    if 'migration' in code_block.get('content', '').lower():
                        chunks.append(Document(
                            page_content=f"Migration Command Example:\n{code_block.get('content', '')}",
                            metadata={
                                "source": doc.get('url', 'unknown'),
                                "content_type": "migration_command",
                                "category": "database_migration"
                            }
                        ))
        
        logger.info(f"Created {len(chunks)} migration script chunks")
        return chunks

    def _create_adapter_config_chunks(self) -> List[Document]:
        """Create adapter configuration chunks."""
        chunks = []
        adapter_docs = self.content_categories['adapter_configs']
        
        for doc in adapter_docs:
            title = doc.get('title', '')
            content = doc.get('searchable_text', '') or doc.get('raw_text', '')
            
            # Extract adapter configurations
            if any(adapter in title.lower() for adapter in ['ldap', 'email', 'local aaa', 'adapter']):
                
                # Create adapter overview chunk
                adapter_overview = f"""
{title} Configuration Guide:

{content[:1000]}...

Key Configuration Areas:
- Service Configuration (properties)
- Broker Configuration
- Authentication Settings
- Connection Parameters
"""
                
                chunks.append(Document(
                    page_content=adapter_overview,
                    metadata={
                        "source": doc.get('url', 'unknown'),
                        "title": title,
                        "content_type": "adapter_config",
                        "category": "adapter_configuration"
                    }
                ))
                
                # Extract JSON configurations
                code_blocks = doc.get('code_blocks', [])
                for code_block in code_blocks:
                    code_content = code_block.get('content', '')
                    if code_content.strip().startswith('{') and len(code_content) > 50:
                        chunks.append(Document(
                            page_content=f"{title} Configuration Example:\n{code_content}",
                            metadata={
                                "source": doc.get('url', 'unknown'),
                                "content_type": "adapter_json_config",
                                "category": "adapter_configuration"
                            }
                        ))
        
        logger.info(f"Created {len(chunks)} adapter configuration chunks")
        return chunks

    def _create_automation_studio_chunks(self) -> List[Document]:
        """Create Automation Studio chunks."""
        chunks = []
        automation_docs = self.content_categories['automation_studio']
        
        for doc in automation_docs:
            content = doc.get('searchable_text', '') or doc.get('raw_text', '')
            title = doc.get('title', '')
            
            # Search Object Attributes
            if 'search object attributes' in content.lower() or 'formdata' in content.lower():
                search_attributes_content = f"""
Automation Studio Search Object Attributes:

{content}

Key Concepts:
- formData object contains form input data
- Use query task to access formData fields
- Field labels are converted to camelCase (FirstName becomes firstName)
- formData must be added manually as job variable
- Access with Reference Variable: formData (case-sensitive)

Example Usage:
- Form field "FirstName" becomes formData.firstName
- Form field "LastName" becomes formData.lastName
- Query tasks can extract specific form values for workflow processing
"""
                
                chunks.append(Document(
                    page_content=search_attributes_content,
                    metadata={
                        "source": doc.get('url', 'unknown'),
                        "title": title,
                        "content_type": "automation_studio",
                        "category": "search_object_attributes",
                        "priority": "high"
                    }
                ))
        
        logger.info(f"Created {len(chunks)} Automation Studio chunks")
        return chunks

    def _create_servicenow_chunks(self) -> List[Document]:
        """Create ServiceNow application component chunks."""
        chunks = []
        servicenow_docs = self.content_categories['servicenow_components']
        
        for doc in servicenow_docs:
            content = doc.get('searchable_text', '') or doc.get('raw_text', '')
            title = doc.get('title', '')
            
            if 'servicenow' in content.lower():
                servicenow_content = f"""
ServiceNow Integration with Itential:

{title}

Content: {content}

ServiceNow Application Components typically include:
- Incident Management
- Service Catalog
- Configuration Management Database (CMDB)
- Change Management
- Problem Management
- Service Portal
- Workflow Engine
- Business Rules
- Script Includes
- UI Actions
"""
                
                chunks.append(Document(
                    page_content=servicenow_content,
                    metadata={
                        "source": doc.get('url', 'unknown'),
                        "title": title,
                        "content_type": "servicenow_integration",
                        "category": "servicenow_components"
                    }
                ))
        
        logger.info(f"Created {len(chunks)} ServiceNow chunks")
        return chunks

    def _create_code_example_chunks(self) -> List[Document]:
        """Create code example chunks."""
        chunks = []
        
        # Process all documents for code blocks
        for category_docs in self.content_categories.values():
            for doc in category_docs:
                code_blocks = doc.get('code_blocks', [])
                for code_block in code_blocks:
                    content = code_block.get('content', '')
                    context = code_block.get('context', '')
                    code_type = code_block.get('code_type', 'Code')
                    
                    if len(content.strip()) > 20:  # Meaningful code only
                        code_chunk = f"""
{code_type} Example:
Context: {context}

{content}

Code Type: {code_type}
Language: {code_block.get('language', 'Unknown')}
Technical Relevance: {code_block.get('technical_relevance', 'Medium')}
"""
                        
                        chunks.append(Document(
                            page_content=code_chunk,
                            metadata={
                                "source": doc.get('url', 'unknown'),
                                "title": doc.get('title', 'Unknown'),
                                "content_type": "code_example",
                                "code_type": code_type,
                                "language": code_block.get('language', 'unknown')
                            }
                        ))
        
        logger.info(f"Created {len(chunks)} code example chunks")
        return chunks

    def _create_procedure_chunks(self) -> List[Document]:
        """Create procedure and troubleshooting chunks."""
        chunks = []
        procedure_docs = (self.content_categories['procedures'] + 
                         self.content_categories['troubleshooting'])
        
        for doc in procedure_docs:
            content = doc.get('searchable_text', '') or doc.get('raw_text', '')
            title = doc.get('title', '')
            
            # Look for step-by-step procedures
            if any(indicator in content.lower() for indicator in ['step 1', 'step 2', 'navigate to', 'procedure']):
                
                # Enhanced chunking for procedures
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=400,
                    separators=['\n\n', '\n', '. ', ' ']
                )
                
                procedure_chunks = splitter.split_text(content)
                
                for i, chunk in enumerate(procedure_chunks):
                    enhanced_chunk = f"""
{title} - Procedure Part {i+1}:

{chunk}

Document Type: Technical Procedure
Source: {doc.get('url', 'unknown')}
"""
                    
                    chunks.append(Document(
                        page_content=enhanced_chunk,
                        metadata={
                            "source": doc.get('url', 'unknown'),
                            "title": title,
                            "content_type": "procedure",
                            "category": "technical_procedure",
                            "part": i+1
                        }
                    ))
        
        logger.info(f"Created {len(chunks)} procedure chunks")
        return chunks

    def _create_general_technical_chunks(self, documents: List[Dict]) -> List[Document]:
        """Create general technical chunks for comprehensive coverage."""
        chunks = []
        
        for doc in documents:
            if doc.get('technical_relevance_score', 0) > 3:
                content = doc.get('searchable_text', '') or doc.get('raw_text', '')
                if len(content) > 300:
                    
                    # Enhanced chunking
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1800,
                        chunk_overlap=300,
                        separators=['\n\n', '\n', '. ', ' ']
                    )
                    
                    text_chunks = splitter.split_text(content)
                    
                    for i, chunk in enumerate(text_chunks):
                        enhanced_chunk = f"""
{doc.get('title', 'Technical Content')}:

{chunk}

Technical Score: {doc.get('technical_relevance_score', 0)}
Content Type: {doc.get('content_type', 'general')}
"""
                        
                        chunks.append(Document(
                            page_content=enhanced_chunk,
                            metadata={
                                "source": doc.get('url', 'unknown'),
                                "title": doc.get('title', 'Untitled'),
                                "content_type": "general_technical",
                                "technical_score": doc.get('technical_relevance_score', 0),
                                "chunk_id": i
                            }
                        ))
        
        logger.info(f"Created {len(chunks)} general technical chunks")
        return chunks

    def _is_dependency_table(self, table: Dict) -> bool:
        """Check if table contains dependency information."""
        headers = table.get('headers', [])
        content = ' '.join(str(h) for h in headers).lower()
        
        dependency_indicators = ['mongodb', 'redis', 'rabbitmq', 'python', 'node', 'version', 'platform', 'iap']
        return sum(1 for indicator in dependency_indicators if indicator in content) >= 2

    def _process_dependency_table(self, table: Dict, doc: Dict) -> List[Document]:
        """Process dependency table into chunks."""
        chunks = []
        
        # Create table overview chunk
        table_content = f"""
Dependency Table from {doc.get('title', 'Unknown')}:

{table.get('markdown', '')}

Context: {table.get('context_before', '')}
Additional Info: {table.get('context_after', '')}
"""
        
        chunks.append(Document(
            page_content=table_content,
            metadata={
                "source": doc.get('url', 'unknown'),
                "content_type": "dependency_table",
                "category": "dependencies"
            }
        ))
        
        # Extract specific dependencies
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        if headers and rows:
            # Find product column
            product_col = None
            for i, header in enumerate(headers):
                if any(term in str(header).lower() for term in ['platform', 'version', 'iap']):
                    product_col = i
                    break
            
            if product_col is not None:
                for row in rows:
                    if len(row) > product_col:
                        product = str(row[product_col])
                        if product and len(product) > 1:
                            # Create product-specific chunk
                            product_deps = ' | '.join(str(cell) for cell in row)
                            product_chunk = f"Product: {product}\nDependencies: {product_deps}"
                            
                            chunks.append(Document(
                                page_content=product_chunk,
                                metadata={
                                    "source": doc.get('url', 'unknown'),
                                    "content_type": "product_dependency",
                                    "product": product
                                }
                            ))
        
        return chunks

    def _test_comprehensive_system(self, vector_store) -> None:
        """Test comprehensive system with diverse queries."""
        logger.info("Testing Comprehensive System...")
        
        test_queries = [
            # Dependency queries
            "What version of Redis is required for IAP 2023.2?",
            "MongoDB version for IAP 2023.2",
            
            # Migration script queries
            "How to run database migration script?",
            "migratePropertiesToDatabase.js command example",
            
            # Adapter configuration queries
            "LDAP adapter configuration example",
            "Email adapter JSON configuration",
            "Local AAA adapter setup",
            
            # Automation Studio queries
            "What are Search Object Attributes in Automation Studio?",
            "How to access formData in workflow?",
            "Form data query task example",
            
            # ServiceNow queries
            "ServiceNow Application Components",
            "ServiceNow integration with Itential",
            
            # Code and procedure queries
            "Adapter installation steps",
            "Troubleshooting adapter connectivity"
        ]
        
        success_count = 0
        category_coverage = {
            'dependencies': 0,
            'migration': 0,
            'adapters': 0,
            'automation_studio': 0,
            'servicenow': 0,
            'procedures': 0
        }
        
        for query in test_queries:
            try:
                results = vector_store.similarity_search(query, k=5)
                if results:
                    success_count += 1
                    
                    # Check category coverage
                    if 'redis' in query.lower() or 'mongodb' in query.lower():
                        category_coverage['dependencies'] += 1
                    elif 'migration' in query.lower():
                        category_coverage['migration'] += 1
                    elif 'adapter' in query.lower():
                        category_coverage['adapters'] += 1
                    elif 'automation studio' in query.lower() or 'formdata' in query.lower():
                        category_coverage['automation_studio'] += 1
                    elif 'servicenow' in query.lower():
                        category_coverage['servicenow'] += 1
                    else:
                        category_coverage['procedures'] += 1
                    
                    logger.info(f"PASS: '{query}' - Found relevant content")
                else:
                    logger.warning(f"FAIL: '{query}' - No results found")
                    
            except Exception as e:
                logger.error(f"ERROR: '{query}' - {e}")
        
        success_rate = success_count / len(test_queries) * 100
        logger.info(f"Test Results: {success_count}/{len(test_queries)} successful ({success_rate:.1f}%)")
        logger.info(f"Category Coverage: {category_coverage}")
        
        if success_rate >= 80:
            logger.info("Comprehensive system test PASSED!")
        else:
            logger.warning("Comprehensive system needs improvement")

def main():
    """Main function to create the comprehensive system."""
    logger.info("COMPREHENSIVE TECHNICAL DOCUMENTATION EMBEDDER")
    logger.info("=" * 80)
    
    embedder = ComprehensiveTechnicalEmbedder("complete_technical_docs.jsonl")
    vector_store = embedder.create_comprehensive_system()
    
    logger.info("\nCOMPREHENSIVE SYSTEM COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Database created: {embedder.db_path}")
    
    # Show content breakdown
    total_content = sum(len(docs) for docs in embedder.content_categories.values())
    logger.info(f"Total content processed: {total_content} documents")
    logger.info("\nContent breakdown:")
    for category, docs in embedder.content_categories.items():
        if docs:
            logger.info(f"  {category}: {len(docs)} documents")
    
    logger.info("\nYour AI can now answer comprehensive questions about:")
    logger.info("- Database Migration Scripts and commands")
    logger.info("- Adapter configurations (LDAP, Email, Local AAA, etc.)")
    logger.info("- Automation Studio features and Search Object Attributes")
    logger.info("- ServiceNow Application Components")
    logger.info("- Code examples and JSON configurations")
    logger.info("- Step-by-step procedures and troubleshooting")
    logger.info("- Network configurations and dependencies")

if __name__ == "__main__":
    main()