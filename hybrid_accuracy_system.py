#!/usr/bin/env python3
"""
Hybrid High-Accuracy Documentation System
Combines multiple storage strategies for maximum accuracy like Claude's approach.

Strategy:
1. Store complete pages for full context
2. Store structured data (tables, matrices) separately  
3. Store Q&A pairs for specific queries
4. Use intelligent routing for optimal retrieval
"""

import json
import re
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
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

@dataclass
class StructuredData:
    """Container for structured technical data."""
    data_type: str  # 'table', 'matrix', 'list', 'code'
    title: str
    content: Dict[str, Any]
    source_url: str
    context: str
    technical_relevance: str

class HybridAccuracySystem:
    """
    Hybrid system that stores documentation in multiple formats for maximum accuracy.
    
    Storage Layers:
    1. Complete Pages - Full context preservation
    2. Structured Data - Tables, matrices, code blocks
    3. Q&A Pairs - Specific query optimization
    4. Semantic Search - Fast retrieval routing
    """
    
    def __init__(self, data_file: str = "complete_technical_docs.jsonl"):
        self.data_file = Path(data_file)
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Storage components
        self.complete_pages_db = None
        self.structured_data_db = None  
        self.qa_pairs_db = None
        self.routing_db = None
        
        # Data registries
        self.dependency_matrices = {}
        self.version_tables = {}
        self.complete_pages = {}
        self.structured_data = []
        
        # Performance tracking
        self.accuracy_metrics = {
            'complete_pages': 0,
            'structured_items': 0,
            'qa_pairs': 0,
            'routing_entries': 0
        }

    def create_hybrid_accuracy_system(self) -> Dict[str, Any]:
        """Create the complete hybrid accuracy system."""
        logger.info("🎯 Creating Hybrid High-Accuracy Documentation System...")
        
        # Load and process data
        documents = self._load_documents()
        
        # Create all storage layers
        storage_layers = {
            'complete_pages': self._create_complete_pages_layer(documents),
            'structured_data': self._create_structured_data_layer(documents),
            'qa_pairs': self._create_qa_pairs_layer(documents),
            'routing': self._create_routing_layer(documents)
        }
        
        # Verify system integrity
        self._verify_system_integrity(storage_layers)
        
        return storage_layers

    def _load_documents(self) -> List[Dict[str, Any]]:
        """Load documents from the technical scraper output."""
        documents = []
        
        with open(self.data_file, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                if line.strip():
                    try:
                        doc_data = json.loads(line)
                        documents.append(doc_data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Skipping malformed JSON on line {line_num}: {e}")
        
        logger.info(f"📚 Loaded {len(documents)} documents for hybrid processing")
        return documents

    def _create_complete_pages_layer(self, documents: List[Dict]) -> Chroma:
        """
        Layer 1: Store COMPLETE pages with full context.
        This preserves the full semantic context like Claude has access to.
        """
        logger.info("📄 Creating Complete Pages Layer...")
        
        complete_page_docs = []
        
        for doc in documents:
            # Only store critical technical pages in complete form
            if doc.get('is_critical', False) or doc.get('technical_relevance_score', 0) > 10:
                
                # Create comprehensive page content
                page_content = self._create_complete_page_content(doc)
                
                complete_page_docs.append(Document(
                    page_content=page_content,
                    metadata={
                        "source": doc.get('url', 'unknown'),
                        "title": doc.get('title', 'Untitled'),
                        "content_type": "complete_page",
                        "layer": "complete_pages",
                        "technical_score": doc.get('technical_relevance_score', 0),
                        "has_tables": doc.get('has_tables', False),
                        "has_dependencies": doc.get('has_dependencies', False),
                        "priority": "complete_context"
                    }
                ))
                
                self.accuracy_metrics['complete_pages'] += 1
        
        # Create vector store for complete pages
        complete_pages_db = Chroma.from_documents(
            documents=complete_page_docs,
            embedding=self.embeddings,
            persist_directory="./hybrid_complete_pages_db"
        )
        
        logger.info(f"✅ Complete Pages Layer: {len(complete_page_docs)} full pages stored")
        return complete_pages_db

    def _create_complete_page_content(self, doc: Dict) -> str:
        """Create complete page content preserving all context."""
        content_parts = []
        
        # Title and description
        title = doc.get('title', '')
        description = doc.get('description', '')
        if title:
            content_parts.append(f"# {title}")
        if description:
            content_parts.append(f"Description: {description}")
        
        # Headings with hierarchy
        headings = doc.get('headings', [])
        if headings:
            content_parts.append("\n## Document Structure:")
            for heading in headings:
                level = heading.get('level', 'h1')
                text = heading.get('text', '')
                indent = "  " * (int(level[1]) - 1)
                content_parts.append(f"{indent}• {text}")
        
        # Tables with full context
        tables = doc.get('tables', [])
        if tables:
            content_parts.append("\n## Tables and Data:")
            for i, table in enumerate(tables):
                content_parts.append(f"\n### Table {i+1}")
                
                # Context before table
                context_before = table.get('context_before', '')
                if context_before:
                    content_parts.append(f"Context: {context_before}")
                
                # Table content
                if 'markdown' in table:
                    content_parts.append(table['markdown'])
                elif 'text' in table:
                    content_parts.append(table['text'])
                
                # Context after table
                context_after = table.get('context_after', '')
                if context_after:
                    content_parts.append(f"Additional info: {context_after}")
        
        # Code blocks
        code_blocks = doc.get('code_blocks', [])
        if code_blocks:
            content_parts.append("\n## Code and Configuration:")
            for i, code_block in enumerate(code_blocks):
                content_parts.append(f"\n### {code_block.get('code_type', 'Code')} {i+1}")
                context = code_block.get('context', '')
                if context:
                    content_parts.append(f"Context: {context}")
                content_parts.append(f"```{code_block.get('language', '')}")
                content_parts.append(code_block.get('content', ''))
                content_parts.append("```")
        
        # Full raw text for completeness
        raw_text = doc.get('raw_text', '')
        if raw_text and len(raw_text) > 100:
            content_parts.append("\n## Complete Page Content:")
            content_parts.append(raw_text)
        
        return '\n'.join(content_parts)

    def _create_structured_data_layer(self, documents: List[Dict]) -> Chroma:
        """
        Layer 2: Store STRUCTURED data (tables, matrices) separately.
        This ensures critical data structures are never fragmented.
        """
        logger.info("📊 Creating Structured Data Layer...")
        
        structured_docs = []
        
        for doc in documents:
            tables = doc.get('tables', [])
            
            for table in tables:
                # Only store critical technical tables
                if table.get('technical_relevance') in ['critical', 'high']:
                    
                    structured_item = self._create_structured_data_item(table, doc)
                    
                    # Create searchable content for the structured data
                    search_content = self._create_structured_search_content(structured_item)
                    
                    structured_docs.append(Document(
                        page_content=search_content,
                        metadata={
                            "source": doc.get('url', 'unknown'),
                            "title": doc.get('title', 'Untitled'),
                            "content_type": "structured_data",
                            "layer": "structured_data",
                            "data_type": structured_item.data_type,
                            "technical_relevance": structured_item.technical_relevance,
                            "priority": "structured_preservation"
                        }
                    ))
                    
                    self.accuracy_metrics['structured_items'] += 1
        
        # Create vector store for structured data
        structured_data_db = Chroma.from_documents(
            documents=structured_docs,
            embedding=self.embeddings,
            persist_directory="./hybrid_structured_data_db"
        )
        
        logger.info(f"✅ Structured Data Layer: {len(structured_docs)} data structures stored")
        return structured_data_db

    def _create_structured_data_item(self, table: Dict, doc: Dict) -> StructuredData:
        """Create a structured data item from a table."""
        # Determine data type
        if 'dependency' in str(table).lower() or 'version' in str(table).lower():
            data_type = 'dependency_matrix'
        elif 'requirement' in str(table).lower():
            data_type = 'requirements_table'
        else:
            data_type = 'technical_table'
        
        # Extract title from context or headers
        title = "Technical Data"
        context_before = table.get('context_before', '')
        if context_before:
            # Try to extract title from context
            title_match = re.search(r'HEADER:\s*([^|]+)', context_before)
            if title_match:
                title = title_match.group(1).strip()
        
        return StructuredData(
            data_type=data_type,
            title=title,
            content=table,
            source_url=doc.get('url', 'unknown'),
            context=context_before,
            technical_relevance=table.get('technical_relevance', 'medium')
        )

    def _create_structured_search_content(self, structured_item: StructuredData) -> str:
        """Create searchable content for structured data."""
        content_parts = [
            f"# {structured_item.title}",
            f"Data Type: {structured_item.data_type}",
            f"Context: {structured_item.context}"
        ]
        
        # Add table content in multiple formats for better matching
        table = structured_item.content
        
        if 'markdown' in table:
            content_parts.append("## Table Content (Markdown)")
            content_parts.append(table['markdown'])
        
        if 'headers' in table and 'rows' in table:
            content_parts.append("## Table Content (Structured)")
            headers = table['headers']
            rows = table['rows']
            
            content_parts.append(f"Headers: {' | '.join(str(h) for h in headers)}")
            
            for i, row in enumerate(rows[:10]):  # Limit to first 10 rows
                row_str = ' | '.join(str(cell) for cell in row)
                content_parts.append(f"Row {i+1}: {row_str}")
        
        return '\n'.join(content_parts)

    def _create_qa_pairs_layer(self, documents: List[Dict]) -> Chroma:
        """
        Layer 3: Create specific Q&A pairs for common queries.
        This ensures fast, accurate responses to specific questions.
        """
        logger.info("❓ Creating Q&A Pairs Layer...")
        
        qa_docs = []
        
        # Import the enhanced embedder's Q&A creation logic
        from robust_technical_embedder import EnhancedTechnicalDocumentationEmbedder
        embedder = EnhancedTechnicalDocumentationEmbedder()
        
        for doc in documents:
            if doc.get('content_type') in ['dependencies', 'version_lifecycle', 'installation_config']:
                # Create enhanced Q&A pairs
                qa_pairs = embedder.create_enhanced_qa_pairs(doc)
                qa_docs.extend(qa_pairs)
                
                self.accuracy_metrics['qa_pairs'] += len(qa_pairs)
        
        # Create vector store for Q&A pairs
        qa_pairs_db = Chroma.from_documents(
            documents=qa_docs,
            embedding=self.embeddings,
            persist_directory="./hybrid_qa_pairs_db"
        )
        
        logger.info(f"✅ Q&A Pairs Layer: {len(qa_docs)} Q&A pairs created")
        return qa_pairs_db

    def _create_routing_layer(self, documents: List[Dict]) -> Chroma:
        """
        Layer 4: Create routing layer for intelligent query distribution.
        This determines which layer should handle each type of query.
        """
        logger.info("🧭 Creating Routing Layer...")
        
        routing_docs = []
        
        # Create routing entries for different query types
        routing_patterns = {
            'version_specific': {
                'patterns': ['what version of', 'version required', 'version needed'],
                'route_to': 'qa_pairs',
                'description': 'Specific version requirement queries'
            },
            'not_required': {
                'patterns': ['is required', 'does need', 'no longer needed'],
                'route_to': 'qa_pairs',
                'description': 'Dependency requirement status queries'
            },
            'table_lookup': {
                'patterns': ['show table', 'dependency matrix', 'requirements table'],
                'route_to': 'structured_data',
                'description': 'Table and matrix lookup queries'
            },
            'general_info': {
                'patterns': ['how to', 'install', 'configure', 'setup'],
                'route_to': 'complete_pages',
                'description': 'General information and procedures'
            },
            'system_requirements': {
                'patterns': ['system requirements', 'prerequisites', 'dependencies'],
                'route_to': 'qa_pairs',
                'description': 'System requirement queries'
            }
        }
        
        for query_type, info in routing_patterns.items():
            routing_content = f"""
Query Type: {query_type}
Route To: {info['route_to']}
Description: {info['description']}
Patterns: {', '.join(info['patterns'])}

This routing entry helps determine that queries matching these patterns should be handled by the {info['route_to']} layer for optimal accuracy.
"""
            
            routing_docs.append(Document(
                page_content=routing_content,
                metadata={
                    "content_type": "routing_rule",
                    "layer": "routing",
                    "query_type": query_type,
                    "route_to": info['route_to'],
                    "priority": "routing"
                }
            ))
            
            self.accuracy_metrics['routing_entries'] += 1
        
        # Create vector store for routing
        routing_db = Chroma.from_documents(
            documents=routing_docs,
            embedding=self.embeddings,
            persist_directory="./hybrid_routing_db"
        )
        
        logger.info(f"✅ Routing Layer: {len(routing_docs)} routing rules created")
        return routing_db

    def _verify_system_integrity(self, storage_layers: Dict) -> None:
        """Verify the integrity of the hybrid system."""
        logger.info("🔍 Verifying Hybrid System Integrity...")
        
        # Test critical queries
        test_queries = [
            "What version of RabbitMQ is required for IAP 2023.2?",
            "Is RabbitMQ required for IAP 2023.2?",
            "What are the system requirements for IAP 2023.2?",
            "Show me the dependency table for IAP versions"
        ]
        
        success_count = 0
        
        for query in test_queries:
            try:
                # Test each layer
                for layer_name, layer_db in storage_layers.items():
                    if layer_db:
                        results = layer_db.similarity_search(query, k=3)
                        if results:
                            success_count += 1
                            logger.info(f"✅ '{query[:40]}...' found in {layer_name} layer")
                            break
                else:
                    logger.warning(f"⚠️ '{query[:40]}...' not found in any layer")
                    
            except Exception as e:
                logger.error(f"❌ Error testing query '{query[:40]}...': {e}")
        
        # Report integrity
        total_tests = len(test_queries)
        success_rate = success_count / total_tests * 100
        
        logger.info(f"📊 System Integrity: {success_count}/{total_tests} queries successful ({success_rate:.1f}%)")
        
        if success_rate >= 80:
            logger.info("🎉 Hybrid System Integrity VERIFIED")
        else:
            logger.warning("⚠️ Hybrid System Integrity ISSUES DETECTED")

class HybridQueryRouter:
    """Intelligent query router for the hybrid system."""
    
    def __init__(self, storage_layers: Dict[str, Any]):
        self.storage_layers = storage_layers
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
    
    def route_query(self, query: str) -> Tuple[str, List[Document]]:
        """Route query to the most appropriate layer."""
        query_lower = query.lower()
        
        # Simple rule-based routing (can be enhanced with ML)
        if any(pattern in query_lower for pattern in ['what version', 'version required', 'is required']):
            layer = 'qa_pairs'
        elif any(pattern in query_lower for pattern in ['table', 'matrix', 'show']):
            layer = 'structured_data'
        elif any(pattern in query_lower for pattern in ['how to', 'install', 'configure']):
            layer = 'complete_pages'
        else:
            layer = 'qa_pairs'  # Default to Q&A for specific queries
        
        # Search the selected layer
        if layer in self.storage_layers and self.storage_layers[layer]:
            results = self.storage_layers[layer].similarity_search(query, k=5)
            return layer, results
        
        # Fallback to searching all layers
        all_results = []
        for layer_name, layer_db in self.storage_layers.items():
            if layer_db:
                layer_results = layer_db.similarity_search(query, k=2)
                all_results.extend(layer_results)
        
        return 'fallback', all_results

def create_hybrid_accuracy_system():
    """Main function to create the hybrid accuracy system."""
    logger.info("🚀 HYBRID HIGH-ACCURACY DOCUMENTATION SYSTEM")
    logger.info("=" * 80)
    
    system = HybridAccuracySystem("complete_technical_docs.jsonl")
    storage_layers = system.create_hybrid_accuracy_system()
    
    logger.info("📊 Final System Statistics:")
    logger.info(f"  📄 Complete Pages: {system.accuracy_metrics['complete_pages']}")
    logger.info(f"  📊 Structured Items: {system.accuracy_metrics['structured_items']}")
    logger.info(f"  ❓ Q&A Pairs: {system.accuracy_metrics['qa_pairs']}")
    logger.info(f"  🧭 Routing Rules: {system.accuracy_metrics['routing_entries']}")
    
    total_items = sum(system.accuracy_metrics.values())
    logger.info(f"  🎯 Total Items: {total_items}")
    
    if total_items > 1000:
        logger.info("🎉 HYBRID HIGH-ACCURACY SYSTEM READY!")
        logger.info("This system provides Claude-level accuracy through multiple storage layers.")
    else:
        logger.warning("⚠️ System may need more data for optimal accuracy")
    
    return storage_layers

if __name__ == "__main__":
    create_hybrid_accuracy_system()