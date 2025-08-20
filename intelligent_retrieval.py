#!/usr/bin/env python3
"""
Smart fix for existing system - no hardcoding, just intelligent document processing
This properly extracts ALL information from your JSONL files including tables
"""

import os
import json
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import logging
from collections import defaultdict

# LangChain imports
try:
    from langchain_ollama import OllamaEmbeddings, OllamaLLM as Ollama
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings
    from langchain_community.llms import Ollama

from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.retrievers import BM25Retriever, EnsembleRetriever

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SmartDocumentProcessor:
    """
    Intelligently processes documents to extract ALL information including tables
    """
    
    def __init__(self):
        self.all_documents = []
        self.bug_fixes = []
        self.version_info = []
        
    def process_jsonl_file(self, filepath: str) -> List[Document]:
        """
        Process JSONL file and extract ALL information intelligently
        """
        documents = []
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line)
                    
                    # Process ALL fields in the JSON
                    docs = self._extract_all_content(data, line_num)
                    documents.extend(docs)
                    
                except json.JSONDecodeError as e:
                    logger.debug(f"Skipping line {line_num}: {e}")
                except Exception as e:
                    logger.debug(f"Error on line {line_num}: {e}")
        
        logger.info(f"Extracted {len(documents)} documents from {filepath}")
        return documents
    
    def _extract_all_content(self, data: Dict, line_num: int) -> List[Document]:
        """
        Extract content from ALL fields in the data
        """
        documents = []
        source = data.get('url', f'line_{line_num}')
        
        # 1. Process tables - MOST IMPORTANT for bug fixes
        if 'tables' in data and data['tables']:
            for table_idx, table in enumerate(data['tables']):
                table_docs = self._process_table_intelligently(table, source, table_idx)
                documents.extend(table_docs)
        
        # 2. Process definition lists (often contain structured info)
        if 'definition_lists' in data and data['definition_lists']:
            for dl in data['definition_lists']:
                if isinstance(dl, dict) and 'items' in dl:
                    for item in dl['items']:
                        if isinstance(item, list) and len(item) >= 2:
                            term = str(item[0])
                            definition = str(item[1]) if len(item) > 1 else ""
                            
                            doc = Document(
                                page_content=f"{term}: {definition}",
                                metadata={
                                    "source": source,
                                    "type": "definition",
                                    "term": term
                                }
                            )
                            documents.append(doc)
        
        # 3. Process raw text with intelligent chunking
        text_content = data.get('searchable_text', '') or data.get('raw_text', '')
        if text_content:
            # Split into meaningful chunks based on structure
            text_docs = self._smart_chunk_text(text_content, source)
            documents.extend(text_docs)
        
        # 4. Process any CSV data (often contains bug lists)
        if 'csv' in data and data['csv']:
            csv_docs = self._process_csv_content(data['csv'], source)
            documents.extend(csv_docs)
        
        return documents
    
    def _process_table_intelligently(self, table: Dict, source: str, table_idx: int) -> List[Document]:
        """
        Process table data - tables often contain bug fixes and structured information
        """
        documents = []
        
        # Get headers if available
        headers = table.get('headers', [])
        rows = table.get('rows', [])
        
        # Also check for markdown and CSV representations
        markdown = table.get('markdown', '')
        csv = table.get('csv', '')
        
        # Process each row as a separate document
        for row_idx, row in enumerate(rows):
            if not row:
                continue
                
            # Build content from row
            content_parts = []
            metadata = {"source": source, "type": "table_row", "table_idx": table_idx}
            
            # If we have headers, create key-value pairs
            if headers and len(headers) == len(row):
                for header, value in zip(headers, row):
                    if value and str(value).strip():
                        content_parts.append(f"{header}: {value}")
                        # Add important fields to metadata
                        if any(key in str(header).lower() for key in ['id', 'key', 'ticket', 'eng', 'ph']):
                            metadata['identifier'] = str(value)
            else:
                # No headers, just join the row
                content_parts = [str(v) for v in row if v]
            
            if content_parts:
                content = " | ".join(content_parts)
                
                # Check if this looks like a bug fix or issue
                if self._is_bug_fix(content):
                    metadata['category'] = 'bug_fix'
                
                doc = Document(
                    page_content=content,
                    metadata=metadata
                )
                documents.append(doc)
        
        # Also process markdown/CSV as a whole for context
        if markdown:
            doc = Document(
                page_content=markdown,
                metadata={
                    "source": source,
                    "type": "table_markdown",
                    "table_idx": table_idx
                }
            )
            documents.append(doc)
        
        return documents
    
    def _process_csv_content(self, csv_text: str, source: str) -> List[Document]:
        """
        Process CSV content - often contains structured bug lists
        """
        documents = []
        
        lines = csv_text.strip().split('\n')
        if len(lines) < 2:
            return documents
        
        # Parse header
        header = lines[0].split(',')
        
        # Process each row
        for line in lines[1:]:
            # Simple CSV parsing (handles basic cases)
            values = line.split(',')
            
            if len(values) == len(header):
                content_parts = []
                metadata = {"source": source, "type": "csv_row"}
                
                for h, v in zip(header, values):
                    if v and v.strip():
                        content_parts.append(f"{h}: {v}")
                        # Capture identifiers
                        if any(key in h.lower() for key in ['id', 'key', 'ticket']):
                            metadata['identifier'] = v.strip()
                
                if content_parts:
                    content = " | ".join(content_parts)
                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)
        
        return documents
    
    def _smart_chunk_text(self, text: str, source: str) -> List[Document]:
        """
        Intelligently chunk text based on its structure
        """
        documents = []
        
        # First, try to identify and extract structured sections
        # Look for patterns like "ENG-XXXX: description" or "PH-XXXX: description"
        bug_pattern = r'([A-Z]{2,4}-\d{3,5})[:\s,]([^.!?\n]+[.!?]?)'
        matches = re.finditer(bug_pattern, text, re.MULTILINE)
        
        for match in matches:
            ticket_id = match.group(1)
            description = match.group(2).strip()
            
            doc = Document(
                page_content=f"{ticket_id}: {description}",
                metadata={
                    "source": source,
                    "type": "extracted_issue",
                    "identifier": ticket_id,
                    "category": "bug_fix" if self._is_bug_fix(description) else "feature"
                }
            )
            documents.append(doc)
        
        # Also chunk the full text for context
        # Use semantic boundaries
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ". ", ", ", " "],
            length_function=len
        )
        
        chunks = splitter.split_text(text)
        
        for i, chunk in enumerate(chunks):
            if len(chunk.strip()) > 30:  # Skip tiny chunks
                # Determine what this chunk is about
                metadata = {
                    "source": source,
                    "type": "text_chunk",
                    "chunk_idx": i
                }
                
                # Check for version information
                if any(word in chunk.lower() for word in ['node', 'python', 'mongodb', 'version', 'requirement']):
                    metadata['category'] = 'version_info'
                elif self._is_bug_fix(chunk):
                    metadata['category'] = 'bug_fix'
                
                doc = Document(
                    page_content=chunk,
                    metadata=metadata
                )
                documents.append(doc)
        
        return documents
    
    def _is_bug_fix(self, text: str) -> bool:
        """
        Determine if text describes a bug fix
        """
        bug_keywords = [
            'fix', 'fixed', 'resolve', 'resolved', 'patch', 'bug', 'issue',
            'error', 'problem', 'correct', 'corrected', 'update', 'updated'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in bug_keywords)

class EnhancedRetriever:
    """
    Enhanced retriever that uses multiple strategies
    """
    
    def __init__(self, documents: List[Document]):
        self.documents = documents
        self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.vector_store = None
        self.bm25_retriever = None
        self.ensemble_retriever = None
        
    def build_retrievers(self):
        """
        Build multiple retrievers for better results
        """
        logger.info("Building enhanced retrievers...")
        
        # 1. Vector store for semantic search
        self.vector_store = Chroma.from_documents(
            documents=self.documents,
            embedding=self.embeddings,
            persist_directory="./enhanced_technical_db"
        )
        
        # 2. BM25 for keyword search
        self.bm25_retriever = BM25Retriever.from_documents(self.documents)
        self.bm25_retriever.k = 5
        
        # 3. Ensemble retriever combining both
        vector_retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
        
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, vector_retriever],
            weights=[0.5, 0.5]  # Equal weight to keyword and semantic search
        )
        
        logger.info("Enhanced retrievers ready!")
    
    def retrieve(self, query: str, k: int = 10) -> List[Document]:
        """
        Retrieve relevant documents using ensemble approach
        """
        # Expand query with synonyms
        expanded_query = self._expand_query(query)
        
        # Get results from ensemble
        results = self.ensemble_retriever.get_relevant_documents(expanded_query)
        
        # Also do a direct search for any ticket IDs mentioned
        ticket_pattern = r'[A-Z]{2,4}-\d{3,5}'
        ticket_matches = re.findall(ticket_pattern, query.upper())
        
        if ticket_matches:
            # Find documents with these tickets
            for doc in self.documents:
                if any(ticket in doc.page_content for ticket in ticket_matches):
                    if doc not in results:
                        results.insert(0, doc)  # Put exact matches first
        
        return results[:k]
    
    def _expand_query(self, query: str) -> str:
        """
        Expand query with synonyms and related terms
        """
        expansions = {
            'eval': 'eval evaluation evaluate evaluator',
            'bug': 'bug fix issue problem error patch resolved',
            'node': 'node nodejs node.js javascript',
            'version': 'version requirement dependency required needs'
        }
        
        expanded = query
        for key, values in expansions.items():
            if key in query.lower():
                expanded += f" {values}"
        
        return expanded

class FixedChatbot:
    """
    Fixed chatbot that actually works
    """
    
    def __init__(self):
        self.processor = SmartDocumentProcessor()
        self.retriever = None
        self.llm = Ollama(model="mistral:7b", temperature=0.1)
        self.documents = []
        
    def initialize(self):
        """
        Initialize the fixed system
        """
        logger.info("Initializing fixed chatbot system...")
        
        # Process all JSONL files
        jsonl_files = [
            "complete_technical_docs.jsonl",
            "comprehensive_itential_docs.jsonl"
        ]
        
        all_docs = []
        for jsonl_file in jsonl_files:
            if Path(jsonl_file).exists():
                docs = self.processor.process_jsonl_file(jsonl_file)
                all_docs.extend(docs)
                logger.info(f"Processed {len(docs)} documents from {jsonl_file}")
        
        self.documents = all_docs
        logger.info(f"Total documents: {len(self.documents)}")
        
        # Build retrievers
        self.retriever = EnhancedRetriever(self.documents)
        self.retriever.build_retrievers()
        
        logger.info("Fixed chatbot ready!")
    
    def answer(self, query: str) -> str:
        """
        Answer a query with proper retrieval
        """
        # Retrieve relevant documents
        relevant_docs = self.retriever.retrieve(query, k=10)
        
        if not relevant_docs:
            return "I couldn't find relevant information to answer your question."
        
        # Build context
        context_parts = []
        seen_content = set()  # Avoid duplicates
        
        for doc in relevant_docs[:7]:  # Use top 7 documents
            content = doc.page_content.strip()
            if content not in seen_content:
                seen_content.add(content)
                context_parts.append(content)
        
        context = "\n\n".join(context_parts)
        
        # Generate answer
        prompt = f"""Based on the following information, answer the question accurately.

Context:
{context}

Question: {query}

Instructions:
- Answer based on the provided context
- Include specific IDs (like ENG-XXXX) when mentioned
- Be comprehensive and include all relevant information
- If multiple items match, list them all

Answer:"""
        
        response = self.llm.invoke(prompt)
        return response

def update_existing_system():
    """
    Update your existing improved_chatbot.py to use the fixed system
    """
    logger.info("Updating existing system...")
    
    # Create the updated chatbot file
    updated_chatbot = '''#!/usr/bin/env python3
"""
Fixed and improved chatbot - no hardcoding, just smart retrieval
"""

import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fix_existing_system import FixedChatbot

class EnhancedTechnicalChatbot:
    """Enhanced chatbot using fixed retrieval"""
    
    def __init__(self, **kwargs):
        self.chatbot = FixedChatbot()
        self.chatbot.initialize()
    
    def chat(self, query: str) -> str:
        return self.chatbot.answer(query)
    
    def process_query(self, query: str) -> str:
        return self.chat(query)

# For testing
if __name__ == "__main__":
    chatbot = EnhancedTechnicalChatbot()
    
    # Test queries
    test_queries = [
        "Were there any maintenance patches related to eval tasks?",
        "What Node.js version is required for IAP 2023.1?",
        "List all bug fixes for evaluation"
    ]
    
    for query in test_queries:
        print(f"\\nQ: {query}")
        print(f"A: {chatbot.chat(query)[:500]}...")
'''
    
    # Save the updated chatbot
    chatbot_path = Path("app/core/improved_chatbot.py")
    chatbot_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Backup original
    if chatbot_path.exists():
        backup_path = chatbot_path.with_suffix('.py.backup')
        shutil.copy(chatbot_path, backup_path)
        logger.info(f"Backed up original to {backup_path}")
    
    with open(chatbot_path, 'w') as f:
        f.write(updated_chatbot)
    
    logger.info(f"Updated {chatbot_path}")
    
    print("\n" + "="*60)
    print("SYSTEM UPDATED!")
    print("="*60)
    print("\nYou can now run:")
    print("1. Streamlit UI:   streamlit run app/enhanced_ui.py")
    print("2. Test directly:  python app/core/improved_chatbot.py")
    print("="*60)

def main():
    """Test and update the system"""
    
    # Initialize and test the fixed system
    chatbot = FixedChatbot()
    chatbot.initialize()
    
    # Test queries
    print("\n" + "="*60)
    print("TESTING FIXED SYSTEM")
    print("="*60)
    
    test_queries = [
        "Were there any maintenance patches related to eval tasks?",
        "What Node.js version is required for IAP 2023.1?",
        "What bugs were fixed for evaluation tasks?",
        "Find ENG-4407"
    ]
    
    for query in test_queries:
        print(f"\nQuestion: {query}")
        answer = chatbot.answer(query)
        print(f"Answer: {answer[:400]}...")
        
        # Check if key information is found
        if "eval" in query.lower():
            if "ENG-4407" in answer or "ENG-4715" in answer:
                print("[SUCCESS - Found eval bug fixes!]")
            else:
                print("[CHECKING - May need to verify extraction]")
    
    # Update the existing system
    print("\n" + "="*60)
    update_existing_system()

if __name__ == "__main__":
    main()