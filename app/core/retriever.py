import numpy as np
from collections import defaultdict
from typing import List, Dict
from langchain_chroma import Chroma

class RobustRetriever:
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        self.token_index = defaultdict(list)
        self.documents = []
        
    def build_index(self, docs: List[Dict]):
        """Build a simple token index for BM25-like retrieval"""
        self.documents = docs
        for idx, doc in enumerate(docs):
            tokens = doc['content'].lower().split()
            for token in set(tokens):  # Only store unique tokens per doc
                self.token_index[token].append(idx)
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Dict]:
        # Vector search
        vector_results = self.vector_store.similarity_search(query, k=top_k*2)
        
        # Token-based search (simplified BM25)
        query_tokens = query.lower().split()
        doc_scores = defaultdict(float)
        
        for token in query_tokens:
            for doc_idx in self.token_index.get(token, []):
                doc = self.documents[doc_idx]
                doc_scores[doc_idx] += 1.0 / (1 + len(self.token_index[token]))
        
        # Combine results
        scored_docs = []
        for doc_idx, score in doc_scores.items():
            scored_docs.append((self.documents[doc_idx], score))
        
        # Add vector results
        for doc in vector_results:
            scored_docs.append(({
                'content': doc.page_content,
                'source': doc.metadata.get('source', '')
            }, 0.5))  # Default score
        
        # Deduplicate and sort
        unique_docs = {}
        for doc, score in scored_docs:
            doc_id = doc['source'] + doc['content'][:100]
            unique_docs[doc_id] = max(unique_docs.get(doc_id, 0), score)
        
        return sorted(unique_docs.items(),
                    key=lambda x: x[1],
                    reverse=True)[:top_k]