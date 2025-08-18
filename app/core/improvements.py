from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import numpy as np

class HybridRetriever:
    def __init__(self, vector_store, docs):
        self.vector_store = vector_store
        self.bm25 = BM25Okapi([doc.page_content.split() for doc in docs])
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    async def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        # Vector search
        vector_results = await self.vector_store.asimilarity_search(query, k=top_k*3)
        
        # BM25 search
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_results = [doc for doc, score in zip(docs, bm25_scores) 
                       if score > 0][:top_k*3]
        
        # Rerank with cross-encoder
        combined = list(set(vector_results + bm25_results))
        cross_input = [(query, doc.page_content) for doc in combined]
        cross_scores = self.cross_encoder.predict(cross_input)
        
        # Combine scores
        ranked_results = sorted(
            zip(combined, cross_scores),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [{
            'content': doc.page_content,
            'source': doc.metadata.get('source', ''),
            'score': float(score)
        } for doc, score in ranked_results]