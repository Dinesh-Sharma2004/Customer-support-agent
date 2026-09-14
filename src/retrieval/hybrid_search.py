def compute_rrf(rankings: list[list[dict]], k: int = 60) -> list[dict]:
    """
    Computes Reciprocal Rank Fusion.
    rankings: A list of lists, where each sublist is the top-N results from a retriever.
    k: The RRF constant (usually 60).
    """
    rrf_scores = {}
    docs_map = {}
    
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            doc_id = doc['doc_id']
            docs_map[doc_id] = doc
            
            if doc_id not in rrf_scores:
                rrf_scores[doc_id] = 0.0
            
            # Rank is 0-indexed, so we add 1
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
            
    # Sort by RRF score descending
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    results = []
    for doc_id, score in sorted_docs:
        doc = docs_map[doc_id].copy()
        doc['rrf_score'] = score
        results.append(doc)
        
    return results

class HybridRetriever:
    def __init__(self, bm25_retriever, dense_retriever):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        # Get more documents from individual retrievers to ensure good RRF overlap
        fetch_k = max(top_k * 2, 20)
        
        bm25_results = self.bm25.search(query, top_k=fetch_k)
        dense_results = self.dense.search(query, top_k=fetch_k)
        
        fused = compute_rrf([bm25_results, dense_results])
        return fused[:top_k]
