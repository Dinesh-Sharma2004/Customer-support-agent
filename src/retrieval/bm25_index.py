import os
import joblib
from rank_bm25 import BM25Okapi
import re
from src.config.settings import settings

def clean_text_bm25(text: str) -> list[str]:
    """Clean text and return token list for BM25."""
    if not isinstance(text, str):
        return []
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Remove @mentions
    text = re.sub(r'@[\w_]+', '', text)
    # Remove #hashtags
    text = re.sub(r'#[\w_]+', '', text)
    # Remove special chars except apostrophe
    text = re.sub(r'[^\w\s\'′]', ' ', text)
    # Lowercase and tokenize
    return text.lower().split()

class BM25Retriever:
    def __init__(self):
        self.bm25 = None
        self.corpus_docs = []
        
    def build(self, documents: list[dict]):
        """
        documents: list of dicts with 'doc_id', 'customer_text', 'brand_response'
        """
        self.corpus_docs = documents
        tokenized_corpus = [clean_text_bm25(doc['customer_text']) for doc in documents]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
    def save(self, filepath: str):
        joblib.dump((self.bm25, self.corpus_docs), filepath)
        
    def load(self, filepath: str):
        self.bm25, self.corpus_docs = joblib.load(filepath)
        
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.bm25:
            raise ValueError("BM25 index not built or loaded.")
            
        tokenized_query = clean_text_bm25(query)
        # Get raw scores
        doc_scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k
        import numpy as np
        top_n_idx = np.argsort(doc_scores)[::-1][:top_k]
        
        results = []
        for idx in top_n_idx:
            results.append({
                "doc_id": self.corpus_docs[idx]['doc_id'],
                "score": float(doc_scores[idx]),
                "customer_text": self.corpus_docs[idx]['customer_text'],
                "brand_response": self.corpus_docs[idx]['brand_response']
            })
            
        return results
