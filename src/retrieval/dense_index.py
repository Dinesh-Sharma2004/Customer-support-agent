import os
import joblib
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config.settings import settings

class DenseRetriever:
    def __init__(self, model_name: str = settings.DENSE_MODEL_NAME):
        self.model_name = model_name
        self.embedder = None
        self.index = None
        self.corpus_docs = []
        
    def _init_embedder(self):
        if self.embedder is None:
            self.embedder = SentenceTransformer(self.model_name, device="cpu")
            
    def build(self, documents: list[dict]):
        self._init_embedder()
        self.corpus_docs = documents
        
        texts = [doc['customer_text'] for doc in documents]
        embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        
        d = embeddings.shape[1]
        # Inner product index since embeddings are normalized -> cosine similarity
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)
        
    def save(self, filepath: str):
        if not self.index:
            raise ValueError("No index to save.")
        
        base_path = filepath.rsplit('.', 1)[0]
        faiss_path = base_path + ".faiss"
        meta_path = base_path + ".meta"
        
        faiss.write_index(self.index, faiss_path)
        joblib.dump(self.corpus_docs, meta_path)
        
    def load(self, filepath: str):
        base_path = filepath.rsplit('.', 1)[0]
        faiss_path = base_path + ".faiss"
        meta_path = base_path + ".meta"
        
        self.index = faiss.read_index(faiss_path)
        self.corpus_docs = joblib.load(meta_path)
        self._init_embedder()
        
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.index:
            raise ValueError("FAISS index not built or loaded.")
            
        self._init_embedder()
        
        query_emb = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True
        )
        
        scores, indices = self.index.search(query_emb, top_k)
        
        results = []
        for i in range(top_k):
            idx = indices[0][i]
            if idx != -1:
                results.append({
                    "doc_id": self.corpus_docs[idx]['doc_id'],
                    "score": float(scores[0][i]),
                    "customer_text": self.corpus_docs[idx]['customer_text'],
                    "brand_response": self.corpus_docs[idx]['brand_response']
                })
                
        return results
