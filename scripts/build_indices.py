import os
import pandas as pd
import json

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.settings import settings
from src.retrieval.bm25_index import BM25Retriever
from src.retrieval.dense_index import DenseRetriever

def build_retrieval_indices():
    splits_dir = os.path.join(settings.DATA_DIR, "splits")
    retrieval_path = os.path.join(splits_dir, "retrieval_corpus.csv")
    
    if not os.path.exists(retrieval_path):
        print(f"Retrieval corpus not found at {retrieval_path}")
        sys.exit(1)
        
    print("Loading retrieval corpus...")
    corpus_df = pd.read_csv(retrieval_path)
    
    # Needs to be a list of dicts with doc_id, customer_text, brand_response
    # Let's add a doc_id if not present
    if 'doc_id' not in corpus_df.columns:
        corpus_df['doc_id'] = [f"doc_{i}" for i in range(len(corpus_df))]
        
    documents = corpus_df[['doc_id', 'customer_text', 'brand_response']].to_dict('records')
    
    print(f"Building indices for {len(documents)} documents...")
    
    # 1. Build BM25
    print("Building BM25 Index...")
    bm25 = BM25Retriever()
    bm25.build(documents)
    bm25_path = os.path.join(settings.ARTIFACTS_DIR, "bm25_index.joblib")
    bm25.save(bm25_path)
    print(f"BM25 saved to {bm25_path}")
    
    # 2. Build Dense
    print("Building FAISS Dense Index...")
    dense = DenseRetriever()
    dense.build(documents)
    dense_path = os.path.join(settings.ARTIFACTS_DIR, "dense_index.faiss")
    dense.save(dense_path)
    print(f"FAISS Dense saved to {dense_path}")

if __name__ == "__main__":
    build_retrieval_indices()
