# RAG And Retrieval

The notebook builds the retrieval corpus from `train_set` only, after Golden leakage exclusions and before validation/Golden evaluation.

RAG corpus fields:

- `tweet_id`
- `customer_text`
- `brand_response`
- `resolution_action`
- `intent`
- `doc_id`

Architecture:

```text
Customer Query
      |
 .----+----.
 |         |
BM25     Dense Search
 |         |
 '----+----'
      |
     RRF
      |
Top-k historical cases
      |
Evidence pack
      |
Grounding / response generation
```

Verified implementation:

- BM25 uses `rank_bm25.BM25Okapi` over cleaned token lists.
- Dense retrieval uses `sentence-transformers` with `all-MiniLM-L6-v2`.
- FAISS indexes normalized float32 embeddings.
- Reciprocal Rank Fusion combines BM25 and dense rankings with `1 / (k + rank)`.
- Validation code evaluates BM25, Dense, and RRF using Recall@k and MRR, but the expected `retrieval_eval.csv` artifact is not present in the current artifact directory. Exact retrieval scores are therefore not verified.

Hybrid retrieval is preferred because BM25 preserves lexical precision on short tweets while dense embeddings help with paraphrases. Similarity is not automatically evidence: a retrieved case may match words while resolving a different issue, or it may be too generic to support an auto-handled response.

## Leakage

Leakage means letting Golden evaluation examples, their exact duplicates, or direct related tweets enter training, retrieval, threshold selection, or prompt/policy tuning.

Implemented exclusions in the notebook:

- Golden `tweet_id_customer` exclusion.
- Exact normalized customer text exclusion.
- Immediate-parent exclusion where the linked parent ID is available.

Guaranteed by current code: exact tweet ID/text matches and immediate-parent checks as implemented.

Not guaranteed: full thread isolation, near-duplicate removal, semantic duplicate removal, or author/customer ID as conversation ID. The pipeline should not be described as zero leakage.

Next: [Response generation](07_response_generation_and_grounding.md).
