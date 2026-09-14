# Document 4: Decision Log

This log captures the non-obvious engineering and research decisions made during the AmazonHelp Customer Support RAG Agent project.

### 1. Realistic Sampling Strategy over Artificial Balancing
- **Chosen Approach:** Sampled 200 evaluation queries to mirror the natural volume distribution of Amazon support traffic, rather than forcing an equal number of examples per intent.
- **Why Chosen / Evidence:** Real-world traffic is heavily skewed towards delivery issues. Evaluating on an artificially balanced set drastically distorts precision/recall expectations in deployment. 
- **Advantages:** Provides a true estimate of operational auto-handle rates and safety risks.
- **Disadvantages:** Minor classes (e.g., Security) have wide confidence intervals due to low N.
- **Alternative Rejected:** SMOTE/oversampling or artificially balanced evaluation sets. Rejected because our primary objective was establishing a defensible baseline of real-world risk, not maximizing a theoretical benchmark score.

### 2. Taxonomy Design (6-Class)
- **Chosen Approach:** Consolidated the massive variety of Amazon intents into 5 primary transactional buckets + 1 "Other / Human Review" bucket.
- **Why Chosen:** Customer support requests often bleed across boundaries (e.g., "Where is my refund for my missing package?"). A highly granular 50-class taxonomy creates unmanageable overlap and lowers inter-annotator agreement. 
- **Alternative Rejected:** Flat 20+ class taxonomy. Rejected because categorical/text characteristics at that granularity require vast amounts of human-labelled data which we did not have, lowering reproducibility.

### 3. Absolute Train/Evaluation Separation
- **Chosen Approach:** Physically partitioned the 200 golden `tweet_id`s from the `twcs.csv` corpus entirely *before* generating the 168k train/dev sets via a strict filtering script.
- **Why Chosen / Evidence:** To prevent data leakage. The previous architecture accidentally split the golden set into training and validation, resulting in artificially inflated accuracy metrics (>95%).
- **Alternative Rejected:** Standard random `train_test_split()` across the whole corpus. Rejected because it risks exposing golden examples to the TF-IDF vectorizer and LLM prompts.

### 4. TF-IDF + Logistic Regression for Intent Classification
- **Chosen Approach:** Used a combined Word/Char n-gram TF-IDF vectorizer fed into a Logistic Regression classifier (`C=5`, balanced class weights).
- **Why Chosen:** Deployment simplicity, high interpretability, and robust handling of high-dimensional text data out-of-the-box.
- **Disadvantages:** Cannot capture deep semantic relationships or paraphrasing as well as embeddings.
- **Alternative Rejected:** Deep learning alternatives (e.g., CatBoost on text features, fine-tuned BERT, SetFit). Rejected because the current objective was to establish a highly reproducible, interpretable, defensible baseline before adding heavy compute complexity.

### 5. Probability Calibration (`CalibratedClassifierCV`)
- **Chosen Approach:** Wrapped the fitted Logistic Regression pipeline in a `FrozenEstimator` and applied Sigmoid calibration (`CalibratedClassifierCV`) using an isolated calibration dataset.
- **Why Chosen:** Logistic Regression raw predict probabilities are often poorly calibrated (overconfident). Because our safety routing relies on confidence thresholds, calibration was mandatory.
- **Alternative Rejected:** Isotonic regression. Rejected because isotonic regression tends to overfit on smaller calibration sets compared to Sigmoid.

### 6. Hybrid Retrieval (BM25 + Dense FAISS via RRF)
- **Chosen Approach:** Combined `rank_bm25` (lexical) with `all-MiniLM-L6-v2` dense embeddings (FAISS), merging results via Reciprocal Rank Fusion (RRF).
- **Why Chosen:** Support queries contain highly specific identifiers (order numbers, exact brand names) where dense embeddings fail, but also varied phrasing where BM25 fails. Hybrid solves both.
- **Alternative Rejected:** Advanced cross-encoder re-ranking (e.g., Cohere Rerank or MS-MARCO fine-tunes). Rejected due to compute constraints and latency overhead; RRF provides 80% of the benefit at 0 runtime cost.

### 7. Fail-Closed Routing and Fallback Behavior
- **Chosen Approach:** Hard-coded policy rules that force an `ESCALATE` decision if the predicted intent is "Security" or "Financial", or if confidence is `< 0.70`, regardless of the LLM's opinion.
- **Why Chosen:** Safety and liability. An LLM cannot be trusted to independently arbitrate severe account locks or financial disputes.
- **Alternative Rejected:** Prompting the LLM to decide whether to escalate. Rejected because LLMs are prone to sycophancy and jailbreaks; deterministic rule-based gating guarantees safety.

### 8. Strict Evidence Verification
- **Chosen Approach:** A deterministic pre-generation check that verifies retrieved documents meet a minimum score threshold and do not contain obvious contradictory signals before passing them to the generator.
- **Why Chosen:** Reduces hallucination. If the retriever fails, the generator is deprived of context and forced to decline.
- **Alternative Rejected:** Relying on the LLM to output "I don't know" when context is poor. Rejected because LLMs often attempt to answer anyway using pre-training knowledge, which is unacceptable for localized brand support.

### 9. Grounded Generation Architecture
- **Chosen Approach:** A provider-agnostic adapter forcing the LLM to output a strict JSON schema containing the answer, a boolean `grounded` flag, and evidence IDs used.
- **Why Chosen:** Allows the pipeline to intercept ungrounded generations programmatically before displaying them to the user.
- **Alternative Rejected:** Streaming plain text responses. Rejected because it precludes post-generation safety checks.

### 10. LLM-as-a-Judge Evaluation Component
- **Chosen Approach:** Used a robust LLM prompt offline to grade Correctness, Relevance, and Groundedness, carefully calibrated against a 50-item human-labelled subset.
- **Why Chosen:** Human evaluation of RAG outputs across hundreds of queries is prohibitively slow and expensive.
- **Alternative Rejected:** Using automated metrics like BLEU or ROUGE. Rejected because these metrics correlate incredibly poorly with actual human judgment for conversational Q&A.

### 11. Human vs Automated Metric Separation
- **Chosen Approach:** Explicitly segregated ground-truth human annotations from automated classification metrics and LLM-judge proxy scores in all documentation.
- **Why Chosen:** Prevents stakeholders from misinterpreting a high LLM-judge score as ground-truth customer satisfaction.
- **Alternative Rejected:** Blending the scores into a single proprietary "Quality Metric". Rejected because it destroys interpretability and obscures which subsystem (retrieval vs generation) is failing.
