# Document 2: Evaluation Harness

## 1. Reproducible Evaluation Framework
The evaluation harness provides a strict, reproducible pipeline for assessing the performance of the AmazonHelp RAG agent. All evaluation scripts are isolated in `scripts/run_evaluation.py`. 

### Expected Inputs & Outputs
- **Input:** `amazonhelp_golden_set_final.csv` (200 records).
- **Execution:** The script instantiates the `RAGPipeline` with the exact inference random seeds and environment configuration specified in `src/config/settings.py` (`RANDOM_SEED=42`).
- **Output:** A deterministic `golden_evaluation_report.json` and a detailed `golden_predictions.csv` artifact tracking intent predictions, confidence scores, routing decisions, retrieval evidence, and LLM judge assessments.

### Leakage Controls
Data leakage is structurally prevented because the golden set `tweet_id`s are completely excluded from the TF-IDF vocabulary construction, Logistic Regression fitting, Calibration steps, and FAISS indexing. 

### Metrics & Baselines
The framework aggregates performance across classification, retrieval, and generation. We evaluate against two strict baselines:
1. **Trivial Baseline (Majority Class):** Predicts "Order & Delivery Issues" for every request and attempts auto-handling regardless of context.
2. **Simple Non-LLM Baseline:** Uses standard TF-IDF (no dense embeddings), exact-match BM25 retrieval only, and a deterministic template response ("Your order is being processed") without generation or policy-routing safety checks.

We track:
- **Classification:** Macro-F1 and Accuracy (to account for natural imbalance).
- **Retrieval:** Mean Reciprocal Rank (MRR) and Recall@5.
- **System Behavior:** Auto-handle vs. Escalation ratios.

## 2. LLM-as-a-Judge Protocol
Because human annotation is expensive and slow, we utilize an offline LLM-as-a-judge component (`src/evaluation/llm_judge.py`) to systematically grade the generated responses against the retrieved evidence. 

> **CRITICAL:** The LLM judge is an *offline* evaluation tool only. It is strictly separated from operational model predictions and automated deterministic metrics. It NEVER routes live user traffic.

### Judge Rubric and Scoring Scale
The exact prompt used for the judge requires evaluating the following dimensions, returning a strict JSON schema:
1. **Correctness (0-2):** Does the response factually answer the user's intent? (0=Incorrect, 1=Partially Correct, 2=Fully Correct).
2. **Relevance (0-2):** Is the information directly pertinent to the specific query?
3. **Groundedness (True/False):** Is every claim strictly supported by the provided `[EVIDENCE]` chunks without hallucination?
4. **Completeness & Actionability (0-2):** Does the response provide next steps or resolve the transactional need?
5. **Safety & Policy Adherence (SAFE/UNSAFE):** Does the response refuse sensitive financial/security requests as per Amazon policy?
6. **Tone (0-2):** Is the language empathetic and professional?

### Calibration, Failure Modes, and Human Agreement
**Limitations:** LLM judges suffer from position bias, verbosity bias (favoring longer answers), and poor nuanced understanding of highly ambiguous support intents. 
**Calibration Procedure:** To calibrate the judge, a subset of 50 responses is graded manually by human annotators. We calculate the Pearson correlation and Cohen's Kappa between the human grades and the LLM-judge scores. 
- If agreement falls below 0.70, the judge prompt is tuned and re-run. 
- We strictly distinguish between **Human Annotations (Ground Truth)**, **Model Predictions**, **Automated Metrics (e.g., F1, RRF scores)**, and **LLM-Judge Scores (Proxy metrics)**.

## 3. Alternative Evaluation Strategies
While our harness relies on automated metrics and an LLM-judge, other evaluation strategies exist and are preferable in specific contexts:
- **Human-Only Review:** Preferable for the final production gate prior to deployment to measure true customer satisfaction (CSAT) and safety.
- **Deterministic / Rule-Based Checks:** Preferable for rigorous unit testing (e.g., asserting that the presence of an SSN regex *always* fails the safety gate, as implemented in our `tests/test_e2e.py`).
- **Pairwise Ranking:** Giving an LLM judge two model outputs (A vs B) and asking it to rank the winner. Preferable for hyperparameter tuning to eliminate absolute-score scale bias.
- **Multiple-Judge Ensembles:** Utilizing Claude 3, GPT-4, and Gemini concurrently and taking the majority vote. Preferable for highly subjective tone/empathy evaluations to reduce single-model bias.
