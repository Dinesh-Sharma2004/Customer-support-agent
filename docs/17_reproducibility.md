# Reproducibility

Primary execution artifact: `HiverAssignment_FINAL__1_.ipynb`.

Required data:

- `twcs/twcs.csv`
- `artifacts/amazonhelp_golden_set_final.csv`

Verified saved artifacts currently present:

- `artifacts/benchmark_table.csv`
- `artifacts/calibration_results.csv`
- `artifacts/human_response_annotations.csv`
- `artifacts/final_classifier/champion_classifier.joblib`
- `artifacts/final_classifier/calibrated_classifier.joblib`

Expected but not currently present:

- `artifacts/retrieval_eval.csv`
- `artifacts/policy_search.csv`
- `artifacts/final_config.json`
- `artifacts/rag_corpus.csv`
- `artifacts/faiss_index.bin`
- `artifacts/corpus_embs.npy`
- `artifacts/golden_evaluation_row_level.csv`
- `artifacts/judge_scores.csv`
- `artifacts/judge_scores_groq_only.csv`
- `artifacts/failure_modes.csv`

Dependencies used by the notebook include `pandas`, `numpy`, `scikit-learn`, `joblib`, `rank_bm25`, `faiss-cpu`, `sentence-transformers`, `torch`, `matplotlib`, `seaborn`, and `groq`.

Environment variables:

- `GROQ_API_KEY` should be provided through the environment or notebook secret management.
- The notebook currently contains a hard-coded Groq key string in source. That should be removed and rotated before sharing.

Execution order:

1. Install dependencies.
2. Place TWCS under the path expected by the notebook or adjust `DATA_PATH`.
3. Place Golden CSV in `artifacts/`.
4. Run notebook sections in order: data construction, leakage exclusion, splits, classifier, calibration, retrieval, policy, Golden evaluation, judge evaluation.
5. Verify expected artifacts exist before reporting final metrics.

Runtime is not verified. Do not claim a specific runtime such as under 15 minutes unless measured on the target machine.

Next: [Current status](18_current_status.md).
