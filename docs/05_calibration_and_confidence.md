# Calibration And Confidence

Raw logistic-regression probability is not automatically a reliable automation signal. The notebook therefore compares post-hoc calibration methods using a separate calibration split and evaluates them on validation data.

Verified calibration candidates from `artifacts/calibration_results.csv`:

| Method | Brier | LogLoss | ECE |
|---|---:|---:|---:|
| Isotonic | 0.204537 | 0.454036 | 0.037474 |
| Sigmoid | 0.207590 | 0.460652 | 0.054491 |
| Uncalibrated | 0.264693 | 0.532334 | 0.040394 |

The notebook selects the best method by sorting validation calibration metrics. Isotonic has the best Brier score and log loss in the saved artifact, and the calibrated classifier artifact exists.

Confidence is later used by retrieval gating and automation policy. The notebook tests candidate intent-confidence thresholds `[0.55, 0.60, 0.65, 0.70, 0.75, 0.80]` for intent-aware retrieval on validation data. The final selected values are not directly verifiable from a saved `final_config.json` or `policy_search.csv` in the current artifact directory, so any exact final threshold should be marked not verified unless the notebook is rerun and those artifacts are produced.

Correct tuning sequence:

```text
candidate thresholds
-> validation evaluation
-> selected operating point
-> frozen threshold
-> Golden evaluation
```

Golden must not be used to choose thresholds. Doing so would turn the Golden set into another validation set and contaminate final evaluation.

Next: [RAG and retrieval](06_rag_and_retrieval.md).
