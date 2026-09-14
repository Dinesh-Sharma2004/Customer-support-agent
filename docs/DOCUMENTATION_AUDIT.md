# Documentation Audit

## Stale Claims Removed

- Claude judge claims removed; notebook uses Groq with `qwen/qwen3.6-27B`.
- Fixed `0.6` automation threshold claims marked not verified unless final artifacts are produced.
- `74%`, `68.2%`, `82%`, and `kappa = 0.78` claims not preserved as current verified results.
- Production-ready, zero-leakage, hallucination-free, and fully conversation-aware language removed.
- Runtime claims such as under 15 minutes marked not verified.

## Notebook / Documentation Inconsistencies Found

- Root Markdown describes RoBERTa classifier behavior, but notebook implements word+char TF-IDF plus logistic regression.
- Root Markdown references artifacts such as final Golden evaluation results, retrieval indexes, and judge scores that are not present in the current `artifacts/` directory.
- Notebook comments still mention Anthropic/Claude-style verification in places, while the judge implementation uses Groq/Qwen.
- Human annotation is described as validation, but the current human annotation artifact contains no human scores.

## Claims That Could Not Be Validated

- Final Golden accuracy, macro-F1, weighted-F1, balanced accuracy.
- Auto-handle rate, escalation rate, correct auto-handle rate, unsupported-response rate, grounding-failure rate.
- Retrieval Hit@5/MRR and hybrid-vs-BM25/dense numeric gains.
- Final selected automation threshold and minimum evidence level.
- Failure-mode counts/rates and representative examples.
- Groq score distributions and judge-human agreement.

## Metrics Whose Source Artifacts Were Verified

- Classifier validation benchmark from `artifacts/benchmark_table.csv`.
- Calibration metrics from `artifacts/calibration_results.csv`.
- Golden set row count, columns, and intent distribution from `artifacts/amazonhelp_golden_set_final.csv`.
- Human annotation file structure and blank score status from `artifacts/human_response_annotations.csv`.

## Human Annotation Methodology Verified

- The annotation CSV contains `tweet_id`, `gold_intent`, `response_text`, `human_overall`, and `human_notes`.
- Groq score and reasoning columns are absent from that file.
- No independent human scores have been entered, so methodology exists only as an unfinished workflow.

## Bias Mitigations Actually Implemented

- Label leakage from Groq scores is reduced in the human annotation CSV because Groq outputs are absent.
- Independent score entry is structurally supported by blank `human_overall` and `human_notes` columns.

## Bias Risks That Remain

- No completed annotations to measure leniency/severity, confirmation bias, order bias, or rubric consistency.
- No multi-annotator data.
- No statistical basis to claim absence of bias.
- Sample representativeness is not established.

## Final System Status

The core ML/RAG pipeline is substantially implemented in the notebook, and classifier/calibration artifacts are present. Final end-to-end evaluation, retrieval metrics, policy selection artifacts, Groq judge outputs, human agreement, and failure-mode counts are not currently verified from saved artifacts. The honest project status is implemented but only partially validated.
