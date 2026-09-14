# Decision Log

| Decision | Alternatives Considered | Chosen | Why | Evidence | Tradeoff | Validation | Remaining Risk |
|---|---|---|---|---|---|---|---|
| Use AmazonHelp | Other TWCS brands | AmazonHelp | High support volume and diverse issue families. | Notebook filters AmazonHelp-like pairs. | Brand filtering is heuristic. | Data construction code. | Exact brand identity not fully proven. |
| Six-class taxonomy | Fine-grained labels, triage-only labels | Six operational classes | Balances actionability and sparsity. | Notebook `FROZEN_TAXONOMY`. | Loses nuance. | Used throughout classifier/eval code. | Ambiguous cases remain hard. |
| Include `Other / Human Review` | Force every case into specific intent | Explicit human-review class | Safer for ambiguity and unsupported cases. | Taxonomy and Golden distribution. | Can absorb too much data. | Golden counts. | Needs careful annotation discipline. |
| Word+char TF-IDF | Unigram only, majority baseline | Word+char fusion | Captures words and noisy tweet spellings. | `benchmark_table.csv`. | Sparse lexical model. | Validation metrics. | Weak multilingual semantics. |
| Logistic Regression | Deep fine-tuning | Balanced multinomial LR | Fast, reproducible, CPU-friendly. | Notebook code. | Limited representation power. | Validation benchmark. | May miss semantic edge cases. |
| Class balancing | No class weights | `class_weight="balanced"` | Counters heuristic-label imbalance. | Notebook code. | Can over-weight rare/noisy labels. | Champion validation. | Rare-class metrics still uncertain. |
| Calibration | Trust raw probabilities | Isotonic/Sigmoid/Uncalibrated comparison | Confidence drives safety. | `calibration_results.csv`. | Extra calibration split. | Validation Brier/log loss/ECE. | Exact final policy threshold missing. |
| Hybrid retrieval | BM25 only, dense only | BM25 + FAISS + RRF | Combines lexical and semantic matching. | Notebook code. | More moving parts. | Retrieval eval code exists. | Retrieval scores artifact missing. |
| RRF | Score averaging | Reciprocal Rank Fusion | Simple rank-level fusion. | Notebook code. | Ignores calibrated retrieval scores. | Intended validation. | Exact benefit not verified. |
| Candidate/top-k retrieval | Single nearest case | Candidate generation then top evidence | Reduces dependence on one retrieval signal. | Notebook functions. | More latency. | Not fully artifact-verified. | Evidence quality can still be weak. |
| Leakage policy | Full thread isolation | ID/text/immediate-parent exclusion | Auditable and simple. | Notebook code. | Not zero leakage. | Golden overlap checks in code. | Near duplicates remain. |
| Separate classifier/retriever representations | One embedding model for all | TF-IDF classifier plus MiniLM retrieval | Optimizes separate tasks independently. | Notebook code. | More complexity. | Validation artifacts partly present. | Representation mismatch possible. |
| Security escalation | Confidence-based security handling | Always escalate security intent | Avoids private/account harm. | Policy code/comments. | More analyst workload. | Policy logic. | Security misclassification still possible. |
| No fake final judge fallback | Deterministic substitute | Only genuine `GROQ_LLM` valid | Prevents false validation claims. | Groq judge code. | Eval can fail without API. | Source marker checks. | No saved Groq results now. |
| Golden lock | Tune on Golden | Evaluation-only Golden | Prevents p-hacking. | Notebook warnings/hash check. | Smaller tuning data. | Integrity code. | CSV has two invalid labels. |

Next: [One-week next steps](15_week_one_next_steps.md).
