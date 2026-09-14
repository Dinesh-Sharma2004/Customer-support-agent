# Golden Evaluation

The Golden set is intended to be a locked, human-reviewed 200-example final evaluation set. The notebook verifies its expected row count and SHA-256 hash before using it.

Verified current Golden artifact:

- File: `artifacts/amazonhelp_golden_set_final.csv`
- Rows: 200
- Columns: 26
- Expected SHA in notebook: `3811c3dadb1435af71bd9885a855bed7cd7503a43367e587b6cfdd4e9070453b`
- Valid taxonomy rows found by inspection: 198
- Invalid/missing `primary_intent` rows found by inspection: 2

Observed `primary_intent` distribution:

| Intent | Count |
|---|---:|
| Other / Human Review | 87 |
| Order & Delivery Issues | 69 |
| Refunds & Financials | 14 |
| Returns & Exchanges | 11 |
| Prime & Membership | 10 |
| Account Security & Private Support | 7 |
| Invalid `en` | 1 |
| Missing | 1 |

The notebook code computes classification accuracy, macro-F1, weighted-F1, balanced accuracy, auto/escalation rates, unsupported-response rate, grounding-failure rate, retrieval Hit@5, and intent-agreement proxy metrics. The expected row-level Golden evaluation artifact `golden_evaluation_row_level.csv` is not present in the current artifact directory, so final Golden numbers are not verified.

## Evaluation Caveats

- The Golden set is evaluation-only; any tuning on it invalidates final claims.
- Two rows currently need intent-label cleanup or explicit exclusion.
- Golden sampling is not proven representative of production traffic.
- Response quality is separate from intent accuracy.
- Retrieval intent-match is a proxy, not proof of evidence correctness.

Next: [LLM judge](10_llm_judge.md).
