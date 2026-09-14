# Week One Next Steps

| Priority | Problem | Action | Why Now | Effort | Expected Impact | Risk/Tradeoff | Success Measure |
|---|---|---|---|---|---|---|---|
| 1 | Missing verified final artifacts | Rerun notebook end-to-end and save `retrieval_eval.csv`, `policy_search.csv`, `final_config.json`, `golden_evaluation_row_level.csv`, `judge_scores_groq_only.csv`, `failure_modes.csv`. | Documentation and interview claims need traceability. | 0.5-1 day | Converts "not verified" into measured claims. | Runtime/API cost. | All expected artifacts exist and match docs. |
| 2 | Golden label defects | Fix or explicitly exclude invalid `primary_intent` rows. | Evaluation metrics need clean denominators. | 1-2 hours | Cleaner final metrics. | Changing Golden requires versioning. | 200 valid labels or documented 198-row eval. |
| 3 | Human judge validation absent | Complete 30-50 independent human response ratings. | LLM judge validity is currently unestablished. | 0.5-1 day | Enables real agreement/kappa. | Small sample still limited. | Non-null human scores and agreement report. |
| 4 | Ambiguous/multi-issue routing | Add explicit multi-issue detector and route to escalation. | Likely high-risk failure class. | 1 day | Better safety on mixed issues. | More false escalations. | Lower incorrect auto-handle rate on flagged cases. |
| 5 | Evidence quality uncertainty | Save evidence packs and verification outcomes per Golden row. | Response quality cannot be audited otherwise. | 1 day | Stronger grounding analysis. | Larger artifacts. | Each auto-handled response has auditable evidence. |
| 6 | Rare class weakness | Add targeted labels for refunds, Prime, security. | Current Golden counts are small. | 1-2 days | Better per-class estimates and training signal. | Annotation effort. | Improved validation macro-F1/per-class recall. |
| 7 | Single-turn limitation | Add thread reconstruction experiment. | Twitter support often depends on context. | 1-2 days | Better ambiguous follow-up handling. | More leakage complexity. | Compare single-turn vs thread-aware validation. |

These are hypotheses and engineering priorities, not promised metric gains.

Next: [Architecture summary](16_architecture_summary.md).
