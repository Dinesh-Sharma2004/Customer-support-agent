# Failure Analysis

The notebook defines code for five failure modes from Golden errors, but the saved `failure_modes.csv` and row-level Golden evaluation artifact are not present. Counts, rates, and representative examples are therefore not verified.

Current evidence-supported failure modes should be phrased as candidates until row-level artifacts exist:

| Failure Mode | Number/Rate | Root Cause Hypothesis | Escalation Mitigation | Recommended Fix |
|---|---:|---|---|---|
| Ambiguous or multi-issue messages misclassified as specific intents | Not verified | Single-label classifier compresses multiple customer problems into one label. | Escalation can help if ambiguity or low confidence is detected. | Add explicit multi-intent detection and human-review routing. |
| Order/Delivery vs Returns boundary confusion | Not verified | Replacement, damaged delivery, and return language overlap. | Evidence checks may catch unsupported responses. | Improve labels and add boundary examples. |
| Account Security underperformance or risky drafting | Not verified | Rare class and high-stakes private-support wording. | Policy escalates security intent. | Do not auto-send security/account responses. |
| Non-English or transliterated tweets | Not verified | TF-IDF classifier is lexical and mostly English-oriented. | Low confidence may escalate. | Add language-aware routing or multilingual model/data. |
| Refunds vs Returns confusion | Not verified | Refunds often appear in return workflows. | Escalation can prevent unsupported money claims. | Add financial-specific examples and features. |

For each final failure mode, the project should include: count/rate, representative `tweet_id`, customer query, correct behavior, observed behavior, root-cause hypothesis, whether escalation caught it, and a measurable fix.

Next: [Misleading headline](13_what_is_misleading.md).
