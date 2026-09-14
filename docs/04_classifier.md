# Classifier

The implemented classifier is supervised learning over heuristic intent labels from non-Golden AmazonHelp pairs.

Architecture:

```text
Customer text
    |
Word TF-IDF ----.
                +--> FeatureUnion --> Logistic Regression --> intent probabilities
Char TF-IDF ----'
                                      --> predicted intent + confidence
```

Verified implementation:

- Word TF-IDF: ngrams `(1, 2)`, `max_features=80000`, `sublinear_tf=True`, `min_df=2`.
- Character TF-IDF: `char_wb`, ngrams `(3, 5)`, `max_features=50000`, `sublinear_tf=True`, `min_df=5`.
- Model: multinomial `LogisticRegression`, `C=5`, `max_iter=1000`, `class_weight="balanced"`, `solver="lbfgs"`, `random_state=42`.
- Saved artifacts verified: `artifacts/final_classifier/champion_classifier.joblib` and `artifacts/final_classifier/calibrated_classifier.joblib`.

Validation benchmarks verified from `artifacts/benchmark_table.csv`:

| Approach | Validation Metric | Why Accepted/Rejected |
|---|---:|---|
| Word+Char TF-IDF + LR | Accuracy 0.8299, Macro-F1 0.7298, Weighted-F1 0.8357, Bal-Acc 0.7470 | Accepted as champion among saved benchmark rows. |
| Unigram TF-IDF + LR | Accuracy 0.7571, Macro-F1 0.6423, Weighted-F1 0.7785, Bal-Acc 0.7304 | Rejected; weaker than word+char fusion. |
| Majority Class | Accuracy 0.6648, Macro-F1 0.1331, Weighted-F1 0.5309, Bal-Acc 0.1667 | Rejected; exposes class imbalance and poor minority handling. |

The major expected confusion is `Order & Delivery Issues` vs `Other / Human Review`, especially when the message is vague, multi-issue, or does not contain enough operational detail. Rare classes such as `Account Security & Private Support`, `Refunds & Financials`, and `Prime & Membership` have fewer Golden examples, so per-class Golden claims should be reported cautiously.

Next: [Calibration](05_calibration_and_confidence.md).
