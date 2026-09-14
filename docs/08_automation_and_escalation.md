# Automation And Escalation

The automation policy combines classifier confidence, evidence quality, grounding, and safety constraints.

Architecture:

```text
Intent confidence
+
Evidence quality
+
Safety constraints
+
Grounding validation
        |
AUTO_HANDLE / ESCALATE
```

Escalation is forced or strongly preferred when:

- classifier confidence is below the selected validation threshold;
- the predicted intent is `Account Security & Private Support`;
- the issue is ambiguous or multi-issue;
- evidence is missing, partial, conflicting, or too generic;
- grounding checks fail;
- the response would require private account/order/refund knowledge.

The notebook searches automation policy pairs on a validation sample with confidence thresholds and minimum evidence levels, optimizing auto-handle correctness while enforcing unsupported-response rate constraints. However, `policy_search.csv` and `final_config.json` are not present in the current artifact directory, so exact final policy threshold values are not verified from artifacts.

Escalation is not automatically a model failure. In this project, escalation is a safety action: it can be the correct behavior for sensitive, ambiguous, or unsupported cases. The tradeoff is safety versus analyst workload.

Next: [Golden evaluation](09_golden_evaluation.md).
