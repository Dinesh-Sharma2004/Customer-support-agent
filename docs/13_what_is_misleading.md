# What Is Misleading

A single headline number is not enough for this project.

## Model Accuracy

Classifier accuracy measures whether the predicted intent equals the Golden intent. It does not say whether retrieved evidence is relevant, whether the generated response is useful, or whether unsafe cases are escalated.

Class imbalance matters. The current Golden artifact has 87 `Other / Human Review` rows and 69 `Order & Delivery Issues` rows, while some classes have 7-14 examples. Macro-F1 and per-class recall are more informative than accuracy alone.

## End-To-End Customer Safety

A wrong intent can be safe if the system escalates before sending a response. A correct intent can still be unsafe if the response invents an order status, refund amount, or account action.

Safety should be measured with unsupported-response rate, grounding-failure rate, security escalation behavior, and incorrect auto-handle rate.

## Automation Quality

Auto-handle rate can be gamed by sending too many responses. Escalation rate can be gamed by escalating everything. The useful metric is correct auto-handle rate under a safety constraint, plus the analyst workload created by false escalations.

## Judge Limitations

LLM-as-judge scores are only credible after validation against independent human response-quality annotations. The current human annotation artifact has no completed scores, so judge-human agreement is not established.

Bottom line: report classifier metrics, retrieval metrics, response-quality metrics, and safety/automation metrics separately.

Next: [Decision log](14_decision_log.md).
