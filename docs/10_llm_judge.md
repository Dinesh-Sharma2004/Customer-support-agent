# LLM Judge

The notebook implements an LLM-as-judge using Groq, not Claude.

Verified notebook configuration:

- SDK: `groq`
- Judge model variable: `GROQ_JUDGE_MODEL = "qwen/qwen3.6-27B"`
- Required source marker for valid results: `judge_source == "GROQ_LLM"`
- The notebook explicitly says Groq failures are evaluation failures and should not be silently replaced by fake LLM scores.
- A deterministic fallback is not valid for final judge evaluation.

The judge is separate from the classifier and the support agent:

| Component | Role |
|---|---|
| Classifier | Predicts one of six intents. |
| Agent response path | Retrieves evidence and produces a conservative response. |
| LLM-as-judge | Evaluates response quality after the response is produced. |

The judge prompt asks for structured JSON scoring across response-quality dimensions. The notebook contains schema validation, JSON extraction, retry logic, and API failure handling.

No saved `judge_scores.csv` or `judge_scores_groq_only.csv` artifact is present in the current artifact directory. Therefore current Groq judge results, average scores, and human agreement statistics are not verified.

Next: [Human annotation and bias validation](11_human_annotation_and_bias_validation.md).
