# Human Annotation And Bias Validation

The intended human annotation task is not intent labelling. It is response-quality evaluation for approximately 30-50 final AI responses using the fixed 0-3 rubric:

- `0`: unacceptable
- `1`: poor
- `2`: acceptable
- `3`: strong

The human reviewer should see:

- customer query
- AI-generated response
- retrieved historical evidence

The human reviewer must not see:

- Groq score
- Groq reasoning
- machine-generated proxy "human" labels

Golden intent annotation answers: "What is the customer's intent?"

Human response annotation answers: "How good is the AI response?"

These are different tasks. Golden intent labels are not sufficient to validate an LLM response-quality judge.

## Current Artifact Status

Verified file: `artifacts/human_response_annotations.csv`

- Rows: 30
- Columns: `tweet_id`, `gold_intent`, `response_text`, `human_overall`, `human_notes`
- Non-null `human_overall` values: 0
- Groq score/reasoning columns: absent

This means the annotation interface/file structure mitigates label leakage from Groq scores, but real human agreement validation is not yet established because no human scores are entered.

## Bias Analysis

| Bias | Risk | Mitigation Implemented | Validation | Remaining Risk |
|---|---|---|---|---|
| Anchoring bias | Human may copy the model score. | Human CSV omits Groq score/reasoning. | Verified by columns. | Reduced, not eliminated; reviewer could see scores elsewhere. |
| Confirmation bias | Reviewer may look for evidence supporting first impression. | Rubric asks reviewer to inspect query, response, and evidence. | Workflow intent documented; no completed scores to audit. | Not directly measurable yet. |
| Order bias | Earlier examples influence later scores. | Not verified. | No randomization artifact found. | Remaining risk. |
| Leniency/severity bias | Single reviewer may score too high/low. | Not verified. | No entered scores or multi-reviewer data. | Remaining risk. |
| Label leakage | Groq output biases human score. | Groq columns absent from human annotation file. | Verified. | Reduced, not eliminated. |
| Rubric ambiguity | Reviewers interpret 0-3 differently. | Fixed rubric documented. | No inter-annotator data. | Remaining risk. |
| Selection bias | 30 examples may not represent Golden distribution. | Sample has Golden IDs/intents, but representativeness not proven. | Can inspect intents; no statistical validation. | Remaining risk. |

## Agreement Analysis

Exact agreement and Cohen's kappa must compare genuine `GROQ_LLM` scores against independent human scores. The current artifact has no human scores and no Groq-only score file, so:

Human agreement validation not yet established.

Next: [Failure analysis](12_failure_analysis.md).
