# Problem Statement

Hiver's take-home asks for an AI support agent for AmazonHelp conversations on Twitter/X. The agent must turn a short, noisy customer message into an operational decision: classify intent, retrieve similar historical Amazon support cases, draft a grounded response, and decide whether the case can be auto-handled or must be escalated.

TWCS is difficult because it is public social-support data rather than clean ticket data. Messages are short, emotional, threaded, multilingual in places, and often contain multiple issues. Customer and brand turns are separated across tweet IDs, and the notebook reconstructs usable pairs by joining inbound customer messages to brand responses.

AmazonHelp was selected because the notebook targets the AmazonHelp support subset and its support volume covers common operational categories such as delivery, returns, refunds, Prime, and account support.

Good means operational safety, not just a high classifier score. A useful system should correctly identify the primary customer issue, retrieve relevant historical resolutions, refuse unsupported claims, and escalate ambiguous, sensitive, or weakly supported cases.

Out of scope:

- Live order, payment, refund, or account APIs.
- Guaranteed full thread understanding.
- Production serving infrastructure and monitoring.
- Proving zero leakage or zero hallucination.
- Multilingual response generation.

Formal problem statement:

Given a customer tweet from the AmazonHelp domain, predict one frozen intent label, retrieve non-leaking historical customer-brand interactions relevant to that issue, synthesize a response constrained by verified evidence, and choose `AUTO_HANDLE` only when confidence, evidence quality, and safety checks are sufficient.

## System objective

Customer message
-> intent
-> historical evidence
-> grounded response
-> automation/escalation decision.

Next: [Data and task definition](02_data_and_task_definition.md).
