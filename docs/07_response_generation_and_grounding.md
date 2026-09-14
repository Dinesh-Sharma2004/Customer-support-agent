# Response Generation And Grounding

The final response path in the notebook is evidence-constrained and conservative. Response generation is deterministic synthesis from verified evidence and fallback messages, not an unconstrained agent LLM.

Flow:

```text
retrieved cases
-> evidence pack
-> evidence verification
-> deterministic grounded response
-> grounding/safety check
-> AUTO_HANDLE or ESCALATE
```

The generator should not invent:

- order numbers
- refund amounts
- account information
- exact dates
- eligibility
- completed actions
- guarantees
- policy details unsupported by evidence

Good grounded response:

> Thanks for reaching out. Similar resolved cases were routed through return support, so the safest next step is to have support review the return details before giving item-specific instructions.

Bad unsupported response:

> Your refund has already been processed and will arrive tomorrow.

The second response is unsafe because the system has no live refund/account data and no verified basis for a completed action or date.

The notebook contains stale comments saying an Anthropic/Claude verifier could be used with deterministic fallback. The current judge implementation is Groq/Qwen, and the evidence verifier behavior should be described as deterministic or not verified unless a real verifier artifact is produced.

Next: [Automation and escalation](08_automation_and_escalation.md).
