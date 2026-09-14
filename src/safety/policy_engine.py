from typing import Tuple

SENSITIVE_INTENTS = [
    "Account Security & Private Support",
    "Refunds & Financials"
]

def apply_routing_policy(intent: str, confidence: float, threshold: float = 0.7) -> Tuple[str, str]:
    """
    Determines if a request should be AUTO_HANDLE, CLARIFY, or ESCALATE.
    Returns (Decision, Reason).
    
    Safety constraints override classification:
    - Sensitive intents always escalate.
    - Low confidence escalates or clarifies.
    """
    if intent in SENSITIVE_INTENTS:
        return "ESCALATE", f"Sensitive intent detected: {intent}."
        
    if intent == "Other / Human Review":
        return "ESCALATE", "Intent requires human review."
        
    if confidence < threshold:
        return "ESCALATE", f"Low confidence prediction ({confidence:.2f} < {threshold})."
        
    return "AUTO_HANDLE", ""
