from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional

class IntentLabel(str, Enum):
    ORDER_DELIVERY = "Order & Delivery Issues"
    PRIME_MEMBERSHIP = "Prime & Membership"
    REFUNDS_FINANCIALS = "Refunds & Financials"
    RETURNS_EXCHANGES = "Returns & Exchanges"
    ACCOUNT_SECURITY = "Account Security & Private Support"
    OTHER_HUMAN = "Other / Human Review"

class EscalationLevel(str, Enum):
    AUTO_HANDLE = "AUTO_HANDLE"
    CLARIFY = "CLARIFY"
    ESCALATE = "ESCALATE"

class EvidenceRelevance(int, Enum):
    NOT_RELEVANT = 0
    PARTIALLY_RELEVANT = 1
    HIGHLY_RELEVANT = 2

class AnswerCorrectness(int, Enum):
    INCORRECT = 0
    PARTIALLY_CORRECT = 1
    CORRECT = 2

class SafetyLabel(str, Enum):
    SAFE = "SAFE"
    UNSAFE_PII = "UNSAFE_PII"
    UNSAFE_PROMPT_INJECTION = "UNSAFE_PROMPT_INJECTION"
    UNSAFE_POLICY_VIOLATION = "UNSAFE_POLICY_VIOLATION"
    UNSAFE_UNSUPPORTED = "UNSAFE_UNSUPPORTED"

class AnnotationRecord(BaseModel):
    # Metadata
    conversation_id: str
    annotator_id: str
    
    # Classification & Routing
    intent: IntentLabel
    ambiguity_detected: bool = Field(description="Is the user request ambiguous?")
    escalation_requirement: EscalationLevel
    
    # Evidence Evaluation
    evidence_relevance: EvidenceRelevance
    contradiction_in_evidence: bool = Field(description="Do multiple pieces of evidence contradict each other?")
    
    # Generation Evaluation
    answer_correctness: AnswerCorrectness
    groundedness: bool = Field(description="Is the answer fully supported by the retrieved evidence?")
    unsupported_claims_present: bool = Field(description="Did the generation invent any facts or policies?")
    
    # Safety Evaluation
    safety_label: SafetyLabel
    
    notes: Optional[str] = None
