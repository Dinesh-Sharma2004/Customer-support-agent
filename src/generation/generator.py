from pydantic import BaseModel, Field
from typing import Optional
from src.generation.llm_client import get_llm_client
from src.memory.store import Message
import json

class GenerationResult(BaseModel):
    decision: str
    answer: str
    reason: str
    evidence_ids: list[str]
    grounded: bool
    confidence: float

class RAGGenerator:
    def __init__(self):
        self.llm = get_llm_client()
        
    def generate(self, user_query: str, evidence: list[dict], context: list[Message], predicted_intent: str, classifier_confidence: float) -> GenerationResult:
        """
        Receives user request, permitted conversation context, verified evidence, and classifier signals.
        Produces a decision (AUTO_HANDLE, CLARIFY, ESCALATE) and a response.
        """
        evidence_text = "\n".join([f"[{d['doc_id']}] {d['brand_response']}" for d in evidence]) if evidence else "No retrieved evidence available."
        context_text = "\n".join([f"{m.role}: {m.content}" for m in context]) if context else "No prior context."
        
        prompt = f"""
You are a customer-support decision and response agent for Amazon. Your job is to determine whether the user's query can be safely and helpfully handled using the provided information.

INPUT SIGNALS:
1. Classifier Predicted Intent: {predicted_intent} (Confidence: {classifier_confidence:.2f})
2. Conversation Context:
{context_text}
3. Retrieved Customer-Support History (TWCS Evidence):
{evidence_text}

USER QUERY:
{user_query}

INSTRUCTIONS:
1. The classifier intent and confidence are supporting signals, not a final routing decision. The classifier labels are weakly supervised and may be incorrect.
2. Do not escalate merely because the classifier confidence is below 0.70, the predicted intent is "Other / Human Review", or the predicted intent is a sensitive category (e.g., Refunds, Security).
3. If the retrieved evidence provides sufficient information to handle this type of issue, return "AUTO_HANDLE" and provide a concise, professional, empathetic response directly grounded in the evidence. DO NOT invent policies, facts, account information, or actions.
4. If the query is understandable but the available evidence is insufficient or contradictory, return "ESCALATE" and briefly explain why.
5. If the request requires genuinely sensitive capabilities (e.g., accessing or changing private account information, identity verification, financial authorization, security intervention), return "ESCALATE" and explain the limitation. However, if the question can be answered safely without human access (e.g., general policy questions), do not escalate.
6. If the request is ambiguous but could likely be resolved by asking the customer for one specific missing piece of information, return "CLARIFY" and ask that question instead of escalating.

Respond with a JSON object matching this schema strictly:
{{
  "decision": "AUTO_HANDLE" | "CLARIFY" | "ESCALATE",
  "answer": "Your concise response to the user, or clarification question, or empty string if escalating",
  "reason": "Brief explanation of why you chose this decision",
  "grounded": true/false (is your answer fully supported by the evidence?),
  "confidence": 0.0-1.0 (your confidence in the decision and answer)
}}
"""
        try:
            raw_response = self.llm.generate(prompt)
            if "```json" in raw_response:
                raw_response = raw_response.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_response:
                raw_response = raw_response.split("```")[1].strip()
                
            data = json.loads(raw_response)
            
            return GenerationResult(
                decision=data.get('decision', 'ESCALATE'),
                answer=data.get('answer', ''),
                reason=data.get('reason', ''),
                evidence_ids=[d['doc_id'] for d in evidence] if evidence else [],
                grounded=data.get('grounded', False),
                confidence=float(data.get('confidence', 0.0))
            )
        except Exception as e:
            return GenerationResult(
                decision="ESCALATE",
                answer="System error during generation.",
                reason=f"Generation failed: {str(e)}",
                evidence_ids=[],
                grounded=False,
                confidence=0.0
            )
