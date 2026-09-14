from pydantic import BaseModel, Field
from typing import Optional
from src.generation.llm_client import get_llm_client
from src.memory.store import Message
import json

class GenerationResult(BaseModel):
    answer: str
    evidence_ids: list[str]
    grounded: bool
    confidence: float
    refusal_or_escalation_reason: Optional[str] = None

class RAGGenerator:
    def __init__(self):
        self.llm = get_llm_client()
        
    def generate(self, user_query: str, evidence: list[dict], context: list[Message]) -> GenerationResult:
        """
        Receives user request, permitted conversation context, and verified evidence.
        Produces a concise support answer.
        """
        if not evidence:
            return GenerationResult(
                answer="I'm sorry, I don't have enough information to help with that.",
                evidence_ids=[],
                grounded=False,
                confidence=0.0,
                refusal_or_escalation_reason="Missing evidence."
            )
            
        evidence_text = "\n".join([f"[{d['doc_id']}] {d['brand_response']}" for d in evidence])
        context_text = "\n".join([f"{m.role}: {m.content}" for m in context])
        
        prompt = f"""
You are an AmazonHelp Customer Support agent. Answer the user's query using ONLY the provided evidence.

EVIDENCE:
{evidence_text}

CONVERSATION CONTEXT:
{context_text}

USER QUERY:
{user_query}

Respond with a JSON object matching this schema:
{{
  "answer": "your concise response",
  "grounded": true/false (is your answer fully supported by the evidence?),
  "confidence": 0.0-1.0,
  "refusal_or_escalation_reason": "reason" or null
}}
"""
        try:
            raw_response = self.llm.generate(prompt)
            # Find JSON if wrapped in markdown
            if "```json" in raw_response:
                raw_response = raw_response.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_response:
                raw_response = raw_response.split("```")[1].strip()
                
            data = json.loads(raw_response)
            
            return GenerationResult(
                answer=data.get('answer', ''),
                evidence_ids=[d['doc_id'] for d in evidence],
                grounded=data.get('grounded', False),
                confidence=float(data.get('confidence', 0.0)),
                refusal_or_escalation_reason=data.get('refusal_or_escalation_reason')
            )
        except Exception as e:
            return GenerationResult(
                answer="System error during generation.",
                evidence_ids=[],
                grounded=False,
                confidence=0.0,
                refusal_or_escalation_reason=f"Generation failed: {str(e)}"
            )
