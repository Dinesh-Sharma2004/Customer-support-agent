from typing import Protocol, Any
import json
from src.config.settings import settings
import groq

class LLMProvider(Protocol):
    def generate(self, prompt: str, **kwargs) -> str:
        ...

class MockProvider(LLMProvider):
    def generate(self, prompt: str, **kwargs) -> str:
        return json.dumps({
            "answer": "This is a mock generated answer based on the evidence.",
            "grounded": True,
            "confidence": 0.95,
            "refusal_or_escalation_reason": None
        })

class GroqProvider(LLMProvider):
    def __init__(self, api_key: str, model_name: str):
        self.client = groq.Groq(api_key=api_key)
        self.model_name = model_name
        
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        except Exception as e:
            return json.dumps({
                "answer": f"Groq API Error: {str(e)}",
                "grounded": False,
                "confidence": 0.0,
                "refusal_or_escalation_reason": "API Error"
            })

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        
    def generate(self, prompt: str, **kwargs) -> str:
        return json.dumps({
            "answer": "Gemini generated answer (stub).",
            "grounded": True,
            "confidence": 0.9,
            "refusal_or_escalation_reason": None
        })

def get_llm_client() -> LLMProvider:
    provider_name = settings.LLM_PROVIDER.lower()
    
    if provider_name == "groq" and settings.GROQ_API_KEY:
        return GroqProvider(api_key=settings.GROQ_API_KEY, model_name=settings.LLM_MODEL_NAME)
    elif provider_name == "gemini" and settings.GEMINI_API_KEY:
        return GeminiProvider(settings.GEMINI_API_KEY)
    else:
        return MockProvider()
