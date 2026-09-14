from typing import Protocol, Any
import json
from src.config.settings import settings

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

class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str):
        # We would initialize the real client here
        self.api_key = api_key
        
    def generate(self, prompt: str, **kwargs) -> str:
        # Pseudo-code for actual API call
        # response = genai.GenerativeModel('gemini-1.5-flash').generate_content(prompt)
        # return response.text
        return json.dumps({
            "answer": "Gemini generated answer (stub).",
            "grounded": True,
            "confidence": 0.9,
            "refusal_or_escalation_reason": None
        })

def get_llm_client() -> LLMProvider:
    provider_name = settings.LLM_PROVIDER.lower()
    
    if provider_name == "gemini" and settings.GEMINI_API_KEY:
        return GeminiProvider(settings.GEMINI_API_KEY)
    else:
        # Default to mock to avoid coupling or failures
        return MockProvider()
