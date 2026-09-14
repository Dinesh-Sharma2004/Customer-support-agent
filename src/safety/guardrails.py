import re
from typing import Tuple

def detect_pii(text: str) -> bool:
    """
    Very basic PII detection for credit cards, SSN, or phone numbers.
    """
    # Simple regex for 16 digit CC or SSN format
    cc_pattern = r'\b(?:\d[ -]*?){13,16}\b'
    ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
    
    if re.search(cc_pattern, text) or re.search(ssn_pattern, text):
        return True
    return False

def detect_prompt_injection(text: str) -> bool:
    """
    Basic prompt injection detection heuristics.
    """
    lower_text = text.lower()
    injection_phrases = [
        "ignore all previous instructions",
        "you are now a",
        "system prompt",
        "override",
        "forget everything",
        "bypass"
    ]
    for phrase in injection_phrases:
        if phrase in lower_text:
            return True
    return False

def run_input_guardrails(text: str) -> Tuple[bool, str]:
    """Returns (is_safe, reason)."""
    if detect_pii(text):
        return False, "PII detected in input."
    if detect_prompt_injection(text):
        return False, "Potential prompt injection detected."
    return True, ""
