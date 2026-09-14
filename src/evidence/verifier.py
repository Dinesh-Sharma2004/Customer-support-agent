from typing import Tuple

def verify_evidence(query: str, retrieved_docs: list[dict], min_score: float = 0.5) -> Tuple[bool, str, list[dict]]:
    """
    Checks whether retrieved evidence actually supports the query and detects contradictions.
    Returns (is_verified, reason, filtered_evidence).
    
    In a real system, this could use an LLM or cross-encoder. 
    Here we use heuristic thresholds for prototype.
    """
    if not retrieved_docs:
        return False, "No evidence retrieved.", []
        
    valid_docs = [doc for doc in retrieved_docs if doc.get('score', 0) >= min_score or doc.get('rrf_score', 0) > 0.0]
    
    if not valid_docs:
        return False, "Retrieved evidence scores too low.", []
        
    # Basic check for contradictory responses in the top docs
    responses = [doc['brand_response'].lower() for doc in valid_docs]
    if any("yes" in r for r in responses) and any("no" in r for r in responses):
        return False, "Contradictory evidence detected.", []
        
    return True, "Evidence verified.", valid_docs
