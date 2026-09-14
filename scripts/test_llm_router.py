import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import RAGPipeline
from src.config.settings import settings

def run_tests():
    pipeline = RAGPipeline()
    
    test_cases = [
        {
            "desc": "(1) Normal customer-support question with strong TWCS evidence -> AUTO_HANDLE",
            "query": "Where is my package? It was supposed to be delivered yesterday."
        },
        {
            "desc": "(2) Low classifier confidence but strong retrieved evidence -> AUTO_HANDLE",
            # We use a weird phrasing that might confuse TF-IDF but is still a delivery issue
            "query": "The transportation vehicle carrying my items has been delayed indefinitely?"
        },
        {
            "desc": "(3) Ambiguous query -> CLARIFY",
            "query": "It didn't work."
        },
        {
            "desc": "(4) Request requiring private account access -> ESCALATE",
            "query": "Can you change the credit card on my account to end in 1234?"
        },
        {
            "desc": "(5) Sensitive-intent prediction where actual question can be answered from TWCS without account access -> AUTO_HANDLE",
            # Intent likely "Refunds & Financials", but just asking about general policy
            "query": "What is the general refund policy for electronics?"
        },
        {
            "desc": "(6) Insufficient/contradictory retrieval evidence -> ESCALATE",
            # A query about a made up policy that shouldn't be in TWCS
            "query": "Does Amazon reimburse me for the emotional damage of a late package?"
        }
    ]
    
    print(f"Running LLM Router Tests using {settings.LLM_PROVIDER}...\n")
    
    for i, tc in enumerate(test_cases):
        print(f"Test {i+1}: {tc['desc']}")
        print(f"Query: '{tc['query']}'")
        
        res = pipeline.process(f"router_test_{i}", tc['query'], is_new_session=True)
        
        print(f"Predicted Intent: {res['intent']} (Conf: {res['confidence']:.2f})")
        print(f"Decision:         {res['decision']}")
        print(f"Reason:           {res['reason']}")
        print(f"Answer:           {res['answer']}")
        print("-" * 80 + "\n")

if __name__ == "__main__":
    # Force Groq for this test
    os.environ["LLM_PROVIDER"] = "groq"
    run_tests()
