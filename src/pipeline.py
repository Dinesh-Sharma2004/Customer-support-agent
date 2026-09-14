from typing import Tuple, Dict, Any
import os

from src.config.settings import settings
from src.safety.guardrails import run_input_guardrails
from src.memory.store import SessionMemory
from src.memory.session import SessionContextManager
from src.classification.model import load_model
from src.retrieval.bm25_index import BM25Retriever
from src.retrieval.dense_index import DenseRetriever
from src.retrieval.hybrid_search import HybridRetriever
from src.evidence.verifier import verify_evidence
from src.generation.generator import RAGGenerator

class RAGPipeline:
    def __init__(self):
        # 1. Load memory
        self.memory_store = SessionMemory()
        self.session_manager = SessionContextManager(self.memory_store)
        
        # 2. Load classifiers
        self.classifier = load_model("calibrated_classifier.joblib")
        
        # 3. Load Retrievers
        self.bm25 = BM25Retriever()
        self.bm25.load(os.path.join(settings.ARTIFACTS_DIR, "bm25_index.joblib"))
        
        self.dense = DenseRetriever()
        self.dense.load(os.path.join(settings.ARTIFACTS_DIR, "dense_index.faiss"))
        
        self.hybrid = HybridRetriever(self.bm25, self.dense)
        
        # 4. Generator
        self.generator = RAGGenerator()
        
    def process(self, session_id: str, query: str, is_new_session: bool = False) -> Dict[str, Any]:
        """
        Executes the full RAG pipeline: 
        Input Guardrails -> Session Memory -> TF-IDF Classification -> 
        Hybrid Retrieval -> Evidence Verification -> LLM Decision + Response -> 
        Post-Generation Safety -> Final Decision
        """
        response_dict = {
            "intent": None,
            "confidence": 0.0,
            "decision": "ESCALATE",
            "reason": "",
            "answer": "",
            "evidence_ids": [],
            "grounded": False
        }
        
        # 1. Input Safety
        is_safe, reason = run_input_guardrails(query)
        if not is_safe:
            response_dict["reason"] = reason
            return response_dict
            
        # 2. Memory Context
        with self.session_manager.context(session_id, is_new=is_new_session) as memory:
            memory.add_message(session_id, "user", query)
            history = memory.get_history(session_id)
            
            # 3. Classification (Routing Signal Only)
            predicted_intent = self.classifier.predict([query])[0]
            probabilities = self.classifier.predict_proba([query])[0]
            classes = self.classifier.classes_
            confidence = float(probabilities[list(classes).index(predicted_intent)])
            
            response_dict["intent"] = predicted_intent
            response_dict["confidence"] = confidence
                
            # 4. Hybrid Retrieval
            retrieved_docs = self.hybrid.search(query, top_k=5)
            
            # 5. Evidence Verification (Filters out low-quality/contradictory evidence)
            is_verified, reason, verified_docs = verify_evidence(query, retrieved_docs)
                
            # 6. LLM Generation and Decision
            gen_result = self.generator.generate(
                user_query=query, 
                evidence=verified_docs, 
                context=history,
                predicted_intent=predicted_intent,
                classifier_confidence=confidence
            )
            
            response_dict["decision"] = gen_result.decision
            response_dict["reason"] = gen_result.reason
            response_dict["answer"] = gen_result.answer
            response_dict["evidence_ids"] = gen_result.evidence_ids
            response_dict["grounded"] = gen_result.grounded
            
            # 7. Post-Generation Safety
            if response_dict["decision"] == "AUTO_HANDLE" and not gen_result.grounded:
                response_dict["decision"] = "ESCALATE"
                response_dict["reason"] = "Generated response not fully grounded."
            
            # 8. Memory Update
            if response_dict["decision"] in ["AUTO_HANDLE", "CLARIFY"]:
                memory.add_message(session_id, "assistant", gen_result.answer)
                
            return response_dict
