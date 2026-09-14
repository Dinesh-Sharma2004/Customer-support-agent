from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional

from src.pipeline import RAGPipeline
from src.config.settings import settings

app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION)
pipeline = RAGPipeline()

class SupportRequest(BaseModel):
    session_id: str
    query: str
    is_new_session: bool = False

class SupportResponse(BaseModel):
    intent: str
    confidence: float
    decision: str
    reason: Optional[str]
    answer: str
    evidence_ids: list[str]
    grounded: bool

@app.post("/support", response_model=SupportResponse)
def handle_support_request(req: SupportRequest):
    try:
        res = pipeline.process(req.session_id, req.query, req.is_new_session)
        return SupportResponse(
            intent=res.get("intent", "Unknown"),
            confidence=res.get("confidence", 0.0),
            decision=res.get("decision", "ESCALATE"),
            reason=res.get("reason"),
            answer=res.get("answer", ""),
            evidence_ids=res.get("evidence_ids", []),
            grounded=res.get("grounded", False)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "ok", "version": settings.APP_VERSION}

@app.get("/version")
def get_version():
    return {"version": settings.APP_VERSION}
