import sys
import os
from fastapi import APIRouter
from api.schemas.requests import AskRequest
from api.schemas.responses import AskResponse
from api.core.exceptions import GraphRAGException

# Add parent dir to path to import engine
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from agents import run_agentic_workflow

router = APIRouter(prefix="/ask", tags=["Chat"])

@router.post("/", response_model=AskResponse)
async def ask_question(request: AskRequest):
    try:
        result = run_agentic_workflow(request.question)
        return AskResponse(
            question=request.question,
            answer=result.answer,
            citations=result.citations,
            follow_up_questions=result.follow_up_questions
        )
    except Exception as e:
        raise GraphRAGException(str(e))
