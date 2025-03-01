from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Optional

from pokemon.agents.supervisor import SupervisorAgent

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    reasoning: Optional[str] = None

@router.post("/chat", response_model=AnswerResponse)
async def chat(request: QuestionRequest):
    """Process a user question using the supervisor agent."""
    supervisor = SupervisorAgent()
    result = supervisor.process_question(request.question)
    
    # Extract answer and reasoning
    answer = result.get("answer", "")
    reasoning = result.get("reasoning", None)
    
    return AnswerResponse(answer=answer, reasoning=reasoning)