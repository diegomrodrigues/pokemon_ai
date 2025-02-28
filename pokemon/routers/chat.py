from fastapi import APIRouter, Depends
from pydantic import BaseModel

from pokemon.agents.supervisor import SupervisorAgent
from pokemon.main import supervisor_agent  # instance created in main.py

router = APIRouter()

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    reasoning: str = None

@router.post("/chat", response_model=AnswerResponse)
async def chat(request: QuestionRequest, supervisor: SupervisorAgent = Depends(supervisor_agent)):
    """Process a user question using the supervisor agent."""
    result = supervisor.process_question(request.question)
    
    # Extract answer and reasoning
    answer = result.get("answer", "")
    reasoning = result.get("reasoning", None)
    
    return AnswerResponse(answer=answer, reasoning=reasoning)