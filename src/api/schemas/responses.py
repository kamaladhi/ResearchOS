from pydantic import BaseModel
from typing import List

class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[str]
    follow_up_questions: List[str]
