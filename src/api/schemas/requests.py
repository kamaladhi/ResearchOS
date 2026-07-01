from pydantic import BaseModel

class AskRequest(BaseModel):
    question: str

class IngestRequest(BaseModel):
    query: str
    max_results: int = 1
