# Pydantic model for the request body

from pydantic import BaseModel

# Define pydantic model for structured output from the LLM
class IntentResponse(BaseModel):
    node: str

class ChatRequest(BaseModel):
    user_question: str
    thread_id: str|None = None # Optional thread ID for new conversations

class ChatResponse(BaseModel):
    answer: str
    # supporting_documents: list[str]