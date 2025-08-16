import uuid
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.graph import app_graph
from schema import ChatRequest, ChatResponse
from src.vectorstore import build_vectorstores

build_vectorstores()

# FastAPI application
app = FastAPI()

@app.post("/chat")
def chat_endpoint(req: ChatRequest) -> ChatResponse:
    config = {"configurable": {"thread_id": req.thread_id or str(uuid.uuid4())}}

    for _ in app_graph.stream(
        {"user_question": req.user_question},
        config=config,
        stream_mode="updates"
    ):
        pass
    final_state = app_graph.get_state(config)

    return ChatResponse(answer=final_state.values.get("answer"))

# Add CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)