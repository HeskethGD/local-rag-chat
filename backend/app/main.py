from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from .models import Body
from .agents.chat import run_agent as chat_agent 
from .agents.rag import run_agent as rag_agent

app = FastAPI()

@app.post("/chat")
def chat(body: Body):
    """
    Generates an LLM stream response to a user question.

    Parameters:
    - body (Body): Messages from the chat history.

    Returns:
    - (StreamingResponse): Streaming response of assistant message.
    """

    messages = body.messages
    if messages == None or messages == []:
        return None
    try:
        return StreamingResponse(chat_agent(messages), media_type="text/html")
    except HTTPException as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    
@app.post("/rag")
def rag(body: Body):
    """
    Generates an LLM stream response to a user question using RAG over a vector db.

    Parameters:
    - body (Body): Messages from the chat history.

    Returns:
    - (StreamingResponse): Streaming response of assistant message.
    """

    messages = body.messages
    if messages == None or messages == []:
        return None
    try:
        return StreamingResponse(rag_agent(messages), media_type="text/html")
    except HTTPException as e:
        raise HTTPException(status_code=500, detail="Internal server error")