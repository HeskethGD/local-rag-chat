from typing import Any, Generator
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import requests
from app.models import Body
import json


def llm_handler(user_question: str) -> Generator[Any, Any, None]:
    

    # Send the POST request with streaming enabled
    model = "llama3.2"
    url = "http://localhost:11434/api/generate"
    with requests.post(url, json={"model": model, "prompt": user_question}, stream=True) as r:
        # Ensure the response is successful
        r.raise_for_status()
        
        # Process the streamed response
        for chunk in r.iter_lines(decode_unicode=True):
            if chunk:  # Skip empty lines
                try:
                    # Parse JSON chunk
                    data = json.loads(chunk)
                    # Yield the "response" field if it exists
                    yield data.get("response", "")
                except json.JSONDecodeError:
                    # Handle decoding errors gracefully
                    yield f"Error decoding response chunk: {chunk}"

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
        # Only consider the last message for now
        user_question = messages[-1].content
        return StreamingResponse(llm_handler(user_question), media_type="text/html")

    except HTTPException as e:
        raise HTTPException(status_code=500, detail="Internal server error")
    