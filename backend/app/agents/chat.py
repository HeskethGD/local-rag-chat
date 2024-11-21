import json
from typing import List
import requests
from dotenv import load_dotenv
import os
from ..models import Message, ContentStream

load_dotenv()
llm_model = os.getenv("LLM_MODEL")
llm_generate_url = os.getenv("LLM_GENERATE_URL")

def run_agent(messages: List[Message],) -> ContentStream:

    # Get the latest user message
    user_question = messages[-1].content

    # Send the POST request with streaming enabled
    with requests.post(
            llm_generate_url, 
            json={
                "model": llm_model, 
                "prompt": user_question
            }, 
            stream=True
        ) as r:
        # Ensure the response is successful
        r.raise_for_status()
        # Process the streamed response
        for chunk in r.iter_lines(decode_unicode=True):
            if chunk:
                try:
                    data = json.loads(chunk)
                    yield data.get("response", "")
                except json.JSONDecodeError:
                    yield f"Error decoding response chunk: {chunk}"