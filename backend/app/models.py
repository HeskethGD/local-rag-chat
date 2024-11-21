from pydantic import BaseModel
from typing import AsyncIterable, Iterable, List, Union

Content = Union[str, bytes]
SyncContentStream = Iterable[Content]
AsyncContentStream = AsyncIterable[Content]
ContentStream = Union[AsyncContentStream, SyncContentStream]

class Message(BaseModel):
    """
    LLM Message object.

    Attributes:
        role (str): The role of the message e.g. 'system', 'assistant' or 'user'
        content (str): The main content of the messgae
    """
    role: str
    content: str

class Body(BaseModel):
    """
    The payload sent in a request to the FastAPI endpoint

    Attributes:
        messages (List[Message]): List of messages from the chat
    """
    messages: List[Message]