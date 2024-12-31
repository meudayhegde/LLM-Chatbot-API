from typing import List, Literal, Optional
from pydantic import BaseModel


class Message(BaseModel):
    role: Literal['user', 'assistant']
    content: str
    key: Optional[str]

class Conversation(BaseModel):
    conversation: List[Message]