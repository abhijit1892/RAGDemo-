from typing import List
from pydantic import BaseModel, Field
from langchain.schema import Document

class RAGState(BaseModel):
    question: str
    retrieved_docs: List[Document] = Field(default_factory=list)
    answer: str = ""