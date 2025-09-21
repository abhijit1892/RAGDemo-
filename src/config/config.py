"""Configuration module for Agentic RAG system"""

import os
from dotenv import load_dotenv
from src.nodes.groq_llm import GroqLLM

load_dotenv()


class Config:
    """Configuration class for RAG system"""

    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    LLM_MODEL = "llama-3.3-70b-versatile"

    # Document Processing
    CHUNK_SIZE = 800
    CHUNK_OVERLAP = 100

    # Default Sources (Indian Constitution)
    DEFAULT_URLS = [
    "https://legislative.gov.in/constitution-of-india",       # Govt source
    "https://en.wikipedia.org/wiki/Constitution_of_India",     # General overview
    "https://www.constitutionofindia.net/",                   # History + drafting notes
    "https://prsindia.org/billtrack/constitution-amendments", # Amendments tracker
    "https://indiankanoon.org/doc/237570/ "                   # Bare text on Indian Kanoon
    ]
    DEFAULT_PDFS = [
        "data/constitution.pdf"   # put your PDF here
    ]

    @classmethod
    def get_llm(cls) -> GroqLLM:
        return GroqLLM(model=cls.LLM_MODEL, api_key=cls.GROQ_API_KEY)
