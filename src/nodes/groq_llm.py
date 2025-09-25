"""Simple Groq chat wrapper"""

import os
import traceback
from typing import List, Optional
from groq import Groq


class GroqLLM:
    """
    Minimal Groq wrapper for chat completions.
    Usage:
        groq_llm = GroqLLM()
        answer = groq_llm.chat(messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is Article 370?"}
        ])
    """

    def __init__(self, model: str = "llama3-70b-8192", api_key: Optional[str] = None):
        """
        Args:
            model: Groq model name (valid options: "llama3-70b-8192", "llama3-8b-8192", "mixtral-8x7b-32768")
            api_key: optional override for Groq API key (else uses env var GROQ_API_KEY)
        """
        self.model = model
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        if not self.client.api_key:
            raise RuntimeError("Groq API key missing. Set GROQ_API_KEY in environment or pass api_key explicitly.")

    def chat(self, messages: List[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
        """
        Send chat-style messages to the Groq LLM.

        Args:
            messages: list of {"role": "system"|"user"|"assistant", "content": str}
            max_tokens: maximum tokens in the reply
            temperature: randomness (0.0 = deterministic)

        Returns:
            str: model output text
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content

        except Exception as e:
            # Log full traceback for debugging in Streamlit/Cloud
            traceback.print_exc()
            raise RuntimeError(f"Groq API call failed: {e}")
