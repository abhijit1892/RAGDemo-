"""Simple Groq chat wrapper"""

import os
from typing import List, Optional, Any
from groq import Groq


class GroqLLM:
    """
    Minimal Groq wrapper.
    Call: groq_llm.chat(messages=[{"role":"system","content":...}, {"role":"user","content":...}])
    """

    def __init__(self, model: str = "llama-3.3-70b-versatile", api_key: Optional[str] = None):
        self.model = model
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))

    def chat(self, messages: List[dict], max_tokens: int = 512, temperature: float = 0.0) -> str:
        """
        messages: list of {"role": "system"|"user"|"assistant", "content": "text"}
        Returns the model's text output (string).
        """
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
        except Exception as e:
            # Bubble up a clear error (keep it simple)
            raise RuntimeError(f"Groq API call failed: {e}")

        # parse best choice
        try:
            return resp.choices[0].message.content
        except Exception:
            # fallback to stringifying the response object
            return str(resp)
