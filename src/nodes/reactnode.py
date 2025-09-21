from typing import List, Optional
from src.state.rag_state import RAGState
from langchain_core.documents import Document


class RAGNodes:
    def __init__(self, retriever, llm):
        """
        retriever: object that has .invoke(query) -> List[Document]
        llm: GroqLLM instance (with .chat(messages=[...]) -> str)
        """
        self.retriever = retriever
        self.llm = llm

    def retrieve_docs(self, state: RAGState) -> RAGState:
        docs = self.retriever.invoke(state.question)
        return RAGState(question=state.question, retrieved_docs=docs)

    def generate_answer(self, state: RAGState) -> RAGState:
        # Gather retrieved docs into a context block
        docs: List[Document] = state.retrieved_docs or []
        context_parts = []
        for i, d in enumerate(docs[:6], start=1):
            meta = getattr(d, "metadata", {}) or {}
            title = meta.get("source") or f"doc_{i}"
            snippet = (d.page_content or "").strip().replace("\n", " ")
            context_parts.append(f"[{i}] {title}: {snippet}")

        context = "\n\n".join(context_parts) if context_parts else "No retrieved context available."

        # Prompt designed for natural language answers
        system_prompt = (
            "You are a helpful legal assistant for the Indian Constitution.\n"
            "You will be given excerpts from the Constitution and amendments.\n"
            "Answer the user's question in clear, natural language, as if explaining to a student or lawyer.\n"
            "Do not just copy raw text — explain its meaning, background, and implications.\n"
            "When relevant, cite the Article or Amendment number, and indicate the supporting source numbers in square brackets.\n"
            "If you cannot find the answer in the sources, say 'I don’t know based on the provided documents.'\n"
        )

        user_prompt = (
            f"Context from retrieved documents:\n{context}\n\n"
            f"Question: {state.question}\n\n"
            f"Answer in detail, with explanation and citations:"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        answer_text = self.llm.chat(messages=messages, max_tokens=1500, temperature=0.2)

        return RAGState(
            question=state.question,
            retrieved_docs=state.retrieved_docs,
            answer=answer_text or "No answer generated"
        )
