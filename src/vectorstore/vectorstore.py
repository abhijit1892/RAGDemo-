"""Vector store module for document embedding and retrieval"""

from typing import List
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document


class RetrieverWrapper:
    """Wrap a LangChain retriever to expose .invoke(query) like your existing code"""
    def __init__(self, retriever):
        self._retriever = retriever

    def invoke(self, query: str, k: int = 4) -> List[Document]:
        # LangChain retrievers typically expose get_relevant_documents
        return self._retriever.get_relevant_documents(query, k=k)


class VectorStore:
    """Manages vector store operations"""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize vector store with a HuggingFace sentence-transformer
        Force CPU to avoid CUDA errors
        """
        self.embedding = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"}   # 👈 force CPU here
        )
        self.vectorstore = None
        self.retriever = None

    def create_vectorstore(self, documents: List[Document]):
        """
        Create vector store from documents

        Args:
            documents: List of langchain.schema.Document
        """
        self.vectorstore = FAISS.from_documents(documents, self.embedding)
        # get a retriever object and wrap it
        retr = self.vectorstore.as_retriever(search_kwargs={"k": 4})
        self.retriever = RetrieverWrapper(retr)

    def get_retriever(self):
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever

    def retrieve(self, query: str, k: int = 4) -> List[Document]:
        if self.retriever is None:
            raise ValueError("Vector store not initialized. Call create_vectorstore first.")
        return self.retriever.invoke(query)
