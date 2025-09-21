from langgraph.graph import StateGraph, END
from src.state.rag_state import RAGState
from src.nodes.reactnode import RAGNodes


class GraphBuilder:
    def __init__(self, retriever, llm):
        # pass retriever and llm into the nodes
        self.nodes = RAGNodes(retriever, llm)
        self.graph = None

    def build(self):
        builder = StateGraph(RAGState)

        builder.add_node("retriever", self.nodes.retrieve_docs)
        builder.add_node("responder", self.nodes.generate_answer)

        builder.set_entry_point("retriever")
        builder.add_edge("retriever", "responder")
        builder.add_edge("responder", END)

        self.graph = builder.compile()
        return self.graph

    def run(self, question: str):
        if self.graph is None:
            self.build()
        initial_state = RAGState(question=question)
        return self.graph.invoke(initial_state)
