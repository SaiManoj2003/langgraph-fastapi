import os
from typing import TypedDict
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from schema import IntentResponse
from src.llm import llm, embeddings

class GraphState(TypedDict):
    user_question: str
    intent: str
    retrieved_context: list[tuple[Document, float]]
    answer: str

def intent_check(state: GraphState) -> GraphState:
    # The prompt and structured output will return a string like "langgraph_retriever"
    question = state['user_question']
    intent_prompt = f"""
        You are an intent classification assistant.
        Given the user's question, decide which retriever node should handle it.
        Rules:
        - If the question is about LangGraph, workflows, agents, state, or edges -> return: "langgraph_retriever"
        - If the question is about Semantic Kernel, plugins, skills, or planners -> return: "semantic_kernel_retriever"
        Respond ONLY in the following JSON format, no extra text:
        {{
        "node": "<node_name>"
        }}
        Question: {question}
    """
    llm_structured = llm.with_structured_output(IntentResponse)
    intent = llm_structured.invoke(intent_prompt)
    return {**state, "intent": intent.node}

def langgraph_retriever(state: GraphState) -> GraphState:
    base_path = os.path.dirname(os.path.dirname(__file__))
    db = FAISS.load_local(
        os.path.join(base_path, "faiss_index_langgraph"),
        embeddings,
        allow_dangerous_deserialization=True
    )
    docs = db.similarity_search_with_score(state['user_question'], k=3)
    return {**state, "retrieved_context": [(doc, float(score)) for doc, score in docs]}

def semantic_kernel_retriever(state: GraphState) -> GraphState:
    base_path = os.path.dirname(os.path.dirname(__file__))
    db = FAISS.load_local(
        os.path.join(base_path, "faiss_index_semantic_kernel"),
        embeddings,
        allow_dangerous_deserialization=True
    )
    docs = db.similarity_search_with_score(state['user_question'], k=3)
    return {**state, "retrieved_context": [(doc, float(score)) for doc, score in docs]}

def generate_answer(state: GraphState) -> GraphState:
    """Generate an answer based on the retrieved context."""
    context = "\n".join(doc.page_content for doc , _ in state['retrieved_context'])
    generate_prompt = f"""
                You are a helpful assistant.
                You will only use the context provided to you to answer the user question.
                context:{context}

                question:{state['user_question']}
                """
    answer_msg = llm.invoke(generate_prompt).content

    supporting_content = "\n\n### Supporting Documents ###\n\n"
    for doc, score in state['retrieved_context']:
        supporting_content += f"Document: {doc.page_content}\nScore: {score}\n\n"

    return {**state, "answer": f"{answer_msg}\n\n{supporting_content}"}

# --- Graph Setup ---
graph_builder = StateGraph(GraphState)
graph_builder.add_node("intent_check", intent_check)
graph_builder.add_node("langgraph_retriever", langgraph_retriever)
graph_builder.add_node("semantic_kernel_retriever", semantic_kernel_retriever)
graph_builder.add_node("generate_answer", generate_answer)

graph_builder.set_entry_point("intent_check")
graph_builder.add_conditional_edges(
    "intent_check",
    lambda state: state["intent"],
    {
        "langgraph_retriever": "langgraph_retriever",
        "semantic_kernel_retriever": "semantic_kernel_retriever",
    }
)

graph_builder.add_edge("langgraph_retriever", "generate_answer")
graph_builder.add_edge("semantic_kernel_retriever", "generate_answer")
graph_builder.add_edge("generate_answer", END)

# Compile the graph with the checkpointer
checkpointer = MemorySaver()
app_graph = graph_builder.compile(checkpointer=checkpointer)