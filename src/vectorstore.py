import os
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from src.llm import embeddings

def build_vectorstores():
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=500,
        chunk_overlap=0,
        is_separator_regex=False
    )

    base_path = os.path.dirname(os.path.dirname(__file__))  # project root
    data_path = os.path.join(base_path, "data")

    # LangGraph
    langgraph_index = os.path.join(base_path, "faiss_index_langgraph")
    if not os.path.exists(langgraph_index):
        docs = PyPDFLoader(os.path.join(data_path, "LangGraph_Overview.pdf")).load()
        texts = text_splitter.split_documents(docs)
        FAISS.from_documents(texts, embeddings).save_local(langgraph_index)

    # Semantic Kernel
    semantic_index = os.path.join(base_path, "faiss_index_semantic_kernel")
    if not os.path.exists(semantic_index):
        docs = PyPDFLoader(os.path.join(data_path, "Semantic_Kernel.pdf")).load()
        texts = text_splitter.split_documents(docs)
        FAISS.from_documents(texts, embeddings).save_local(semantic_index)