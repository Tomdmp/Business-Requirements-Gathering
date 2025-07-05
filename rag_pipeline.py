def generate_response(prompt): 

  return output 

import os
from typing import List, Dict, Any

from langchain.chat_models import init_chat_model
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain import hub
from langgraph.graph import START, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI

# Optional: set GOOGLE_API_KEY as env variable before running
def setup_environment(api_key: str = None):
    """
    Setup environment variables required for Google Gemini API.
    """
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
    elif "GOOGLE_API_KEY" not in os.environ:
        raise ValueError("Google API key is required. Set 'GOOGLE_API_KEY' environment variable or pass api_key.")

def load_knowledge_base(doc_path: str) -> List[Document]:
    """
    Load and split knowledge base document into chunks.
    """
    loader = UnstructuredWordDocumentLoader(doc_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(docs)

def build_vector_store(documents: List[Document]):
    """
    Create embeddings and build FAISS vector store from documents.
    """
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_documents(documents, embedding_model)

def build_rag_prompt():
    """
    Load RAG prompt from Langchain Hub.
    """
    return hub.pull("rlm/rag-prompt")

# Define the RAG state dictionary type
from typing_extensions import TypedDict

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve_documents(vector_store, state: State) -> Dict[str, Any]:
    """
    Retrieve similar documents from vector store based on question.
    """
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate_answer(llm, prompt, state: State) -> Dict[str, Any]:
    """
    Generate an answer from LLM given the question and retrieved context.
    """
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

def build_langgraph(vector_store, llm) -> StateGraph:
    """
    Build LangGraph pipeline for RAG retrieval and generation.
    """

    # Tool for retrieval (example usage of langchain_core.tools)
    @tool(response_format="content_and_artifact")
    def retrieve(query: str):
        retrieved_docs = vector_store.similarity_search(query, k=2)
        serialized = "\n\n".join(
            (f"Source: {doc.metadata}\nContent: {doc.page_content}")
            for doc in retrieved_docs
        )
        return serialized, retrieved_docs

    # Step 1: Query or Respond node
    def query_or_respond(state):
        llm_with_tools = llm.bind_tools([retrieve])
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    # Step 2: Execute retrieval tool
    tools = ToolNode([retrieve])

    # Step 3: Generate answer node
    def generate(state):
        recent_tool_messages = []
        for message in reversed(state["messages"]):
            if message.type == "tool":
                recent_tool_messages.append(message)
            else:
                break
        tool_messages = recent_tool_messages[::-1]

        docs_content = "\n\n".join(doc.content for doc in tool_messages)
        system_message_content = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say so. "
            "Use three sentences max and keep it concise.\n\n"
            f"{docs_content}"
        )
        conversation_messages = [
            message
            for message in state["messages"]
            if message.type in ("human", "system") or (message.type == "ai" and not message.tool_calls)
        ]
        prompt = [SystemMessage(system_message_content)] + conversation_messages

        response = llm.invoke(prompt)
        return {"messages": [response]}

    graph_builder = StateGraph()
    graph_builder.add_node(query_or_respond)
    graph_builder.add_node(tools)
    graph_builder.add_node(generate)
    graph_builder.set_entry_point("query_or_respond")
    graph_builder.add_conditional_edges("query_or_respond", tools_condition, {END: END, "tools": "tools"})
    graph_builder.add_edge("tools", "generate")
    graph_builder.add_edge("generate", END)

    # Add memory checkpointing
    memory = MemorySaver()
    graph = graph_builder.compile(checkpointer=memory)

    return graph

def run_chat(graph: StateGraph, input_message: str, thread_id: str = "default_thread"):
    """
    Stream conversation with the graph given an input message.
    """
    config = {"configurable": {"thread_id": thread_id}}
    responses = []
    for step in graph.stream({"messages": [{"role": "user", "content": input_message}]}, stream_mode="values", config=config):
        responses.append(step["messages"][-1].content)
    return responses


# Example usage if run as script
if __name__ == "__main__":
    # Setup environment (set your API key here or export GOOGLE_API_KEY)
    setup_environment()

    # Load knowledge base and create vector store
    docs = load_knowledge_base("/content/knowledge base.docx")
    vector_store = build_vector_store(docs)

    # Initialize LLM
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

    # Load prompt
    prompt = build_rag_prompt()

    # Example state and run
    example_state: State = {"question": "What is the project timeline?", "context": [], "answer": ""}

    retrieved = retrieve_documents(vector_store, example_state)
    example_state.update(retrieved)

    generated = generate_answer(llm, prompt, example_state)
    example_state.update(generated)

    print("Answer:", example_state["answer"])
