import streamlit as st
import os
from typing import List, TypedDict
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage
import faiss
import tempfile

# Page configuration
st.set_page_config(page_title="RAG-Powered Trackbot", page_icon="🤖", layout="wide")

@st.cache_resource
def initialize_rag_components():
    """Initialize RAG components once and cache them."""
    try:
        # Initialize embeddings
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
        
        # Load knowledge base document
        knowledge_base_path = "knowledge base.docx"
        if not os.path.exists(knowledge_base_path):
            st.error(f"Knowledge base file '{knowledge_base_path}' not found!")
            return None, None
        
        # Load and process the knowledge base document
        loader = UnstructuredWordDocumentLoader(knowledge_base_path)
        docs = loader.load()
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits = text_splitter.split_documents(docs)
        
        # Create FAISS vector store from documents
        vector_store = FAISS.from_documents(all_splits, embedding_model)
        
        return embedding_model, vector_store
    except Exception as e:
        st.error(f"Error initializing RAG components: {e}")
        return None, None

@st.cache_resource
def load_llm():
    """Load and cache the Gemini LLM."""
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=st.secrets["GOOGLE_API_KEY"],
            convert_system_message_to_human=True
        )
    except Exception as e:
        st.error(f"Error loading LLM: {e}")
        return None

# Define RAG State
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def setup_rag_pipeline(vector_store, llm):
    """Set up the RAG pipeline using LangGraph."""
    
    def retrieve(state: State):
        """Retrieve relevant documents from vector store."""
        if not vector_store or not state.get("question"):
            return {"context": []}
        
        try:
            retrieved_docs = vector_store.similarity_search(state["question"], k=3)
            return {"context": retrieved_docs}
        except Exception as e:
            st.error(f"Error in retrieval: {e}")
            return {"context": []}

    def generate(state: State):
        """Generate answer using LLM and retrieved context."""
        if not llm or not state.get("question"):
            return {"answer": "Sorry, I couldn't process your question."}
        
        try:
            # Prepare context from retrieved documents
            docs_content = "\n\n".join(doc.page_content for doc in state.get("context", []))
            
            # Create prompt for the LLM
            if docs_content:
                prompt_text = f"""You are a helpful assistant. Use the following context to answer the user's question. If the context doesn't contain relevant information, say so.

Context:
{docs_content}

Question: {state["question"]}

Answer:"""
            else:
                prompt_text = f"""You are a helpful assistant. Answer the user's question to the best of your ability.

Question: {state["question"]}

Answer:"""

            # Get response from LLM
            response = llm.invoke([HumanMessage(content=prompt_text)])
            return {"answer": response.content}
        except Exception as e:
            st.error(f"Error in generation: {e}")
            return {"answer": "Sorry, I encountered an error while generating the response."}

    # Build RAG graph
    try:
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()
    except Exception as e:
        st.error(f"Error building RAG pipeline: {e}")
        return None

def main():
    st.title("🤖 RAG-Powered Trackbot")
    st.markdown("Chat with an AI that can reference your knowledge base!")
    
    # Initialize components
    embedding_model, vector_store = initialize_rag_components()
    llm = load_llm()
    
    if not embedding_model or not vector_store or not llm:
        st.error("Failed to initialize components. Please check your configuration and ensure 'knowledge base.docx' exists.")
        return
    
    # Sidebar for information and settings
    with st.sidebar:
        st.header("� Knowledge Base")
        
        # Show knowledge base status
        if embedding_model and vector_store:
            st.success("✅ Knowledge base loaded successfully!")
            st.info("Ready to answer questions about your documents.")
        else:
            st.error("❌ Failed to load knowledge base")
            st.error("Please ensure 'knowledge base.docx' exists in the app directory.")
        
        st.markdown("---")
        st.markdown("### ℹ️ How it works")
        st.markdown("""
        1. The app automatically loads 'knowledge base.docx'
        2. Ask questions about the content
        3. Get AI responses based on your knowledge base
        4. If no relevant content is found, the AI will use its general knowledge
        """)
        
        st.markdown("---")
        st.markdown("### 🔧 Features")
        st.markdown("""
        - **RAG-powered responses**: Uses your document content
        - **Source transparency**: Shows which parts of the document informed the response
        - **Fallback knowledge**: Uses Gemini's general knowledge when needed
        - **Persistent chat**: Maintains conversation history
        """)
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show source documents if available
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(message["sources"], 1):
                        st.write(f"**Source {i}:**")
                        st.write(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                        if hasattr(doc, 'metadata') and doc.metadata:
                            st.write(f"*Metadata: {doc.metadata}*")
                        st.write("---")
    
    # Chat input
    if prompt := st.chat_input("Ask me anything about your knowledge base..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Set up RAG pipeline
        rag_graph = setup_rag_pipeline(vector_store, llm)
        
        if rag_graph:
            # Process with RAG
            with st.chat_message("assistant"):
                with st.spinner("Searching knowledge base and generating response..."):
                    try:
                        # Run RAG pipeline
                        result = rag_graph.invoke({"question": prompt})
                        answer = result.get("answer", "Sorry, I couldn't generate a response.")
                        context_docs = result.get("context", [])
                        
                        # Display answer
                        st.markdown(answer)
                        
                        # Show sources if available
                        if context_docs:
                            with st.expander("📚 Source Documents Used"):
                                for i, doc in enumerate(context_docs, 1):
                                    st.write(f"**Source {i}:**")
                                    st.write(doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content)
                                    if hasattr(doc, 'metadata') and doc.metadata:
                                        st.write(f"*Metadata: {doc.metadata}*")
                                    st.write("---")
                        
                        # Add assistant response to chat history
                        assistant_message = {
                            "role": "assistant", 
                            "content": answer
                        }
                        if context_docs:
                            assistant_message["sources"] = context_docs
                        
                        st.session_state.messages.append(assistant_message)
                        
                    except Exception as e:
                        error_msg = f"Error processing your question: {e}"
                        st.error(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": error_msg
                        })
        else:
            st.error("Failed to set up RAG pipeline.")
    
    # Add clear chat button
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()
