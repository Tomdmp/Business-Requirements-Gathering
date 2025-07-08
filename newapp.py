import streamlit as st
import os
import json
import re
from typing import List, TypedDict, Dict, Any
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langgraph.graph import START, StateGraph
from langchain_core.messages import HumanMessage
from langchain.schema import SystemMessage

# Page configuration
st.set_page_config(page_title="RAG-Powered Trackbot", page_icon="🤖", layout="wide")

def load_txt(file_path):
    """Load text content from a file."""
    try:
        with open(file_path, encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return ""
    
# Load prompts from external files  
system_prompt = SystemMessage(content=load_txt("system_prompt.txt"))
input_instruction = SystemMessage(content=load_txt("input_prompt.txt"))

@st.cache_resource
def initialize_rag_components():
    """Initialize RAG components once and cache them."""
    try:
        # Initialize embeddings
        embedding_model = HuggingFaceEmbeddings(model_name='all-mpnet-base-v2')
        
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
    extracted_data: Dict[str, Any]
    missing_fields: List[str]
    clarification_questions: List[str]

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
        
    def generate_answer(state: State):
        """Analyze communication dump and extract structured information."""
        if not llm or not state.get("question"):
            return {"answer": "Sorry, I couldn't process your communication dump."}
        
        try:
            # Prepare context from retrieved documents (knowledge base requirements)
            docs_content = "\n\n".join(doc.page_content for doc in state.get("context", []))
            
            # Create specialized prompt for communication analysis
            if docs_content:
                prompt_text = f""" {input_instruction.content} 
                Knowledge Base  (use this to understand what data is needed):
                {docs_content}

                Communication Dump to Analyze:
                {state["question"]}"""
            else:
                prompt_text = f""" {input_instruction.content} 
                Communication Dump to Analyze:
                {state["question"]}"""

            # Get response from LLM
            response = llm.invoke([HumanMessage(content=prompt_text)])
            
            # Extract structured information from response
            extracted_data = extract_structured_data(response.content)
            missing_fields = extract_missing_fields(response.content)
            clarification_questions = extract_clarification_questions(response.content)

            return {
                "answer": response.content,
                "extracted_data": extracted_data,
                "missing_fields": missing_fields,
                "clarification_questions": clarification_questions
            }
        except Exception as e:
            st.error(f"Error in analysis: {e}")
            return {
                "answer": "Sorry, I encountered an error while analyzing the communication dump.",
                "extracted_data": {},
                "missing_fields": [],
                "clarification_questions": []
            }
        
    # Build RAG graph
    try:
        graph_builder = StateGraph(State).add_sequence([retrieve, generate_answer])
        graph_builder.add_edge(START, "retrieve")
        return graph_builder.compile()
    except Exception as e:
        st.error(f"Error building RAG pipeline: {e}")
        return None
    
def extract_structured_data(response_text: str) -> Dict[str, Any]:
    """Extract structured data from AI response."""
    extracted_data = {}
    
    # Look for EXTRACTED DATA section
    if "EXTRACTED DATA:" in response_text:
        extracted_section = response_text.split("EXTRACTED DATA:")[1]
        if "MISSING DATA:" in extracted_section:
            extracted_section = extracted_section.split("MISSING DATA:")[0]
        elif "QUESTIONS FOR CLARIFICATION:" in extracted_section:
            extracted_section = extracted_section.split("QUESTIONS FOR CLARIFICATION:")[0]
        
        # Parse key-value pairs from extracted section
        lines = extracted_section.strip().split('\n')
        for line in lines:
            if ':' in line and line.strip():
                parts = line.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip().strip('-').strip('*').strip()
                    value = parts[1].strip()
                    if key and value and value != "":
                        extracted_data[key] = value
    
    return extracted_data

def extract_missing_fields(response_text: str) -> List[str]:
    """Extract missing fields from AI response."""
    missing_fields = []
    
    if "MISSING DATA:" in response_text:
        missing_section = response_text.split("MISSING DATA:")[1]
        if "QUESTIONS FOR CLARIFICATION:" in missing_section:
            missing_section = missing_section.split("QUESTIONS FOR CLARIFICATION:")[0]

        lines = missing_section.strip().split('\n')
        for line in lines:
            line = line.strip().strip('-').strip('*').strip()
            if line and line != "":
                missing_fields.append(line)
    
    return missing_fields

def extract_clarification_questions(response_text: str) -> List[str]:
    """Extract clarification questions from AI response."""
    questions = []
    if "QUESTIONS FOR CLARIFICATION:" in response_text:
        questions_section = response_text.split("QUESTIONS FOR CLARIFICATION:")[1]
        lines = questions_section.strip().split('\n')
        for line in lines:
            line = line.strip().strip('-').strip('*').strip()
            if line and line != "" and '?' in line:
                questions.append(line)
    
    return questions

def main():
    st.title("🤖 RAG-Powered Trackbot")
    st.markdown("Analyze communication dumps to extract structured information and generate JSON output!")

    # Initialize components
    embedding_model, vector_store = initialize_rag_components()
    llm = load_llm()

    # Initialize session state for data tracking
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "extracted_data" not in st.session_state:
        st.session_state.extracted_data = {}
    if "missing_fields" not in st.session_state:
        st.session_state.missing_fields = []
    if "clarification_questions" not in st.session_state:
        st.session_state.clarification_questions = []
    if "current_question_index" not in st.session_state:
        st.session_state.current_question_index = 0
    if "asking_clarification" not in st.session_state:
        st.session_state.asking_clarification = False


    with st.sidebar:
        st.header("📋 JSON Requirements")
        
        # Show knowledge base status
        if embedding_model and vector_store:
            st.success("✅ Knowledge base loaded successfully!")
            st.info("Ready to analyze communication dumps.")
        else:
            st.error("❌ Failed to load knowledge base")
            st.error("Please ensure 'knowledge base.docx' exists in the app directory.")

        st.markdown("---")
        st.markdown("### ℹ️ How it works")
        st.markdown("""
        1. Paste your communication dump in the chat
        2. AI extracts available information
        3. AI asks for missing required data
        4. Provide answers or say "skip" for null values
        5. Generate final JSON with all data
        """)   

    # Display chat history
    for message in st.session_state.messages:
       with st.chat_message(message["role"]):
            st.markdown(message["content"])


    #Turn on clarification mode if there are missing fields
    if st.session_state.asking_clarification and st.session_state.clarification_questions:
        current_idx = st.session_state.current_question_index
        if current_idx < len(st.session_state.clarification_questions):
            question = st.session_state.clarification_questions[current_idx]
            st.info(f"Question {current_idx + 1} of {len(st.session_state.clarification_questions)}: {question}")

    # Chat input
    input_placeholder = "Paste your communication dump here..." if not st.session_state.asking_clarification else "Provide the requested information or type 'skip' to leave as null..."



    
    prompt = st.chat_input(input_placeholder)
    if prompt:

        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Handle clarification responses
        if st.session_state.asking_clarification and st.session_state.clarification_questions:
            pass
        else:
            # Process communication dump
            rag_graph = setup_rag_pipeline(vector_store, llm)

            if rag_graph:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing communication dump..."):
                        try:
                            # Run RAG pipeline
                            result = rag_graph.invoke({"question": prompt})
                            answer = result.get("answer", "Sorry, I couldn't analyze the communication dump.")
                            
                            # Update session state with extracted information (append, don't replace)
                            new_extracted_data = result.get("extracted_data", {})
                            st.session_state.extracted_data.update(new_extracted_data)
                            
                            # Add new missing fields (avoid duplicates)
                            new_missing_fields = result.get("missing_fields", [])
                            for field in new_missing_fields:
                                if field not in st.session_state.missing_fields and field not in st.session_state.extracted_data:
                                    st.session_state.missing_fields.append(field)
                            
                            # Add new clarification questions (avoid duplicates)
                            new_clarification_questions = result.get("clarification_questions", [])
                            for question in new_clarification_questions:
                                if question not in st.session_state.clarification_questions and question not in st.session_state.extracted_data:
                                    st.session_state.clarification_questions.append(question)
                            
                            # Display analysis result
                            st.markdown(answer)
                            
                            # Start clarification process if needed
                            if st.session_state.clarification_questions and not st.session_state.asking_clarification:
                                st.session_state.asking_clarification = True
                                st.session_state.current_question_index = 0
                                clarification_msg = f"I found some information but need clarification on {len(st.session_state.clarification_questions)} items. I'll ask you one by one:"
                                st.markdown(clarification_msg)
                                st.session_state.messages.append({"role": "assistant", "content": clarification_msg})
                            
                            # Add main response to chat history
                            st.session_state.messages.append({"role": "assistant", "content": answer})
                            
                        except Exception as e:
                            error_msg = f"Error analyzing communication dump: {e}"
                            st.error(error_msg)
                            st.session_state.messages.append({
                                "role": "assistant", 
                                "content": error_msg
                            })
            else:
                st.error("Failed to set up analysis pipeline.")

        