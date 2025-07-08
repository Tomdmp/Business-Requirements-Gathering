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
#json_instruction = load_txt("json_template.txt")
#summary_prompt = SystemMessage(content=load_txt("summary_prompt.txt"))

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

    def generate(state: State):
        """Analyze communication dump and extract structured information."""
        if not llm or not state.get("question"):
            return {"answer": "Sorry, I couldn't process your communication dump."}
        
        try:
            # Prepare context from retrieved documents (knowledge base requirements)
            docs_content = "\n\n".join(doc.page_content for doc in state.get("context", []))
            
            # Create specialized prompt for communication analysis
            if docs_content:
                prompt_text = f"""{system_prompt.content}

Knowledge Base Requirements (use this to understand what data is needed):
{docs_content}

Communication Dump to Analyze:
{state["question"]}

Please analyze the communication dump and:
1. Extract all available information that matches the knowledge base requirements
2. Identify any missing required information
3. If information is missing, ask specific questions to obtain it
4. Format your response clearly with these sections:

EXTRACTED DATA:
[List all information found in key: value format]

MISSING DATA:
[List specific fields/information still needed]

QUESTIONS FOR CLARIFICATION:
[Specific questions to ask the user to get missing information]

Analysis:"""
            else:
                prompt_text = f"""{system_prompt.content}

Communication Dump to Analyze:
{state["question"]}

Please analyze the communication dump and extract key structured information:
- Key participants and their roles
- Important dates and times
- Action items or decisions
- Contact information
- Project details
- Requirements and specifications
- Any other structured data present

Format your response with:
EXTRACTED DATA:
[key: value pairs of found information]

MISSING DATA:
[any standard fields that appear to be missing]

QUESTIONS FOR CLARIFICATION:
[questions to get more complete information]

Analysis:"""

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
        graph_builder = StateGraph(State).add_sequence([retrieve, generate])
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
        elif "NEXT STEPS:" in missing_section:
            missing_section = missing_section.split("NEXT STEPS:")[0]
        
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
        if "JSON OUTPUT:" in questions_section:
            questions_section = questions_section.split("JSON OUTPUT:")[0]
        
        lines = questions_section.strip().split('\n')
        for line in lines:
            line = line.strip().strip('-').strip('*').strip()
            if line and line != "" and '?' in line:
                questions.append(line)
    
    return questions

def generate_final_json(extracted_data: Dict[str, Any], missing_fields: List[str]) -> str:
    """Generate final JSON output with extracted data and null values for missing fields."""
    # Create a structured JSON based on the system prompt template
    json_output = {
        "Clients": {},
        "Project": {},
        "Requirements": [],
        "Constraints": [],
        "ProjectTechnology": []
    }
    
    # Map extracted data to appropriate sections
    for key, value in extracted_data.items():
        key_lower = key.lower()
        
        # Skip if this is a duplicate field (question format vs clean format)
        if key.startswith("1.") or "**" in key or key.endswith("?"):
            continue
        
        # Client information
        if any(term in key_lower for term in ['client', 'company', 'contact', 'email', 'phone', 'location', 'industry']):
            # Clean up the key for the JSON
            clean_key = key.replace("Clients: ", "").replace("Client: ", "")
            json_output["Clients"][clean_key] = value
        # Project information
        elif any(term in key_lower for term in ['project', 'start date', 'end date', 'budget', 'users', 'status', 'delivery']):
            clean_key = key.replace("Project: ", "")
            json_output["Project"][clean_key] = value
        # Technology information
        elif any(term in key_lower for term in ['technology', 'tech', 'system', 'platform', 'tool', 'software']):
            json_output["ProjectTechnology"].append({
                "TechName": key.replace("Technology: ", "").replace("Tech: ", ""),
                "Details": value,
                "Status": "Mentioned"
            })
        # Requirements
        elif any(term in key_lower for term in ['requirement', 'feature', 'functionality', 'spec']):
            json_output["Requirements"].append({
                "Description": f"{key}: {value}",
                "Status": "Identified",
                "Source": "Communication Analysis"
            })
        # Constraints
        elif any(term in key_lower for term in ['constraint', 'limitation', 'deadline', 'restriction']):
            json_output["Constraints"].append({
                "Description": f"{key}: {value}",
                "Source": "Communication Analysis"
            })
        # General project data
        else:
            json_output["Project"][key] = value
    
    # Add null values for missing fields
    for field in missing_fields:
        field_lower = field.lower()
        clean_field = extract_field_name_from_question(field)
        
        if any(term in field_lower for term in ['client', 'company', 'contact', 'email', 'phone', 'location', 'industry']):
            clean_key = clean_field.replace("Clients: ", "").replace("Client: ", "")
            if clean_key not in json_output["Clients"]:
                json_output["Clients"][clean_key] = None
        elif any(term in field_lower for term in ['project', 'start date', 'end date', 'budget', 'users', 'status', 'delivery']):
            clean_key = clean_field.replace("Project: ", "")
            if clean_key not in json_output["Project"]:
                json_output["Project"][clean_key] = None
        else:
            if clean_field not in json_output["Project"]:
                json_output["Project"][clean_field] = None
    
    # Clean up empty sections
    if not json_output["Clients"]:
        del json_output["Clients"]
    if not json_output["Project"]:
        del json_output["Project"]
    if not json_output["Requirements"]:
        del json_output["Requirements"]
    if not json_output["Constraints"]:
        del json_output["Constraints"]
    if not json_output["ProjectTechnology"]:
        del json_output["ProjectTechnology"]
    
    return json.dumps(json_output, indent=2)

def extract_field_name_from_question(question: str) -> str:
    """Extract a clean field name from a clarification question."""
    # Remove common question prefixes and suffixes
    question = question.strip()
    
    # Look for patterns like "What is the..." or "Please provide the..."
    patterns = [
        r"what is the (.+?)\?",
        r"please provide the (.+?)\?",
        r"could you provide the (.+?)\?",
        r"what (.+?)\?",
        r"clients?:\s*(.+?)\?",
        r"project:\s*(.+?)\?",
        r"(.+?)\?",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, question.lower())
        if match:
            field_name = match.group(1).strip()
            # Clean up the field name
            field_name = field_name.replace("for ", "").replace("the ", "").replace("is ", "")
            # Convert to title case
            field_name = " ".join(word.capitalize() for word in field_name.split())
            return field_name
    
    # If no pattern matches, try to extract from the question structure
    # Look for "Clients: ContactNumber" pattern
    if ":" in question:
        parts = question.split(":")
        if len(parts) >= 2:
            section = parts[0].strip()
            field = parts[1].strip().replace("**", "").replace("?", "").strip()
            return f"{section}: {field}"
    
    # Fallback: use the question itself but clean it up
    clean_question = question.replace("?", "").replace("**", "").strip()
    return clean_question

def clean_extracted_data(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean extracted data to remove duplicates and normalize field names."""
    cleaned_data = {}
    
    # Group similar fields together
    field_groups = {}
    
    for key, value in extracted_data.items():
        # Skip overly long keys that look like questions
        if len(key) > 100 or key.startswith("1.") or "**" in key:
            continue
        
        # Normalize the key
        normalized_key = key.lower().replace("clients:", "").replace("client:", "").replace("project:", "").strip()
        
        # Look for existing similar keys
        found_group = False
        for group_key in field_groups:
            if normalized_key in group_key or group_key in normalized_key:
                # Update with the cleaner key name
                if len(key) < len(group_key):
                    field_groups[key] = field_groups.pop(group_key)
                    field_groups[key] = value
                else:
                    field_groups[group_key] = value
                found_group = True
                break
        
        if not found_group:
            field_groups[key] = value
    
    return field_groups

def main():
    st.title("🤖 Communication Analysis & JSON Generator")
    st.markdown("Analyze communication dumps to extract structured information and generate JSON output!")
    
    # Initialize components
    embedding_model, vector_store = initialize_rag_components()
    llm = load_llm()
    
    if not embedding_model or not vector_store or not llm:
        st.error("Failed to initialize components. Please check your configuration and ensure 'knowledge base.docx' exists.")
        return
    
    # Initialize session state for data tracking
    if "messages" not in st.session_state:
        st.session_state.messages = []
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
    
    # Sidebar for information and settings
    with st.sidebar:
        st.header("📋 JSON Requirements")
        
        # Show knowledge base status
        if embedding_model and vector_store:
            st.success("✅ Knowledge base loaded successfully!")
            st.info("Ready to analyze communication dumps.")
        else:
            st.error("❌ Failed to load knowledge base")
            st.error("Please ensure 'knowledge base.docx' exists in the app directory.")
        
        # Show current extraction status
        if st.session_state.extracted_data or st.session_state.missing_fields:
            st.markdown("### 📊 Extraction Status")
            total_possible = len(st.session_state.extracted_data) + len(st.session_state.missing_fields)
            if total_possible > 0:
                completion = (len(st.session_state.extracted_data) / total_possible) * 100
                st.progress(completion / 100)
                st.write(f"Completion: {completion:.1f}%")
                st.write(f"Extracted: {len(st.session_state.extracted_data)} fields")
                st.write(f"Missing: {len(st.session_state.missing_fields)} fields")
                
                # Show pending clarification questions
                if st.session_state.clarification_questions:
                    st.write(f"Pending questions: {len(st.session_state.clarification_questions)}")
                    if st.session_state.asking_clarification:
                        st.info(f"Currently asking question {st.session_state.current_question_index + 1} of {len(st.session_state.clarification_questions)}")
        
        # Show extracted data preview
        if st.session_state.extracted_data:
            with st.expander("📋 Current Extracted Data"):
                for key, value in st.session_state.extracted_data.items():
                    st.write(f"**{key}:** {value}")
        
        # Show missing fields
        if st.session_state.missing_fields:
            with st.expander("❓ Missing Fields"):
                for field in st.session_state.missing_fields:
                    st.write(f"- {field}")
        
        # JSON Generation Button
        if st.button("📄 Generate Final JSON"):
            if st.session_state.extracted_data or st.session_state.missing_fields:
                # Clean the extracted data first
                cleaned_data = clean_extracted_data(st.session_state.extracted_data)
                final_json = generate_final_json(
                    cleaned_data, 
                    st.session_state.missing_fields
                )
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"**Final JSON Output:**\n\n```json\n{final_json}\n```"
                })
                st.rerun()
            else:
                st.warning("No data extracted yet. Please analyze a communication dump first.")
        
        # Clear data button
        if st.button("🗑️ Clear All Data & Memory"):
            st.session_state.extracted_data = {}
            st.session_state.missing_fields = []
            st.session_state.clarification_questions = []
            st.session_state.current_question_index = 0
            st.session_state.asking_clarification = False
            st.session_state.messages = []
            st.success("All data and memory cleared!")
            st.rerun()
        
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
    
    # Handle clarification questions
    if st.session_state.asking_clarification and st.session_state.clarification_questions:
        current_idx = st.session_state.current_question_index
        if current_idx < len(st.session_state.clarification_questions):
            question = st.session_state.clarification_questions[current_idx]
            st.info(f"Question {current_idx + 1} of {len(st.session_state.clarification_questions)}: {question}")
    
    # Chat input
    input_placeholder = "Paste your communication dump here..." if not st.session_state.asking_clarification else "Provide the requested information or type 'skip' to leave as null..."
    
    if prompt := st.chat_input(input_placeholder):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Handle clarification responses
        if st.session_state.asking_clarification and st.session_state.clarification_questions:
            current_idx = st.session_state.current_question_index
            if current_idx < len(st.session_state.clarification_questions):
                question = st.session_state.clarification_questions[current_idx]
                
                # Process the answer
                with st.chat_message("assistant"):
                    if prompt.lower().strip() in ['skip', 'null', 'n/a', 'none']:
                        # Extract clean field name from question
                        clean_field_name = extract_field_name_from_question(question)
                        st.session_state.extracted_data[clean_field_name] = None
                        response = f"Noted. '{clean_field_name}' will be set to null in the final JSON."
                    else:
                        # Extract clean field name from question
                        clean_field_name = extract_field_name_from_question(question)
                        st.session_state.extracted_data[clean_field_name] = prompt
                        response = f"Thank you! I've recorded: {clean_field_name} = {prompt}"
                    
                    # Remove the question from missing fields
                    if question in st.session_state.missing_fields:
                        st.session_state.missing_fields.remove(question)
                    
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Move to next question
                st.session_state.current_question_index += 1
                
                # Check if all questions are answered
                if st.session_state.current_question_index >= len(st.session_state.clarification_questions):
                    st.session_state.asking_clarification = False
                    st.session_state.current_question_index = 0
                    st.session_state.clarification_questions = []  # Clear answered questions
                    completion_msg = "All clarification questions completed! You can now generate the final JSON using the button in the sidebar, or provide additional communication dumps to extract more information."
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": completion_msg
                    })
                    st.markdown(completion_msg)
                
                st.rerun()
        
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
                            
                            # Show extracted data summary
                            if st.session_state.extracted_data:
                                with st.expander("📊 Extracted Data Summary"):
                                    for key, value in st.session_state.extracted_data.items():
                                        st.write(f"**{key}:** {value}")
                            
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

if __name__ == "__main__":
    main()
