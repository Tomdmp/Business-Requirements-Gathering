# RAG-Powered Chatbot Setup Instructions

## Overview
The `track_app.py` file is a Streamlit chatbot application that combines:
- Retrieval-Augmented Generation (RAG) pipeline using FAISS and HuggingFace embeddings
- Google Gemini AI for chat responses
- Pre-loaded knowledge base from 'knowledge base.docx'
- Interactive chat interface with clarification logic

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up API Key
Make sure your Google API key is configured in Streamlit secrets:
- Create `.streamlit/secrets.toml` file
- Add: `GOOGLE_API_KEY = "your_api_key_here"`

### 3. Ensure Knowledge Base File
Make sure `knowledge base.docx` is in the same directory as `track_app.py`

### 4. Run the Application
```bash
streamlit run track_app.py
```

## Features

### Pre-loaded Knowledge Base
- Automatically loads 'knowledge base.docx' on startup
- Documents are processed and indexed automatically
- No manual upload required

### RAG-Powered Chat
- AI searches the knowledge base for relevant content
- Responses are generated using both document context and Gemini's knowledge

### Clarification Logic
- Handles missing fields by asking clarification questions
- Iteratively refines responses based on user input
- Sends clarification answers back to the RAG pipeline for further processing

### Chat Interface
- Persistent chat history during session
- Clear chat history option
- Real-time response generation


## How It Works

1. **Automatic Document Loading**: 
   - App automatically loads 'knowledge base.docx' on startup
   - Documents are split into chunks using RecursiveCharacterTextSplitter
   - Embeddings are generated using HuggingFace sentence-transformers
   - Chunks are stored in FAISS vector store

2. **RAG Pipeline**:
   - User question is embedded and used to search vector store
   - Top 3 most similar document chunks are retrieved
   - Gemini generates response using retrieved context + user question

3. **Clarification Process**:
   - Missing fields are identified in the AI response
   - Clarification questions are generated and presented to the user
   - User answers are sent back to the RAG pipeline for iterative refinement

4. **LangGraph Integration**:
   - Uses the StateGraph pattern for sequential retrieve → generate workflow
   - Proper error handling throughout

## Key Differences from streamlit_app.py

- **Pre-loaded knowledge base**: Automatically loads 'knowledge base.docx'
- **Clarification logic**: Iteratively refines responses based on user input
- **RAG integration**: Uses vector search instead of just chat history
- **Source transparency**: Shows which documents informed the response
- **Caching**: Uses Streamlit caching for better performance

## File Requirements

- `knowledge base.docx` must be in the same directory as `track_app.py`
- The file will be automatically processed on app startup
- If the file is missing, the app will show an error message

## Troubleshooting

- If embeddings fail to load, ensure you have internet connection for HuggingFace downloads
- If FAISS errors occur, try reinstalling faiss-cpu
- If 'knowledge base.docx' not found error: ensure the file is in the correct directory
- For document processing errors, check that 'knowledge base.docx' is a valid Word document
- API key errors: verify your Google API key in secrets.toml
