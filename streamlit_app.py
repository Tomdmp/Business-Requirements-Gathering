from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st


# Set up Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    convert_system_message_to_human=True
)

st.title("Gemini-like clone")

system_prompt = SystemMessage(
    content=(
        "You are a helpful AI assistant that helps consultants extract and structure project requirements from client conversations. "
        "Keep answers concise, clear, and structured. Assume the client is non-technical unless stated otherwise. At the start of every response you must say: This is what I think Tom:"
    )
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("What is up?"):
    # Save user input
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    # Convert chat history to LangChain format
    message_chain = [system_prompt] + [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in st.session_state.messages
    ]

    # Get response from Gemini
    response = llm.invoke(message_chain)

    with st.chat_message("assistant"):
        st.markdown(response.content)

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": response.content})
