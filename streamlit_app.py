from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
import streamlit as st

# Set up the Gemini model using your secret key
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # You can use gemini-1.5-pro if needed
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    convert_system_message_to_human=True

st.title("ChatGPT-like clone")

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

        # Build message list for Gemini
    message_chain = [
        HumanMessage(msg["content"]) if msg["role"] == "user" else AIMessage(msg["content"])
        for msg in st.session_state.messages
    ]


    with st.chat_message("assistant"):
        response = llm.invoke(message_chain)
        st.markdown(response.content)
        
    st.session_state.messages.append({"role": "assistant", "content": response})
