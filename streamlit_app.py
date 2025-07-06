import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

st.title("Business Requirements Chatbot") 

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

response = f"Echo: {prompt}"
# Display assistant response in chat message container
with st.chat_message("assistant"):
    st.markdown(response)
# Add assistant response to chat history
st.session_state.messages.append({"role": "assistant", "content": response})

#llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") 


#user_input = st.text_input("Enter a copy of a transcript, email or document:")

#if user_input: 
#  response = llm.invoke(user_input)
#  st.write(response.content)
