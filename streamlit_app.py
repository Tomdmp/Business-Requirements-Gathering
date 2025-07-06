import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

st.title("Business Requirements Chatbot") 

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") 


user_input = st.text_input("Enter a copy of a transcript, email or document:")

if user_input: 
  response = llm.invoke(user_input)
  st.write(response.content)
