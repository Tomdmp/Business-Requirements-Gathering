import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI

st.title("Business Requirements Chatbot") 

import getpass
import os

if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash") 


user_input = st.text_input("Enter a copy of a transcript, email or document:")

if user_input: 
  response = llm.invoke(user_input)
  st.write(response.content)
