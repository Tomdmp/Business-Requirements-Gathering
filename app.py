import streamlit as st
from rag_pipeline import generate_response
from db_utils import store_requirements 

st.title("Business Requirements Chatbot") 

user_input = st.text_input("Enter a copy of a transcript, email or document:")

if user_input: 
  response_json = generate_response(user_input)
  st.json(response_json)
  result = store_requirements(response_json)
  st.success("Requirements saved to database")
