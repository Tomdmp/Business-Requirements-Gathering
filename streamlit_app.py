from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import streamlit as st
import io 

def load_txt(file_path):
    with open(file_path, "r") as file:
        return file.read()

system_prompt = SystemMessage(content=load_txt("system_prompt.txt"))
json_instruction = load_txt("json_template.txt")
summary_prompt = SystemMessage(content=load_txt("summary_prompt.txt"))

# Set up Gemini model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    convert_system_message_to_human=True
)

st.title("Business Requirements Gathering")

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

# Generate JSON output from chat when a button clicked
if st.button("Generate JSON Summary"):
    with st.spinner("Generating summary..."):
        # Append user instruction to generate JSON
        final_chain = [system_prompt] + [
            HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
            for m in st.session_state.messages
        ] + [HumanMessage(content=f"Using this format:\n{json_instruction}\nPlease summarise the conversation as JSON.")]

        json_response = llm.invoke(final_chain)

        st.subheader("🧾 JSON Output")
        st.code(json_response.content, language="json")

         # ✅ Convert to file-like object for download
        json_bytes = io.BytesIO(json_response.content.encode("utf-8"))

        # ✅ Add download button
        st.download_button(
            label="📥 Download JSON",
            data=json_bytes,
            file_name="requirements_summary.json",
            mime="application/json"
        )
        
#Summary of requirements Button
if st.button("Create Summary of Requirements"):
    summary_chain = [system_prompt] + [summary_prompt] + [
        HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
        for m in st.session_state.messages
    ]

    response = llm.invoke(summary_chain)

    # Display the summary
    st.subheader("📝 Summary of Requirements")
    st.markdown(response.content)
