import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from src.utils import get_text_from_files, get_text_chunks, get_vectorstore, get_conversation_chain
import time

load_dotenv()

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    return response['answer']

def clear_chat():
    st.session_state.chat_history = [
        AIMessage(content="Hello, I'm a MFBot. How can I help you?"),
    ]

def ui():
    st.set_page_config(
        page_title="Multiple Docs QueryBot",
        page_icon=":books:",
        layout="wide"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I'm a MFBot. How can I help you?"),
        ]

    st.header("Multi-Files QueryBot :books:")
    # User input
    user_question = st.chat_input("Ask a question about your documents:")
    # Clear chat button
    if st.button("Clear Chat"):
        clear_chat()

    # Chat container
    chat_container = st.container()

    # Display chat history
    with chat_container:
        for message in st.session_state.chat_history:
            if isinstance(message, AIMessage):
                with st.chat_message("AI"):
                    st.write(message.content)
            elif isinstance(message, HumanMessage):
                with st.chat_message("Human"):
                    st.write(message.content)

    # Handle user input
    if user_question:
        with st.chat_message("Human"):
            st.write(user_question)
        
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        
        with st.chat_message("AI"):
            message_placeholder = st.empty()
            full_response = handle_userinput(user_question)
            
            # Simulate typing effect
            for i in range(len(full_response)):
                message_placeholder.markdown(full_response[:i+1] + "▌")
                time.sleep(0.01)
            
            message_placeholder.markdown(full_response)
        
        st.session_state.chat_history.append(AIMessage(content=full_response))

    with st.sidebar:
        st.title("About Project")
        st.markdown("The **<u>Multi-Files QueryBot</u>** is a Python-based tool that allows users to interact with multiple document types, including `PDFs`, `.docx`, and `.json` files, through natural language queries.**\n* Users can ask questions based on the content of these documents, and the app provides accurate, context-aware responses.\n* It's designed to help users efficiently navigate and extract insights from large sets of documents.", unsafe_allow_html=True)
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Documents")

        api_key = st.text_input(label="Enter OpenAI api key", type="password", placeholder="OPENAI_API_KEY")
        docs = st.file_uploader(
            "Upload your files here (`PDFs`, `DOCX`, `JSON`) and click on `Process`",
            type=["pdf", "docx", "json"],
            accept_multiple_files=True
        )

        if st.button("Process"):
            with st.status("In process...", expanded=True, state="running") as status:
                st.write("Get the text from files")
                raw_text = get_text_from_files(docs)
                time.sleep(5)

                st.write(raw_text)

                st.write("Dividing text into chunks")
                text_chunks = get_text_chunks(raw_text)
                time.sleep(3)

                st.write("Passing chunks in vectorstore")
                vectorstore = get_vectorstore(text_chunks, api_key)
                time.sleep(1)
                status.update(
                    label="Successfully process! Now you can ask a question", state="complete", expanded=False)

                st.session_state.conversation = get_conversation_chain(vectorstore, api_key)

if __name__ == '__main__':
    ui()