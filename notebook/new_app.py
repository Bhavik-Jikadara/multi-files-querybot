import os
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.memory import ConversationBufferMemory
from src.htmlTemplates import css, bot_template, user_template
import time
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
st.write(css, unsafe_allow_html=True)


# Step 1
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Step 2
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Step 3
def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Step 4
def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME"))
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )   
    return conversation_chain

def response_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            # Display user message in chat message container
            st.chat_message("user").markdown(user_question)
            st.session_state.messages.append({"role": "user", "content": user_question})
        else:
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

# --------------------------------- UI Part -----------------------------

def sidebar():
    # user input
    user_question = st.chat_input("Ask a Question.")
    st.title("Chat with Multiple PDFs :books:")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = []

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # handle user input
    if user_question:
        response_userinput(user_question)


    # This is sidebar input where user can upload multiple times
    with st.sidebar:
        st.subheader("Your documents")

        # file uploader instance
        pdf_docs = st.file_uploader(
            label="Upload your PDFs here and click on 'Process'",
            accept_multiple_files=True,
            type=[".pdf"]
        ) 

        if st.button("Process"): 
            with st.status("In process...", expanded=True) as status:
                # Get text from PDFs
                st.write("Get the text from the PDFs files")
                raw_text = get_pdf_text(pdf_docs)
                time.sleep(3)

                # After getting text, texts converting into chunks
                st.write("Diving text into chunks")
                text_chunks = get_text_chunks(raw_text)
                time.sleep(2)

                # chunks are passing in vectorstore
                st.write("Passing chunks in the vectorstore")
                vectorstore = get_vectorstore(text_chunks)
                time.sleep(1)

                status.update(
                    label="Successfully process!",
                    state="complete"
                )
                status.info("Now you can ask a question!")

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)

        st.write(st.session_state.conversation)



if __name__ == "__main__":
    sidebar()