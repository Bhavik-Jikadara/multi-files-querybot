# importing dependencies
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_openai import ChatOpenAI
import json
import docx
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# .docx to text
def get_docx_text(docx_files):
    text = ""
    for docx_file in docx_files:
        doc = docx.Document(docx_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text

# JSON to text
def get_json_text(json_files):
    text = ""
    for json_file in json_files:
        data = json.load(json_file)
        text += json.dumps(data, indent=4)
    return text

# Get text from files
def get_text_from_files(files):
    text = ""
    pdf_files = []
    docx_files = []
    json_files = []

    for file in files:
        if file.name.endswith('.pdf'):
            pdf_files.append(file)
        elif file.name.endswith('.docx'):
            docx_files.append(file)
        elif file.name.endswith('.json'):
            json_files.append(file)

    if pdf_files:
        text += get_pdf_text(pdf_files)
    if docx_files:
        text += get_docx_text(docx_files)
    if json_files:
        text += get_json_text(json_files)
    
    return text


# converting text to chunks
def get_text_chunks(raw_text):
    text_splitter=CharacterTextSplitter(separator="\n",
                                        chunk_size=100,
                                        chunk_overlap=10,
                                        length_function=len)   
    chunks=text_splitter.split_text(raw_text)
    return chunks

# using all-MiniLm embeddings model and faiss to get vectorstore
def get_vectorstore(text_chunks, api_key):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=api_key)
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# creating custom template to guide llm model
custom_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""

CUSTOM_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)


# generating conversation chain  
def get_conversation_chain(vectorstore, api_key):
    llm = ChatOpenAI(api_key=api_key, model="gpt-3.5-turbo")
    memory = ConversationBufferMemory(
        memory_key='chat_history', 
        return_messages=True,
        output_key='answer'
    ) # using conversation buffer memory to hold past information
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        condense_question_prompt=CUSTOM_QUESTION_PROMPT,
        memory=memory
    )
    
    return conversation_chain