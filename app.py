import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain##
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import faiss
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_ollama import OllamaEmbeddings


from dotenv import load_dotenv
load_dotenv()

## load the groq api
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key = groq_api_key,model="Llama3-8b-8192")

# chatprompt template
prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only
please provide the most accurate response based on the question
<context>
{context}
<context>
Question:{input}
"""
)

# st.session_state to keep the variables after the rerun also
def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings()
        st.session_state.loader = PyPDFDirectoryLoader("research_papers") ## data ingestion step from folder
        st.session_state.docs = st.session_state.loader.load()# document loading
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors = faiss.from_documents(st.session_state.final_documents,st.session_state.embeddings)


# prompt
user_prompt = st.text_input("Enter your query from the research paper uploaded")

if st.button("Document Embedding"):
    create_vector_embeddings()
    st.write("Vector Database is ready")

# create a chain for passing a list of documents to a model(create_stuff_documents_chain)
import time

if user_prompt:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retriever = st.session_state.vector.as_retriver()#interface to pass the query the retriver
    retrival_chain = create_retrieval_chain(retriever,document_chain)

    start= time.process_time()
    response = retrival_chain.invoke({"input":user_prompt})
    print(f"Response time :{time.process_time()-start}")

    st.write(response["answer"])

    ## with streamlit expander st.expander() is a UI container widget that lets you hide or show content inside a collapsible section
    with st.expander("Document similarity search"):
        for i,doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------------")

    












