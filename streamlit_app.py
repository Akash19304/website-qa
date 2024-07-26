import streamlit as st
from langchain import hub
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import requests
import os
import pickle
import time



def scrape_website(base_url, timeout=10):
    pages = []
    try:
        response = requests.get(base_url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        links = soup.select('a')
        for link in links:
            url = link.get('href')
            if url and url.startswith('/'):
                full_url = urljoin(base_url, url)
                pages.append(full_url)
                
    except requests.exceptions.RequestException as e:
        st.error(f"Error while scraping the website: {e}")
        
    return pages



def extract_text_from_url(url, timeout=10):
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator=' ')
        return text, url
    except requests.exceptions.RequestException as e:
        st.error(f"Error while extracting text from {url}: {e}")
        return "", url

def format_answer(answer, context):
    sources = {doc.metadata['url'] for doc in context}
    formatted_answer = f"{answer}\n\nSources:\n" + "\n".join(sources)
    return formatted_answer



base_url = 'https://docs.arbitrum.foundation/gentle-intro-dao-governance'
file_path = 'documents.pkl'


if not os.path.exists(file_path):
    pages = scrape_website(base_url)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = []
    for page in pages:
        text, url = extract_text_from_url(page)
        if text:
            doc = Document(page_content=text, metadata={"url": url})
            chunks = text_splitter.split_documents([doc])
            documents.extend(chunks)

    with open(file_path, 'wb') as file:
        pickle.dump(documents, file)
    print("Documents saved to documents.pkl")
else:
    with open(file_path, 'rb') as file:
        documents = pickle.load(file)
    print("Loaded documents from documents.pkl")




persist_directory = "./chroma_db"
embedding_function = HuggingFaceEmbeddings()

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)
    vectorstore = Chroma.from_documents(documents, embedding_function, persist_directory=persist_directory)
else:
    vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_function)





retriever = vectorstore.as_retriever(k=1)  
llm = ChatGroq(model="llama3-8b-8192", temperature=0)





retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat") 
document_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)





st.title("Arbitrum Foundation Information Retrieval")

question = st.text_input("Enter your question:")
if st.button("Submit"):
    if question:
        with st.empty():  
            st.spinner("Processing...")
            time.sleep(1) 
            
            try:
                response = retrieval_chain.invoke({"input": question})
                answer = response['answer']
                context = response['context']
                formatted_response = format_answer(answer, context)
                
                st.write("**Answer:**")
                st.write(formatted_response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.write("Please enter a question.")
