from chromadb import Embeddings
# import requests
# from bs4 import BeautifulSoup
# from langchain_core.documents import Document
# from langchain_text_splitters import RecursiveCharacterTextSplitter
from urllib.parse import urljoin

from langchain_community.document_loaders import WebBaseLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain

import os
# from langchain_google_genai import GoogleGenerativeAIEmbeddings

from langchain_core.messages import HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from dotenv import load_dotenv
# import sentence_transformers
load_dotenv()

print("started")

# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
# }

# Function to scrape links from the base URL
import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def scrape_website(base_url):
    pages = []
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract links from sidebar or other navigation elements
    links = soup.select('a')
    for link in links:
        url = link.get('href')
        if url and url.startswith('/'):
            full_url = urljoin(base_url, url)
            pages.append(full_url)
            
    return pages

def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    text = soup.get_text(separator=' ')
    return text, url

# Step 2: Extract text from all pages
base_url = 'https://docs.arbitrum.foundation/gentle-intro-dao-governance'
pages = scrape_website(base_url)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

documents = []
for page in pages:
    text, url = extract_text_from_url(page)
    doc = Document(page_content=text, metadata={"url": url})
    chunks = text_splitter.split_documents([doc])
    documents.extend(chunks)




# print("*"*100)
# print(len(documents))
# print("*"*100)

print("Documents created")

persist_directory = "./chroma_db"

# embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
embedding_function = HuggingFaceEmbeddings()

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

    vectorstore = Chroma.from_documents(documents, embedding_function, persist_directory=persist_directory)

else:
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embedding_function)

print("Vectorstore created")

llm = ChatGroq(
    model="llama3-8b-8192",
    temperature=0
)

# llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

question = input("Enter your question: ")

retriever = vectorstore.as_retriever(k=1)  

docs = retriever.invoke(question)



# print("*"*100)
# print(docs)
# print("*"*100)

from langchain import hub

SYSTEM_TEMPLATE = """
Answer the user's questions based on the below context. ALWAYS PROVIDE THE SOURCE URL IN THE RESPONSE.
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know". 

Context with sources:
{context}
"""
retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")

# question_answering_prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             SYSTEM_TEMPLATE,
#         ),
#         MessagesPlaceholder(variable_name="messages"),
#     ]
# )

document_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)


# response = retrieval_chain.invoke(
#     {
#         "context": docs,
#         "messages": [
#             HumanMessage(content=question)
#         ],
#     }
# )


response = retrieval_chain.invoke({"input": question})

print(response)
