from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from pydantic import BaseModel

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

import os
from dotenv import load_dotenv
import pickle

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub
from langchain_openai import ChatOpenAI, OpenAIEmbeddings



load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)


class Question(BaseModel):
    question: str


def scrape_website(base_url):
    pages = []
    response = requests.get(base_url)
    soup = BeautifulSoup(response.text, "html.parser")

    links = soup.select("a")
    for link in links:
        url = link.get("href")
        if url and url.startswith("/"):
            full_url = urljoin(base_url, url)
            pages.append(full_url)

    return pages


def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    text = soup.get_text(separator=" ")
    return text, url


def initialize_vectorstore():
    base_url = "https://docs.arbitrum.foundation/gentle-intro-dao-governance"

    pages = scrape_website(base_url)

    file_path = "documents.pkl"

    if not os.path.exists(file_path):
        pages = scrape_website(base_url)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        documents = []
        for page in pages:
            text, url = extract_text_from_url(page)
            if text:
                doc = Document(page_content=text, metadata={"url": url})
                chunks = text_splitter.split_documents([doc])
                documents.extend(chunks)

        with open(file_path, "wb") as file:
            pickle.dump(documents, file)
        print("Documents saved to documents.pkl")
    else:
        with open(file_path, "rb") as file:
            documents = pickle.load(file)
        print("Loaded documents from documents.pkl")

    persist_directory = "./chroma_db"

    embedding_function = HuggingFaceEmbeddings()
    # embedding_function = OpenAIEmbeddings()

    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)
        vectorstore = Chroma.from_documents(
            documents, embedding_function, persist_directory=persist_directory
        )
    else:
        vectorstore = Chroma(
            persist_directory=persist_directory, embedding_function=embedding_function
        )

    return vectorstore


def initialize_retrieval_chain(vectorstore):
    retriever = vectorstore.as_retriever(k=4)

    llm = ChatGroq(model="llama3-8b-8192", temperature=0.2, streaming=True)

    # llm = ChatOpenAI(temperature=0.3, streaming=True)

    retrieval_qa_chat_prompt = (
        hub.pull("langchain-ai/retrieval-qa-chat")
        + "Give a detailed answer to the question."
    )

    document_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    print("Retrieval chain created")

    return retrieval_chain


def format_answer(answer, context):
    sources = {doc.metadata["url"] for doc in context}
    formatted_answer = f"{answer}\n\n\nSources:\n" + "\n".join(sources)
    return formatted_answer



async def generate_chat_responses(retrieval_chain, question):
    try:
        full_answer = ""
        context_data = []

        async for chunk in retrieval_chain.astream({"input": question.question}):
            if "answer" in chunk:
                full_answer += chunk["answer"]

            if "context" in chunk:
                context_data.append(chunk["context"])

        if full_answer:
            full_answer = full_answer.replace("\n", "<br>")
            sources = set()
            if context_data:
                for doc in context_data[0]:
                    sources.add(doc.metadata["url"])
            formatted_answer = f"{full_answer}\n\n\nSources:\n" + "\n".join(sources)
            yield f"Answer: {formatted_answer}\n\n"

    except Exception as e:
        yield f"Error: {str(e)}\n\n"


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.post("/ask")
async def ask_question(question: Question):
    try:
        vectorstore = initialize_vectorstore()
        retrieval_chain = initialize_retrieval_chain(vectorstore)
        return StreamingResponse(
            generate_chat_responses(retrieval_chain, question),
            media_type="text/event-stream",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
