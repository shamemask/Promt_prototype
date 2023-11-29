from typing import List

import openai
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from langchain.chains import ConversationalRetrievalChain
import os

from langchain_core.documents import Document


def load_pdf_to_list(path: str) -> List[Document]:
    """
    Load the PDF file into a list.
    Args:
        path (str): The path to the PDF file.
    Returns:
        list: A list containing the loaded PDF documents.
    """
    loader = [PyPDFLoader(path)]
    docs = []
    for l in loader:
        docs.extend(l.load())
    return docs

def chunk_split(docs:List[Document]) -> List[Document]:
    """
        Splits a list of documents into smaller chunks using a text splitter.
        Args:
            docs (List[Document]): The list of documents to be split.
        Returns:
            List[Document]: The list of documents after splitting.
        Raises:
            None
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    return text_splitter.split_documents(docs)

def vectorize(docs: List[Document]):
    """
    Vectorizes a list of documents using the Chroma vector store.
    Args:
        docs (List[Document]): A list of Document objects representing the documents to vectorize.
    Returns:
        Chroma: The Chroma vector store containing the vectorized documents.
    """
    from main import OPENAI_API_KEY
    vector_store = Chroma(
        collection_name="full_documents",
        embedding_function=OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
        ),
    )
    vector_store.add_documents(docs)
    return vector_store

def convert_to_chains(vector_store: Chroma) -> ConversationalRetrievalChain:
    """
    Converts a Chroma vector store into a ConversationalRetrievalChain.
    Args:
        vector_store (Chroma): The Chroma vector store containing the vectorized documents.
    Returns:
        ConversationalRetrievalChain: The ConversationalRetrievalChain containing the vectorized documents.
    """
    from main import OPENAI_API_KEY
    return ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
        vector_store.as_retriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    )


def answer_uniswap_question(question: str):
    docs = load_pdf_to_list(os.path.join("..","marketer.pdf"))
    docs = chunk_split(docs)
    vector_store = vectorize(docs)
    qa = convert_to_chains(vector_store)
    response = qa({"question": question})
    return response["answers"]


