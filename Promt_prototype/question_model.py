import os
from typing import List

from langchain.chains.conversational_retrieval.base import BaseConversationalRetrievalChain
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from langchain.chains import ConversationalRetrievalChain

from langchain_core.documents import Document


path_to_pdf = os.path.join("pdf","marketer.pdf")

def load_pdf_to_list(path: str) -> List[Document]:
    """
    Загружает PDF-файл в список документов.

    Аргументы:
        path (str): Путь к PDF-файлу.

    Возвращает:
        List[Document]: Список документов, полученных из PDF-файла.
    """
    loader = [PyPDFLoader(path)]
    docs = []
    for l in loader:
        docs.extend(l.load())
    return docs

def chunk_split(docs:List[Document]) -> List[Document]:
    """
    Разбивает список документов на части.

    Аргументы:
        docs (List[Document]): Список документов, которые нужно разбить на части.

    Возвращает:
        List[Document]: Список документов, разделенных на части.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000)
    return text_splitter.split_documents(docs)

def vectorize(docs: List[Document]):
    """
    Векторизует список документов.

    Аргументы:
        docs (List[Document]): Список документов, которые нужно векторизовать.

    Возвращает:
        Chroma: Хранилище векторов, содержащее векторизованные документы.
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

def convert_to_chains(vector_store: Chroma) -> BaseConversationalRetrievalChain:
    """
    Преобразует векторное хранилище в ConversationalRetrievalChain.

    Аргументы:
        vector_store (Chroma): Векторное хранилище для преобразования.

    Возвращает:
        ConversationalRetrievalChain: Преобразованный ConversationalRetrievalChain.
    """
    from main import OPENAI_API_KEY
    return ConversationalRetrievalChain.from_llm(
        OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY),
        vector_store.as_retriever(),
        memory=ConversationBufferMemory(memory_key="chat_history", return_messages=True),
    )


def get_qa():
    """
    Получить вопросы и ответы из PDF-файла.

    Эта функция загружает PDF-файл и преобразует его в список строк, где каждая строка представляет собой
    страницу PDF. PDF-файл находится по пути "../pdf/marketer.pdf". Затем список строк разбивается
    на более мелкие части с помощью функции `chunk_split`. Затем вызывается функция `vectorize`,
    чтобы преобразовать фрагменты текста в числовые векторы. Наконец, вызывается функция `convert_to_chains`,
    чтобы преобразовать векторы в цепочечное представление.

    Возвращает:
        Цепочечное представление вопросов и ответов из PDF-файла.

    Выбрасывает:
        FileNotFoundError: Если PDF-файл по пути "../pdf/marketer.pdf" не существует.
        PDFParsingError: Если произошла ошибка при разборе PDF-файла.
    """
    docs = load_pdf_to_list(path_to_pdf)
    docs = chunk_split(docs)
    vector_store = vectorize(docs)
    return convert_to_chains(vector_store)