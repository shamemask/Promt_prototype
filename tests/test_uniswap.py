import os
import unittest

from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.chroma import Chroma
from langchain_core.documents import Document

from Promt_prototype.question_model import load_pdf_to_list, chunk_split, vectorize, convert_to_chains

path_to_pdf = os.path.join("..","pdf","marketer.pdf")
class TestUniswap(unittest.TestCase):

    def test_load_pdf_to_list(self):
        # Test case 1: Test with a valid PDF file
        result = load_pdf_to_list(path_to_pdf)
        self.assertIsInstance(result, list)

        # Test case 2: Test with str in list
        result = load_pdf_to_list(path_to_pdf)
        self.assertIsInstance(result[-1], Document)

        # Test case 3: Test with count element in list more 0
        result = load_pdf_to_list(path_to_pdf)
        self.assertGreater(len(result), 0)

    def test_chunk_split(self):
        # Test case 1: Test with empty list
        docs = []
        result = chunk_split(docs)
        self.assertEqual(result, [])

        # Test case 2: Test type single document
        docs = load_pdf_to_list(path_to_pdf)
        result = chunk_split(docs)
        self.assertIsInstance(result[-1], Document)

    def test_vectorize(self):
        # Test case 1: Test with non-empty list
        docs = load_pdf_to_list(path_to_pdf)
        chunked = chunk_split(docs)
        result = vectorize(chunked)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Chroma)
        self.assertEqual(result._collection.name, "full_documents")
        self.assertIsInstance(result._embedding_function, OpenAIEmbeddings)

    def test_convert_to_chains(self):
        # Test case 1: Test with non-empty list
        docs = load_pdf_to_list(path_to_pdf)
        chunked = chunk_split(docs)
        result = vectorize(chunked)
        self.assertIsNotNone(result)
        self.assertIsInstance(result, Chroma)
        self.assertEqual(result._collection.name, "full_documents")
        self.assertIsInstance(result._embedding_function, OpenAIEmbeddings)

    def test_convert_to_chains(self):
        # Test case 1: Test with type returned by convert_to_chains
        docs = load_pdf_to_list(path_to_pdf)
        docs = chunk_split(docs)
        vector_store = vectorize(docs)
        qa = convert_to_chains(vector_store)
        self.assertIsInstance(qa,ConversationalRetrievalChain)


if __name__ == '__main__':
    unittest.main()