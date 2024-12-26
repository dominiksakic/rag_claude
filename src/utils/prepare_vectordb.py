import os
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_google_genai import GoogleGenerativeAIEmbeddings

class PrepareVectorDB:
    def __init__(self,
                 data_dir: str,
                 persist_dir: str,
                 embedding_model_engine: str,
                 chunk_size: int,
                 chunk_overlap: int,
    ) -> None:
        self.embedding_model_engine = embedding_model_engine
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n"," ",""]
        )
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.embedding = embedding_model_engine

    def __load_all_documents(self) -> List:
        doc_counter = 0
        if isinstance(self.data_dir, list):
            print("Loading the uploaded documents...")
            docs = []
            for doc_dir in self.data_dir:
                docs.extend(PyPDFLoader(doc_dir).load())
                doc_counter += 1
            print("Number of loaded documents:", doc_counter)
            print("Number of pages:", len(docs), "\n\n")
        else:
            print("Loading documents manually...")
            document_list = os.listdir(self.data_dir)
            docs = []
            for doc_name in document_list:
                docs.extend(PyPDFLoader(os.path.join(
                    self.data_dir, doc_name)).load())
                doc_counter += 1
            print("Number of loaded documents:", doc_counter)
            print("Number of pages:", len(docs), "\n\n")

        return docs

    def __chunk_documents(self, docs: List) -> List:
        print("Chunking documents...")
        chunked_documents = self.text_splitter.split_documents(docs)
        print("Number of chunks:", len(chunked_documents), "\n\n")
        return chunked_documents

    def prepare_and_save_vectordb(self):
        docs = self.__load_all_documents()
        chunked_documents = self.__chunk_documents(docs)
        print("Preparing vectordb...")
        vectordb = Chroma.from_documents(
            documents=chunked_documents,
            embedding=self.embedding,
            persist_directory=self.persist_dir
        )
        print("VectorDB is created and saved.")
        print("Number of vectors in vectordb:",
              vectordb._collection.count(), "\n\n")
        return vectordb
