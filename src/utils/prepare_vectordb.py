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
                 chunk_overlap: int)
