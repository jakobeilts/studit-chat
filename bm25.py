# With langchain:
from langchain_community.retrievers import BM25Retriever
from pathlib import Path
from langchain.schema import Document
import nltk
from nltk.tokenize import word_tokenize
from chunk_documents import chunk_documents

nltk.download("punkt_tab")
child_chunks, parent_docs = chunk_documents("seiten_export")

# Retrieve documents with BM25 search
def retrieveBM25(query,k=12):
    retriever = BM25Retriever.from_documents(child_chunks, k=k, preprocess_func=word_tokenize)
    result = retriever.invoke(query)
    return result