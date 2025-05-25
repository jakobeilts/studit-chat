# With langchain:
from langchain_community.retrievers import BM25Retriever
from load_and_embed_html import load_and_embed_documents
import nltk
from nltk.tokenize import word_tokenize

nltk.download("punkt_tab")
vs, _ = load_and_embed_documents()           # builds or loads the index
child_chunks = list(vs.docstore._dict.values())

# Retrieve documents with BM25 search
def retrieveBM25(query,k=12):
    retriever = BM25Retriever.from_documents(child_chunks, k=k, preprocess_func=word_tokenize)
    result = retriever.invoke(query)
    return result