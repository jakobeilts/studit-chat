import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.schema import Document

openai_api_key = st.secrets["OPENAI_API_KEY"]

# Function to load all .txt files and extract URL metadata
def load_and_embed_documents():
    documents = []

    for filename in os.listdir("documents"):
        if filename.endswith(".txt"):
            file_path = os.path.join("documents", filename)

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

                if len(lines) < 2:
                    continue  # Skip files without content

                url = lines[0].strip()  # Extract first line as URL
                content = "".join(lines[1:]).strip()  # Remaining content

                # Create a LangChain Document with metadata
                documents.append(Document(page_content=content, metadata={"source": filename, "url": url}))

    # Generate embeddings
    embeddings = OpenAIEmbeddings(api_key=openai_api_key, model="text-embedding-3-small")
    vectorstore = FAISS.from_documents(documents, embeddings)

    return vectorstore