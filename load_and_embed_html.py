import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from custom_embeddings import AcademicCloudEmbeddings

def load_and_embed_documents():
    # Path to save/load the FAISS index
    persist_path = "vectorstore_index"

    if os.path.exists(persist_path):
        # If already exists, load from disk
        vectorstore = FAISS.load_local(persist_path, AcademicCloudEmbeddings(api_key=st.secrets["GWDG_API_KEY"]), allow_dangerous_deserialization=True)
        return vectorstore

    documents = []

    for filename in os.listdir("documents"):
        if filename.endswith(".txt"):
            file_path = os.path.join("documents", filename)

            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

                if len(lines) < 2:
                    continue

                url = lines[0].strip()
                content = "".join(lines[1:]).strip()

                documents.append(Document(page_content=content, metadata={"source": filename, "url": url}))

    # Use AcademicCloud embeddings
    embeddings = AcademicCloudEmbeddings(api_key=st.secrets["GWDG_API_KEY"])

    # Create FAISS vectorstore
    vectorstore = FAISS.from_documents(documents, embeddings)

    # Save to disk
    vectorstore.save_local(persist_path)

    return vectorstore
