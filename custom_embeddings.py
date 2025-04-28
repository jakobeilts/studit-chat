import requests
import numpy as np
from langchain_core.embeddings import Embeddings
import streamlit as st

class AcademicCloudEmbeddings(Embeddings):
    def __init__(self, api_key: str, model: str = "e5-mistral-7b-instruct", url: str = st.secrets["BASE_URL_EMBEDDINGS"]):
        self.api_key = api_key
        self.model = model
        self.url = url

    def embed_documents(self, texts):
        return self._embed(texts)

    def embed_query(self, text):
        return self._embed([text])[0]

    def _embed(self, texts):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "input": texts,  # passing lists of text
            "model": self.model,
            "encoding_format": "float"
        }
        response = requests.post(self.url, headers=headers, json=payload)
        response.raise_for_status()
        embeddings = [np.array(d["embedding"]) for d in response.json()["data"]]
        return embeddings
