import streamlit as st

from openai import OpenAI

from custom_embeddings import AcademicCloudEmbeddings
import numpy as np
#EMBEDS EXACTLY LIKE API CALL!!


client = OpenAI(api_key=st.secrets["GWDG_API_KEY"], base_url="https://chat-ai.academiccloud.de/v1")

model = "e5-mistral-7b-instruct"

# Sample prompts.
prompts = [
    "a",
]
task = "Instruct: Given a web search query, retrieve relevant passages that answer the query"

def as_query(q):          # build the full prompt string
    return f"{task}\nQuery: {q}"

query_vectors = client.embeddings.create(
    model="e5-mistral-7b-instruct",
    input=[as_query("how much protein should a female eat")],
    encoding_format="float"
)
