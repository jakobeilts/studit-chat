from custom_embeddings import AcademicCloudEmbeddings
import streamlit as st

embedder = AcademicCloudEmbeddings(api_key=st.secrets["GWDG_API_KEY"])

print(embedder.embed_documents("Test, ich bin Jakob"))

print(embedder.embed_query("Wie heisst du?"))