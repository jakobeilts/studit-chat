import requests
import streamlit as st


url = "https://chat-ai.academiccloud.de/v1/embeddings"
headers = {
    "Authorization": f"Bearer {st.secrets['GWDG_API_KEY']}",
    "Content-Type": "application/json"
}
data = {
    "input": "The food was delicious and the waiter...",
    "model": "e5-mistral-7b-instruct",
    "encoding_format": "float"
}

response = requests.post(url, headers=headers, json=data)
print(response.json())