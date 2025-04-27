import streamlit as st
from create_txt_file import urlToTxt
import os

# Page Title
st.title("Add a New Document")

# Input Fields
url = st.text_input("Enter the URL of the document:")
keyword = st.text_input("Enter a keyword for saving the file (e.g., 'cloudstorage'):")

# Submit Button
if st.button("Save Document"):
    if url and keyword:
        try:
            urlToTxt(url, keyword)
            st.success(f"Document saved as **'{keyword}.txt'** in the **'documents'** folder.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter both a **URL** and a **keyword**!")