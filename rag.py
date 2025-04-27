from langchain_openai import ChatOpenAI
import streamlit as st
from load_and_embed_html import load_and_embed_documents

# Load OpenAI API Key
#os.environ["OPENAI_API_KEY"] = st.secrets['OPENAI_API_KEY']

gwdg_api_key = st.secrets["GWDG_API_KEY"]
base_url = st.secrets["BASE_URL"]
model = "meta-llama-3.1-8b-instruct"

# Initialize model
llm = ChatOpenAI(
    model=model,
    temperature=0,
    base_url=base_url,
    api_key=gwdg_api_key
)

# Load vectorstore only once
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = load_and_embed_documents()

# Streamlit UI
st.title("StudIT RAG Chat")

prompt = st.chat_input("Frag mich etwas...")

# Create session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    st.chat_message(message["role"]).markdown(message["content"])

if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Retrieve relevant documents
    retriever = st.session_state.vectorstore.as_retriever()
    docs = retriever.get_relevant_documents(prompt)

    # Extract document content and URLs
    context = "\n".join([
        f"({doc.metadata['source']}, {doc.metadata['url']}): {doc.page_content}"
        for doc in docs
    ])

    sources = list(set(doc.metadata["url"] for doc in docs if "url" in doc.metadata))

    # Define system prompt
    system_prompt = f"""
    You are a helpful AI assistant. You have access to the following document excerpts:

    {context}

    Based on this information, answer the user's question.

    User's question: {prompt}
    """

    # Generate response
    response = llm(system_prompt)

    # Append sources if available
    source_text = "\n\n🔗 **Source(s):**\n" + "\n".join(sources) if sources else ""
    final_response = response.content + source_text

    # Store response in session state
    st.session_state.messages.append({"role": "assistant", "content": final_response})
    st.chat_message("assistant").markdown(final_response)
