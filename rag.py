from langchain_openai import ChatOpenAI
import streamlit as st
from load_and_embed_html import load_and_embed_documents
import time
from langchain_core.messages import SystemMessage

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
    Du bist ein hilfreicher KI-Assistent für Studierende und Mitarbeitende an der Georg-August-Universität Göttingen.
    Deine Aufgabe ist es, Fragen zu Universitätsdiensten basierend auf den bereitgestellten Dokumentenauszügen zu beantworten.
    Wenn eine Information nicht in den Auszügen enthalten ist, sage ehrlich, dass du dazu keine verlässliche Antwort geben kannst.
    
    Beantworte Fragen möglichst in der Sprache der Anfrage (Deutsch oder Englisch).
    Formuliere deine Antworten klar, freundlich und präzise. Fasse dich kurz, es sei denn, die Frage verlangt nach mehr Details.
    
    Die folgenden Informationen verwendest du um die Fragen zu beantworten. Diese Informationen werden dir nicht direkt vom Nutzer zu Verfügung gestellt. Verwende also nicht Sätze wie "Auf Grundlage der bereitgestellten Informationen". 
    Es soll sich anfühlen, als wären diese Informationen dein natürliches Wissen.
    ANFANG KONTEXT:
    {context}
    ENDE KONTEXT
    Außerdem bekommst du zur weiteren Einschätzung des Gesprächsinhalts im folgenden den bisherigen Gesprächsverlauf. Falls dieser nicht leer ist verwende ihn um den Kontext besser einzuschätzen:
    ANFANG BISHERIGER GESPRÄCHSVERLAUF:
    {st.session_state.messages}
    ENDE BISHERIGER GESPRÄCHSVERLAUF
    Nutze nur diese Informationen zur Beantwortung der folgenden Frage:
    
    Frage: {prompt}
    """

    print(system_prompt)

    # Generate response
    response = llm([SystemMessage(content=system_prompt)])

    # Append sources if available
    source_text = "\n\n🔗 **Source(s):**\n" + "\n".join(sources) if sources else ""
    final_response = response.content + source_text

    assistant_message = st.chat_message("assistant")
    placeholder = assistant_message.empty()
    typed_text = ""

    for char in final_response:
        typed_text += char
        placeholder.markdown(typed_text)
        time.sleep(0.003)  # Typing speed

    # Store response in session state
    st.session_state.messages.append({"role": "assistant", "content": final_response})

    print("Nachrichten:")
    print(st.session_state.messages)
    print("USED CONTEXT:")
    print(context)
    print("System Message:")
    print(system_prompt)