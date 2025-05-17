import time, streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain.retrievers import ParentDocumentRetriever

from load_and_embed_html import load_and_embed_documents, splitter   # splitter reuse

# ----------------- Grund­daten --------------------
llm = ChatOpenAI(
    model="meta-llama-3.1-8b-instruct",
    temperature=0,
    base_url=st.secrets["BASE_URL"],
    api_key=st.secrets["GWDG_API_KEY"],
)

# -------------- Ressourcen einmal laden ----------
if "vectorstore" not in st.session_state:
    vs, parent_store = load_and_embed_documents()
    st.session_state.vectorstore = vs
    st.session_state.parent_store = parent_store

if "parent_retriever" not in st.session_state:
    st.session_state.parent_retriever = ParentDocumentRetriever(
        vectorstore=st.session_state.vectorstore,
        docstore=st.session_state.parent_store,   # <-- jetzt vorhanden!
        child_splitter=splitter,                 # Pflicht-Arg.
        id_key="source",
        child_to_parent_key="source",
    )

# ----------------- Streamlit-UI -------------------
st.title("StudIT RAG Chat")
prompt = st.chat_input("Frag mich etwas...")

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

# ============  Auf neue User-Eingabe =============
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 1) Parent-Retrieval  (k Dateien)
    docs = st.session_state.parent_retriever.invoke(
        prompt, search_kwargs={"k": 3}
    )

    MAX_CHARS = 4_000
    context_parts, sources = [], set()
    for d in docs:
        snippet = d.page_content[:MAX_CHARS] + ("…" if len(d.page_content) > MAX_CHARS else "")
        context_parts.append(f"({d.metadata['source']}, {d.metadata['url']}):\n{snippet}")
        sources.add(d.metadata["url"])

    context = "\n\n".join(context_parts)

    system_prompt = f"""
    Du bist ein hilfreicher KI-Assistent für Studierende und Mitarbeitende an der Georg-August-Universität Göttingen.
    Antworten **ausschließlich** anhand des folgenden Kontexts.
    ANFANG KONTEXT
    {context}
    ENDE KONTEXT
    
    ANFANG BISHERIGER VERLAUF
    {st.session_state.messages}
    ENDE BISHERIGER VERLAUF
    
    Frage: {prompt}
    """.strip()

    # 2) Antwort erzeugen
    resp = llm([SystemMessage(content=system_prompt)])
    answer = resp.content + (
        "\n\n🔗 **Source(s):**\n" + "\n".join(sorted(sources)) if sources else ""
    )

    # 3) Tippen simulieren
    placeholder = st.chat_message("assistant").empty()
    buf = ""
    for ch in answer:
        buf += ch
        placeholder.markdown(buf)
        time.sleep(0.003)

    st.session_state.messages.append({"role": "assistant", "content": answer})
