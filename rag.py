import time, streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain.retrievers import ParentDocumentRetriever

from load_and_embed_html import load_and_embed_documents, splitter
from bm25 import retrieveBM25

# ----------------- Grunddaten --------------------
llm = ChatOpenAI(
    model="meta-llama-3.1-8b-instruct",
    temperature=0,
    base_url=st.secrets["BASE_URL"],
    api_key=st.secrets["GWDG_API_KEY"],
)

# -------------- Ressourcen einmal laden ----------
if "vectorstore" not in st.session_state:
    vs, parent_store = load_and_embed_documents()
    st.session_state.vectorstore     = vs
    st.session_state.parent_store    = parent_store

if "parent_retriever" not in st.session_state:
    st.session_state.parent_retriever = ParentDocumentRetriever(
        vectorstore        = st.session_state.vectorstore,
        docstore           = st.session_state.parent_store,
        child_splitter     = splitter,
        id_key             = "source",
    )

# ------------- zusätzlicher Speicher -------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_user_question" not in st.session_state:
    st.session_state.last_user_question = ""

# ----------------- Streamlit-UI ------------------
st.title("StudIT RAG Chat")
prompt = st.chat_input("Frag mich etwas...")

for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

# ============  neue User-Eingabe =============
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # ---------- Stand-Alone-Query bauen ----------
    previous_q       = st.session_state.last_user_question
    standalone_query = f"{previous_q.strip()} {prompt.strip()}".strip()

    # =========================================================
    # 1) Vektor-Retrieval  +  BM25-Retrieval
    # =========================================================
    docs_vec = st.session_state.parent_retriever.invoke(
        standalone_query, search_kwargs={"k": 6}
    )

    docs_bm25_child = retrieveBM25(standalone_query, k=12)
    ### NEU >>>  Child-→Parent-Mapping  ------------------------
    parent_store = st.session_state.parent_store
    docs_bm25 = []
    for child in docs_bm25_child:
        parent_id   = child.metadata["source"]
        parent_doc  = parent_store.mget([parent_id])[0]
        if parent_doc:
            docs_bm25.append(parent_doc)

    # =========================================================
    # 2) Union  +  Deduplikation  +  Limit
    # =========================================================
    n_final = 6
    docs_union, seen = [], set()

    # Reihenfolge bestimmt Priorität: erst Vektor-Treffer, dann BM25-Treffer
    for d in docs_vec + docs_bm25:
        key = d.metadata["source"]          # Parent-ID als Duplikat-Schlüssel
        if key not in seen:
            docs_union.append(d)
            seen.add(key)
        if len(docs_union) >= n_final:
            break

    docs = docs_union                     # ab hier heißt es wieder ›docs‹

    # =========================================================
    # 3) Kontext für das LLM zusammenbauen
    # =========================================================
    MAX_CHARS = 8000
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

    # =========================================================
    # 4) Antwort erzeugen
    # =========================================================
    resp   = llm([SystemMessage(content=system_prompt)])
    answer = resp.content + (
        "\n\n🔗 **Source(s):**\n" + "\n".join(sorted(sources)) if sources else ""
    )

    # Tippen simulieren
    placeholder, buf = st.chat_message("assistant").empty(), ""
    for ch in answer:
        buf += ch
        placeholder.markdown(buf)
        time.sleep(0.003)

    st.session_state.messages.append({"role": "assistant", "content": answer})

    # ---------- Main-Topic updaten ----------
    if len(prompt.split()) > 3:
        st.session_state.last_user_question = prompt
