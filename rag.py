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

if "user_queries" not in st.session_state:
    st.session_state.user_queries = []

# ----------------- Streamlit-UI ------------------
st.title("StudIT RAG Chat")
prompt = st.chat_input("Frag mich etwas...")

for m in st.session_state.messages:
    st.chat_message(m["role"]).markdown(m["content"])

# ============  neue User-Eingabe =============
if prompt:
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.user_queries.append(prompt)

    print(f"\Alle Nachrichten:\n{st.session_state.messages}\n\n")

    # ---------- Stand-Alone-Query bauen ----------
    previous_qs_str = (
            "Bisherige Nutzeranfragen: "
            + ", ".join(st.session_state.user_queries[:-1])
    )
    standalone_query = f"{previous_qs_str}\nAktuelle Nutzeranfrage: {prompt.strip()}".strip()
    print(f"\nStandalone query: {standalone_query}\n")

    # -----------------------------------------------------------
    # 1) Retrieval – getrennte Kontingente
    # -----------------------------------------------------------
    VEC_TARGET   = 4          # gewünschte Zahl einmaliger Vector-Parents
    BM25_TARGET  = 2          # gewünschte Zahl einmaliger BM-25-Parents
    OVERFETCH    = 4          # Sicherheitsaufschlag gegen Duplikate

    # ---------- Vector-Retrieval --------------------------------
    vec_candidates = st.session_state.parent_retriever.invoke(
        standalone_query,
        search_kwargs={"k": VEC_TARGET + OVERFETCH},   # etwas mehr holen
    )

    # doppelte Chunks entfernen
    seen_parents = set()       # globaler Duplikat-Tracker
    docs_vec     = []
    for d in vec_candidates:
        pid = d.metadata["source"]
        if pid not in seen_parents:
            docs_vec.append(d)
            seen_parents.add(pid)
        if len(docs_vec) == VEC_TARGET:
            break

    # ---------- BM-25-Retrieval ---------------------------------
    bm25_childs   = retrieveBM25(standalone_query, k=BM25_TARGET + OVERFETCH*2)
    parent_store  = st.session_state.parent_store
    docs_bm25     = []

    for child in bm25_childs:
        pid = child.metadata["source"]
        if pid in seen_parents:          # schon via Vector drin → überspringen
            continue
        parent_doc = parent_store.mget([pid])[0]
        if parent_doc:
            docs_bm25.append(parent_doc)
            seen_parents.add(pid)
        if len(docs_bm25) == BM25_TARGET:
            break

    # ------------- Finale Doc-Liste ------------------------------
    docs = docs_vec + docs_bm25            # max. 8 eindeutige Parent-Docs
    print(f"---Vector Docs:---\n {docs_vec}\n---BM25 Docs:---\n {docs_bm25}\n")
    print(f"Retrieved docs through both (union):\n{docs}\n")

    # =========================================================
    # 3) Kontext für das LLM zusammenbauen
    # =========================================================
    MAX_CHARS = 20000
    context_parts, sources = [], set()
    for d in docs:
        snippet = d.page_content[:MAX_CHARS] + ("…" if len(d.page_content) > MAX_CHARS else "")
        context_parts.append(f"({d.metadata['source']}, {d.metadata['url']}):\n{snippet}")
        sources.add(d.metadata["url"])

    context = "\n\n".join(context_parts)
    print(f"final context:\n{context}\n")

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
