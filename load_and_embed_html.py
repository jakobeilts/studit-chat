import os, streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.storage import InMemoryStore
from custom_embeddings import AcademicCloudEmbeddings
from chunk_documents import chunk_documents

# --------------------------------------------------
CHUNK_SIZE, CHUNK_OVERLAP = 800, 100
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)
PERSIST_PATH = "vectorstore_index"
DOCUMENT_PATH = "documents"


def load_and_embed_documents():
    # custom_embeddings.py sets default values for model, prompt instructions etc.
    embedder = AcademicCloudEmbeddings(api_key=st.secrets["GWDG_API_KEY"], url=st.secrets["BASE_URL_EMBEDDINGS"])

    # ---------- 1) Index already existing ----------
    if os.path.exists(os.path.join(PERSIST_PATH, "index.faiss")):
        vs = FAISS.load_local(
            PERSIST_PATH, embeddings=embedder, allow_dangerous_deserialization=True
        )
        parent_store = _build_parent_store_from_txts()
        return vs, parent_store

    # ---------- 2) create new Index if non-existant ------------
    child_chunks, parent_docs = chunk_documents(DOCUMENT_PATH)

    # Parent-Docs ins Store schreiben  (key = Dateiname)
    parent_store = InMemoryStore()
    parent_store.mset([(d.metadata["source"], d) for d in parent_docs])

    vs = FAISS.from_documents(child_chunks, embedder)
    vs.save_local(PERSIST_PATH)
    return vs, parent_store


# --------------------------------------------------
def _build_parent_store_from_txts():
    """Wird aufgerufen, wenn der Vektorindex schon existiert."""
    store = InMemoryStore()
    for fn in os.listdir(DOCUMENT_PATH):
        if not fn.endswith(".txt"):
            continue
        url, *rest = open(os.path.join(DOCUMENT_PATH, fn), encoding="utf-8").readlines()
        full_text = "".join(rest).strip()
        store.mset([(fn, Document(full_text, metadata={"source": fn, "url": url.strip()}))])
    return store