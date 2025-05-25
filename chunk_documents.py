import os
from pathlib import Path
from typing import List, Tuple, Optional
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI  # pip install langchain-openai

# ---------------------------------------------------------------------------
# Configuration
#TODO HIER VERWENDE ICH OPENAI, WEIL GWDG LIMIT NICHT REICHT
CONTEXT_MODEL = "gpt-4.1-nano"
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
MAX_CONTEXT_TOKENS = 120  # how long the generated context can be

# German prompt (keeps everything in the same language as your corpus)
_CONTEXT_PROMPT = """
<document>
{document}
</document>
Hier ist der Chunk, den wir innerhalb des gesamten Dokuments einordnen möchten
<chunk>
{chunk}
</chunk>
Gib bitte einen kurzen, prägnanten Kontext, der diesen Chunk innerhalb des Gesamtdokuments verortet, um die Suche zu verbessern. Antworte nur mit diesem Kontext und nichts weiter.
"""

# Build one splitter instance up‑front so it can be reused across calls
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)


# ---------------------------------------------------------------------------
# Internal helpers

def _default_llm() -> ChatOpenAI:
    """Return a *stateless* ChatOpenAI client using the preferred model.

    Setting ``max_tokens`` **here** prevents us from passing extra parameters
    at call‑time that some back‑ends reject (e.g. ``max_completion_tokens``).
    """
    #TODO HIER VERWENDE ICH OPENAI, WEIL GWDG LIMIT NICHT REICHT
    return ChatOpenAI(
        model=CONTEXT_MODEL,
        temperature=0,
    )


def _generate_context(document_text: str, chunk_text: str, llm: ChatOpenAI) -> str:
    """Ask the LLM for a ≈50‑100‑token description that disambiguates *chunk_text*."""
    prompt = _CONTEXT_PROMPT.format(
        document=document_text[:16000],  # safeguard prompt length
        chunk=chunk_text[:2000],
    )
    # We no longer pass ``max_tokens`` here – the limit is baked into *llm*.
    return llm.predict(prompt).strip()


# ---------------------------------------------------------------------------
# Public API

def chunk_documents(
        folder: str = "documents",
        contextualize: bool = True,
        llm: Optional[ChatOpenAI] = None,
) -> Tuple[List[Document], List[Document]]:
    """Split every *.txt file inside *folder* into retrievable chunks.

    With *contextualize*=True (default) each chunk is *prepended* with a short
    LLM‑generated description of where it lives inside the parent document, as
    proposed in Anthropic's Contextual Retrieval article.  The extra text lives
    *inside* ``Document.page_content`` so it is seen by both embedding and BM25
    pipelines.  The original, raw chunk is preserved in
    ``Document.metadata['raw_chunk']`` so you can still display it unchanged.
    """
    if contextualize and llm is None:
        llm = _default_llm()

    child_chunks: List[Document] = []
    parent_docs: List[Document] = []

    for fn in Path(folder).glob("*.txt"):
        lines = fn.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue  # skip empty files

        url, *rest = lines
        full_text = "\n".join(rest).strip()
        meta = {"source": fn.name, "url": url.strip()}

        # full‑document record (helps with debugging & retrieval‑time display)
        parent_docs.append(Document(page_content=full_text, metadata=meta))

        # --- split into child chunks -------------------------------------------------
        for chunk_doc in splitter.create_documents([full_text], metadatas=[meta]):
            raw_chunk = chunk_doc.page_content  # keep a copy before we mutate it

            if contextualize:
                assert llm is not None  # for static type checkers
                context = _generate_context(full_text, raw_chunk, llm)
                print(f"Context generated for chunk {chunk_doc.metadata['source']}")
                chunk_doc.page_content = f"{context}\n\n{raw_chunk}"
                chunk_doc.metadata["context"] = context
            chunk_doc.metadata["raw_chunk"] = raw_chunk

            child_chunks.append(chunk_doc)

    return child_chunks, parent_docs