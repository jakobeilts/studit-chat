import os
from pathlib import Path
from typing import List, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------------------
CHUNK_SIZE, CHUNK_OVERLAP = 800, 100
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", " ", ""],
)

def chunk_documents(folder: str = "seiten_export") -> Tuple[List[Document], List[Document]]:
    """
    Walk *folder* and create
      • parent_docs – one Document per .txt file (full text)
      • child_chunks – split-up versions of each parent, carrying the same metadata
    Returns (child_chunks, parent_docs)
    """
    child_chunks, parent_docs = [], []

    for fn in Path(folder).glob("*.txt"):
        url, *rest = fn.read_text(encoding="utf-8").splitlines()
        full_text = "\n".join(rest).strip()
        meta = {"source": fn.name, "url": url.strip()}

        parent_doc = Document(page_content=full_text, metadata=meta)
        parent_docs.append(parent_doc)

        child_chunks.extend(
            splitter.create_documents([full_text], metadatas=[meta])
        )

    return child_chunks, parent_docs
