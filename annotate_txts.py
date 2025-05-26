from pathlib import Path
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import streamlit as st
import json
from typing import Union

"""
Annotate each .txt file in a directory with a one‑line German description.

Call `annotate_txts(<folder>)` from other scripts, for example:
    from annotate_txts import annotate_txts
    annotate_txts("seiten_export")

If you execute this script directly (`python annotate_txts.py`), it will
process the default `./documents` folder.
"""

# ── LLM setup ──────────────────────────────────────────────────────────────────
# DAS HIER MACHT OFT PROBLEME, WEIL API LIMIT ÜBERSCHRITTEN WIRD. ANNOTATIONS LIEBER ERSTMAL MIT EIGENEN OPENAI KONTINGENTEN!
try:
    llm = ChatOpenAI(
        model="meta-llama-3.1-8b-instruct",
        temperature=0,
        base_url=st.secrets["BASE_URL"],
        api_key=st.secrets["GWDG_API_KEY"],
    )
except KeyError as e:
    raise RuntimeError(
        "Missing Streamlit secret. Run inside Streamlit or set BASE_URL and "
        "GWDG_API_KEY in st.secrets."
    ) from e

prompt_template = PromptTemplate.from_template(
    """
You will receive a plain‑text wiki article (without markup).

1. Write a single concise description of the **whole article** in **max 80 words**.
2. Write **in German only**.
3. Use important keywords that appear in the article to make later keyword search easier.
4. If you don't understand the context or the context is too small, return an empty string.

Return **JSON** exactly in the form:
{{"description":"…"}}

---
ARTICLE:
{context}
---
"""
)

# Build the runnable sequence (Prompt → LLM)
runnable = prompt_template | llm


# ── Helper ────────────────────────────────────────────────────────────────────

def _to_text(msg: Union[str, "langchain_core.messages.BaseMessage"]) -> str:
    """Return the content string regardless of msg type (raw str or LC message)."""
    if isinstance(msg, str):
        return msg
    return getattr(msg, "content", str(msg))


# ── Main function ─────────────────────────────────────────────────────────────

def annotate_txts(folder_path: str = "./documents") -> None:
    """Insert a JSON summary as the 2nd visible line of every *.txt in *folder_path*.

    • If the file already has content on line 2, we *insert* a blank line first so
      the description never overwrites existing content.
    • Existing (non‑blank) descriptions therefore remain untouched; fresh ones
      appear right above them.
    """

    loader = DirectoryLoader(folder_path, glob="**/*.txt")
    docs = loader.load()
    print(f"{len(docs)} documents found and loaded from {Path(folder_path).resolve()}")

    for doc in docs:
        file_path: str = doc.metadata["source"]
        content: str = doc.page_content

        # ── Ask the model for a summary ────────────────────────────────────
        try:
            raw_response = runnable.invoke({"context": content})
            result_text = _to_text(raw_response)
            parsed = json.loads(result_text)
            description: str = parsed.get("description", "")
        except Exception as e:
            description = ""
            print(f"Warning: Error processing {file_path}: {e}")

        # ── Insert description into the file ───────────────────────────────
        path_obj = Path(file_path)
        lines = path_obj.read_text(encoding="utf-8").splitlines(keepends=True)

        description_line = f"{description}\n"

        if lines:
            if len(lines) == 1:
                # Only one line exists → append description as 2nd line
                lines.append(description_line)
            else:
                # There is already a 2nd line
                if lines[1].strip():
                    # Non‑empty → keep it, insert new description above it
                    lines.insert(1, description_line)
                else:
                    # Empty 2nd line → replace it
                    lines[1] = description_line
        else:
            # Empty file
            lines = [description_line]

        path_obj.write_text("".join(lines), encoding="utf-8")
        print(f"Updated {file_path} with summary: {description[:60]}…")
