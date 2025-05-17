from pathlib import Path
import json
from langchain.chat_models import ChatOpenAI
import streamlit as st

# ------------ CONFIG ---------------------------------------------------------
MODEL_NAME = "meta-llama-3.1-8b-instruct"       # or whatever you like
BASE_URL   = st.secrets["BASE_URL"]             # if needed
API_KEY    = st.secrets["GWDG_API_KEY"]         # 👉 replace or read from env



# -----------------------------------------------------------------------------

llm = ChatOpenAI(
    model=MODEL_NAME,
    temperature=0,
    base_url=BASE_URL,
    api_key=API_KEY,
)

PROMPT_TEMPLATE = """
You will receive a plain-text wiki article (without markup).
1. Write a single concise description of the **whole article** in max 50 words.
2. Invent a short HEADER — max 3 words — that captures its topic. Write the HEADER in a way that it can be used as a file title like this "test_file_name"
Write in german only!
Return **JSON** exactly in the form:
{{"description":"…","header":"…"}}

---
ARTICLE:
{content}
---
"""

def generate_meta(text: str) -> tuple[str, str]:
    """Ask the LLM for description + header and return them."""
    prompt = PROMPT_TEMPLATE.format(content=text[:8000])  # stay within context
    response = llm.invoke(prompt).content.strip()
    try:
        data = json.loads(response)
        desc  = data["description"].strip().replace("\n", " ")
        head  = data["header"].strip().replace("\n", " ")
        return desc, head
    except (ValueError, KeyError):
        # Fallback if the model did not return valid JSON
        return "", ""

def annotate_file(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines:
        return
    url_line  = lines[0]
    body_text = "\n".join(lines[1:])          # feed everything except URL
    desc, head = generate_meta(body_text)

    new_lines = [url_line, desc, head]        # URL + description + header
    if len(lines) > 1:
        new_lines += lines[2:]                # keep rest of original file

    path.write_text("\n".join(new_lines), encoding="utf-8")
    print(f"✓ {path.name}  →  '{head}'")


def annotate_txts(txt_folder):
    for txt_file in Path(txt_folder).glob("*.txt"):
        annotate_file(txt_file)
