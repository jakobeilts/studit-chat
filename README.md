## `rag.py`:
- Model wird initialisiert
- Streamlit App wird initialisiert
- Prompt wird erstellt
- Chat-Verhalten wird erstellt
  - session_state Variable für die Nachrichten wird folgendermaßen aufgebaut: Zuerst wird Nachricht vom Nutzer hinzugefügt, dann die Nachricht vom LLM, wo auch der Kontext dazu gehört

## `load_and_embed_html.py`:
- lädt alle documente aus `documents`
- erstellt langchain documents und fügt metadaten hinzu (source und url)
- splittet dokumente (Chunking)
- erstellt embeddings (OpenAIEmbeddings)
- returns vetorstore

## `pages/1_Add_Document.py`:
- zweite Seite der Streamlit App
- zur Vereinfachung können hier die txt Dateien erstellt werden, welche in `documents`gespeichert werden, dahinter steckt `create_txt_file.py`

## `create_txt_file.py`:
- enthält eine Funktion, die url und ein keyword übergeben bekommt und diese url dann liest und den Inhalt als keyword.txt in `documents`speichert


# Modellauswahl:
https://chat-ai.academiccloud.de/chat/feecae75-27a6-48c5-80ec-3eced75d593c