## `rag.py`:
1. llm wird geladen 
   - dazu wird `ChatOpenAI` von Langchain verwendet
3. Vectorstore und Parent Store werden von `load_and_embed_html.py` geladen
4. Parent Retriever von Langchain `ParentDocumentRetriever`<sup>1</sup> wird eingerichtet
5. es werden sowohl "alle Nachrichten", als auch nur die "letzte Nutzeranfrage" in unterschiedlichen Listen gespeichert
6. Nach Eingabe des Prompt:
   - Eingegebene Nachricht (Prompt) wird der Liste aller Nachrichten und der Liste der letzten Nurzernachrichten angehangen
   - Standalone query wird gebaut. Diese besteht aus den letzten Nachrichten und dem Prompt
   - Retrieval Prozess auf Parent Retriever (Vectorsuche)
   - Retrieval Prozess BM25 (Keyword Suche)
   - Da Vektorsuche verlässlicher ist als BM25 werden von der BM25 Suche nur Chunks übernommen, die noch nicht in der Vektorsuche gefunden wurden<sup>2</sup>


#### Modellauswahl: https://chat-ai.academiccloud.de/chat/feecae75-27a6-48c5-80ec-3eced75d593c

---

<sup>1</sup>`ParentDocumentRetriever` gibt zuerst die kleinen Chunks zurück, die im Vectorstore sind und gibt auf dieser Grundlage die Parent Chunks zurück. Die kleinen Chunks und Parent Chunks sind über ihre "source" zueinander zuzuordnen. Dadurch kann die Suche dank kleiner Chunks genauer durchgeführt werden und in der Fragenbeantwortung hat das Chat LLM einen größeren Kontext dank der Parent Chunks

<sup>2</sup>Ob Vektorsuche tatsächlich besser ist als BM25 muss erst noch überprüft werden, das ist aktuell nur eine Annahme