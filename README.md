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

## `annotate_txts.py`:
Hier werden jedem Dokument Beschreibungen hinzugefügt, wodurch die Genauigkeit bei der Suche erhöht werden soll
--> ist noch nicht sinnvoll evaluiert, scheint aber etwas gebracht zu haben

## `load_and_embed_html.py`:
1. embedder wird geladen (`AcademicCloudEmbeddings` ist eigene Klasse, es kann bspw. auch "OpenAIEmbeddings" verwendet werden)
2. Vector Store wird erstellt, falls noch keiner besteht
3. Parent Store wird erstellt (große Chunks, die den kleinen Chunks zuzuordnen sind)
--> Parent Chunks werden später dem LLM übergeben, nachdem die kleinen zugehörigen Chunks im Retrieval Prozess gefunden wurden

## `custom_embeddings.py`:
In dieser Klasse wird `AcademicCloudEmbeddings` erstellt, welches das Aufrufen des embedding-Modell von der GWDG ermöglicht
- es werden zusätzlich "instructions" übergeben, waas scheinbar bei e5-mistral-7b-instruct notwendig ist
- OpenAI-Embedding-Modelle haben grundsätzlich etwas besser funktioniert (ist aber auch noch nicht vollständig evaluiert)

## `list_of_html.py`:
- Liste aller Webseiten des StudIT-Wiki
- wird an `create_txt_file.py` übergeben

## `create_txt_file.py`:
#### NUR RELEVANT FÜR STUDIT WEBSEITEN
1. bekommt Liste von URLS aus `list_of_html.py`
2. geht jede einzelne URL durch:
   1. gesamte Text der HTML wird geladen
   2. "unnötige Elemente" werden entfernt (z.B. header, footer, nav)
   3. Headings werden so verarbeitet, dass sie weiter als Überschriften erkannt werden - CAPS LOCK für Überschriften
   4. Listen werden verarbeitet, damit sie auch in der txt Datei als Liste stehen, sowohl ul als auch ol
   5. Tabellen werden verarbeitet, damit in der txt Datei eine Textdarstellung der Tabellen ist, welche die Zusammenhänge von Spalten und Zeilen sinnvoll darstellt
   6. mehrfache Zeilenumbrüche werden gelöscht
   7. als Dateiname wird entweder die Überschrift genommen, welche normalerweise in der dritten Zeile erscheint oder falls keine Überschrift die URL der Seite
   8. unerlaubte Zeichen werden entfernt (falls bspw. Emojis vorkommen)
   9. alle Links (<a href) werden in Textform dargestellt
   10. inline Tags werden ignoriert
   11. txt Dateien werden in "documents" Ordner gespeichert

## `chunk_documents.py`:
- es werden child_chunks und parent_chunks erstellt
1. parent_chunks sind die vollständigen txt Dateien
2. child_chunks sind die kleinen Chunks, die durch `RecursiveCharacterTextSplitter` erstellt werden
3. child_chunks und parent_chunks haben die gleichen Metadaten, also die gleiche URL beispielsweise


#### Modellauswahl GWDG: https://chat-ai.academiccloud.de/chat/feecae75-27a6-48c5-80ec-3eced75d593c

---

<sup>1</sup>`ParentDocumentRetriever` gibt zuerst die kleinen Chunks zurück, die im Vectorstore sind und gibt auf dieser Grundlage die Parent Chunks zurück. Die kleinen Chunks und Parent Chunks sind über ihre "source" zueinander zuzuordnen. Dadurch kann die Suche dank kleiner Chunks genauer durchgeführt werden und in der Fragenbeantwortung hat das Chat LLM einen größeren Kontext dank der Parent Chunks

<sup>2</sup>Ob Vektorsuche tatsächlich besser ist als BM25 muss erst noch überprüft werden, das ist aktuell nur eine Annahme