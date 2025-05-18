from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse
import os
import time
from list_of_htmls import urls
from annotate_txts import annotate_txts

# Hilfsfunktion: Extrahiere Tabellen-Inhalte
def extract_table_as_text(table_tag):
    rows = []
    table_data = []

    for tr in table_tag.find_all("tr"):
        row = []
        for cell in tr.find_all(["td", "th"]):
            text = cell.get_text(strip=True)
            if text:
                row.append(text)
        if row:
            table_data.append(row)

    if not table_data:
        return rows

    num_cols = max(len(row) for row in table_data)

    # Fall 1: Label-Spalte (th links, td rechts)
    if all(tr.find("th") and tr.find("td") for tr in table_tag.find_all("tr")) and num_cols == 2:
        for row in table_data:
            label = row[0]
            value = row[1]
            rows.append(f"{label}: {value}")
        return rows

    # Fall 2: klassische Matrix mit Kopfzeile oben (th in erster Zeile)
    if table_tag.find("tr").find("th"):
        headers = table_data[0]
        for row in table_data[1:]:
            cells = []
            for i, cell in enumerate(row):
                header = headers[i] if i < len(headers) else f"Spalte {i+1}"
                cells.append(f"{header}: {cell}")
            rows.append(" | ".join(cells))
        return rows

    # Fall 3: einfache Matrix ohne th
    for row in table_data:
        rows.append(" | ".join(row))

    return rows

# Links ersetzen durch Klartext + URL
def replace_all_links_with_text_and_url(soup):
    for a in soup.find_all("a"):
        label = a.get_text(strip=True)
        href = a.get("href")
        if href:
            a.replace_with(f"{label} ({href})")
        else:
            a.replace_with(label)

# Inline-Tags säubern
def clean_inline_tags(tag):
    for inner_tag in tag.find_all(["strong", "em", "span", "b", "i", "u"]):
        text = inner_tag.get_text(strip=True)
        inner_tag.replace_with(text + " ")


# Ausgabe-Verzeichnis
os.makedirs("documents", exist_ok=True)

# Jede URL verarbeiten
for url in urls:
    try:
        print(f"Verarbeite: {url}")
        html = requests.get(url, timeout=15).text
        soup = BeautifulSoup(html, "html.parser")

        # -- Unnötige Elemente entfernen --
        for tag in soup(["script", "style", "header", "footer", "nav", "form", "aside"]):
            tag.decompose()
        for toc in soup.find_all("div", class_="dw__toc"):
            toc.decompose()
        for br in soup.find_all("br"):
            br.replace_with("\n")
        replace_all_links_with_text_and_url(soup)

        # -- Inhalt zusammensetzen --
        output_lines = [url, ""]                     # 1. & 2. Zeile
        content_root = soup.find("div", class_="pad group")
        if content_root:
            content_root = content_root.find("div", class_="page group")
        if not content_root:
            content_root = soup.body

        for elem in content_root.descendants:
            if isinstance(elem, str):
                continue
            clean_inline_tags(elem)

            if elem.name in ["h1", "h2", "h3"]:
                output_lines.append(elem.get_text(strip=True).upper())
                output_lines.append("")

            elif elem.name == "p":
                text = " ".join(elem.get_text(separator=" ", strip=True).split())
                if len(text) > 30:
                    output_lines.append(text)
                    output_lines.append("")

            elif elem.name == "ol":
                for i, li in enumerate(elem.find_all("li"), start=1):
                    output_lines.append(f"{i}. {li.get_text(strip=True)}")
                output_lines.append("")

            elif elem.name == "ul":
                for li in elem.find_all("li"):
                    output_lines.append(f"- {li.get_text(strip=True)}")
                output_lines.append("")

            elif elem.name == "table":
                output_lines.extend(extract_table_as_text(elem))
                output_lines.append("")

            elif elem.name == "pre":
                output_lines.extend(["```", elem.get_text(separator="\n", strip=True), "```", ""])

        # -- Zusätzliche Funktionalität: Mehrfach-Zeilenumbrüche reduzieren --
        cleaned_output_lines = []
        prev_blank = False
        for line in output_lines:
            if not line.strip():
                if prev_blank:
                    # Skip zusätzliche leere Zeilen
                    continue
                prev_blank = True
                cleaned_output_lines.append("")
            else:
                prev_blank = False
                cleaned_output_lines.append(line)
        output_lines = cleaned_output_lines

        # -- Dateiname aus dritter Zeile gewinnen --
        if len(output_lines) < 3 or not output_lines[2].strip():
            # Fallback: Slug aus URL
            title_for_name = urlparse(url).path.rstrip("/").split("/")[-1] or "index"
        else:
            title_for_name = output_lines[2].strip()

        # Unerlaubte Zeichen entfernen & Leerstellen zu _
        safe_name = "".join(c for c in title_for_name if c not in r'\\/:*?"<>|').strip()
        safe_name = "_".join(filter(None, safe_name.split()))
        filepath = os.path.join("documents", f"{safe_name}.txt")

        # -- Schreiben --
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))

        time.sleep(0.5)  # fair use

    except Exception as e:
        print(f"Fehler bei {url}: {e}")
