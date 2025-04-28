import requests
import os
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathvalidate import sanitize_filename  # Stellen Sie sicher, dass pathvalidate installiert ist


start_url = "https://wiki.student.uni-goettingen.de"
visited = set()
found_urls = []

def crawl(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        for link_tag in soup.find_all('a', href=True):
            href = link_tag['href']

            # Ignore links that contain '#', '?', or '/en/'
            if '#' in href or '?' in href or '/en/' in href:
                continue

            full_url = urljoin(url, href)
            parsed_full_url = urlparse(full_url)

            # Stay inside the domain
            if parsed_full_url.netloc == urlparse(start_url).netloc:
                clean_url = parsed_full_url.scheme + "://" + parsed_full_url.netloc + parsed_full_url.path

                if clean_url not in visited:
                    visited.add(clean_url)
                    found_urls.append(clean_url)
                    crawl(clean_url)  # Recursive crawl
    except Exception as e:
        print(f"Error crawling {url}: {e}")

    return found_urls

#crawl(start_url)

# Now `found_urls` contains all collected URLs
#print(f"Total {len(found_urls)} URLs found.")


# Make sure the output directory exists
os.makedirs('documents', exist_ok=True)

def urlsToTxt(url_list):
    for url in url_list:
        # URL in einen gültigen Dateinamen umwandeln
        # Methode 1: URL kodieren
        # filename = quote(url, safe='')

        # Methode 2: Ungültige Zeichen entfernen
        filename = sanitize_filename(url)

        # Optional: Dateiendung hinzufügen
        filename += '.txt'

        try:
            response = requests.get(url)
            soup = BeautifulSoup(response.text, 'html.parser')

            # Findet das div mit class="page group"
            div = soup.find('div', class_='page group')

            if div:
                cleaned_text_parts = []
                for element in div.descendants:
                    if element.name in ['h1', 'h2', 'h3']:
                        cleaned_text_parts.append("\n\n" + element.get_text(strip=True).upper() + "\n")
                    elif element.name == 'p':
                        cleaned_text_parts.append("\n" + element.get_text(strip=True) + "\n")
                    elif element.name == 'li':
                        cleaned_text_parts.append("- " + element.get_text(strip=True))

                cleaned_text = "\n".join(cleaned_text_parts).strip()
            else:
                cleaned_text = None

            # Sicherstellen, dass das Ausgabeverzeichnis existiert
            os.makedirs('documents', exist_ok=True)

            # Datei speichern
            with open(os.path.join('documents', filename), 'w', encoding="utf-8") as file:
                file.write(f"{url}\n\n{cleaned_text if cleaned_text else 'No content found.'}")

        except Exception as e:
            print(f"Fehler beim Verarbeiten von {url}: {e}")


# Example usage:
urlsToTxt(crawl(start_url))
