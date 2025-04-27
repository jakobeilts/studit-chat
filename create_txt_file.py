import requests
from bs4 import BeautifulSoup

def urlToTxt(url, keyword):
    # fetch content from URL
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # get the text and strip whitespace
    cleaned_text = soup.get_text(separator=' ', strip=True)

    # write cleaned text to txt file with URL in first line
    with open(f'documents/{keyword}.txt', 'w', encoding="utf-8") as file:
        file.write(f"{url}\n{cleaned_text}")

# Example usage:
# urlToTxt("https://wiki.student.uni-goettingen.de/support/lsg/mobile_buchung", "mobile_buchung")
