import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path

BASE_URL = "https://biomolecula.ru"
THEME_URL = "https://biomolecula.ru/themes/techno"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BioMolRetrieval/1.0)"
}

def get_article_links(URL):
    """Собираем ссылки на статьи со страницы темы"""
    links = set()
    for i in range(2,30):
        page_url  = URL + f'?page={i}'
        print(page_url)
        resp = requests.get(page_url, headers=HEADERS)
        resp.raise_for_status()

        soup = BeautifulSoup(resp.text, "html.parser")

        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/articles/"):
                if '#article-comments' in href: 
                    continue
                links.add(BASE_URL + href)
        time.sleep(1)

    return list(links)

article_links = get_article_links(THEME_URL)
print(f"Found {len(article_links)} article links")