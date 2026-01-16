import requests
from bs4 import BeautifulSoup
import json
import time
from pathlib import Path

BASE_URL = "https://biomolecula.ru"
THEME_URL = "https://biomolecula.ru/themes/techno"

OUTPUT_PATH = Path("data/raw/articles_raw.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; BioMolRetrieval/1.0)"
}

def get_article_links(URL):
    """Собираем ссылки на статьи со страницы темы"""
    links = set()
    for i in range(2,64):
        try:
            page_url  = URL + f'?page={i}'
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
        except Exception as e:
            print(f"Ошибка на странице {i} во время сборки ссылок: {e}")

    return list(links)


def parse_article(url):
    """Парсим одну статью"""
    resp = requests.get(url, headers=HEADERS)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Заголовок
    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else ""

    # Основной текст (может потребовать уточнения селектора)
    content_div = soup.find("div", class_="article_content vizred")
    if content_div:
        text_raw = content_div.get_text(separator=" ", strip=True)
    else:
        text_raw = ""

    return title, text_raw


def main():
    article_links = get_article_links(THEME_URL)
    print(f"Found {len(article_links)} article links")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for idx, url in enumerate(article_links):
            try:
                title, text_raw = parse_article(url)

                if not text_raw:
                    continue

                record = {
                    "article_id": f"biomol_{idx:05d}",
                    "url": url,
                    "title": title,
                    "text_raw": text_raw
                }

                f.write(json.dumps(record, ensure_ascii=False) + "\n")

                print(f"[{idx+1}/{len(article_links)}] Parsed: {title[:60]}")

                time.sleep(1)

            except Exception as e:
                print(f"Error parsing {url}: {e}")


if __name__ == "__main__":
    main()