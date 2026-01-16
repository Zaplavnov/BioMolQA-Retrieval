import json
from pathlib import Path
import re
from bs4 import BeautifulSoup
import unicodedata

INPUT_PATH = Path("data/raw/articles_raw.jsonl")
OUTPUT_PATH = Path("data/processed/articles_clean.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def clean_text(text):
    # удаляем HTML-теги на всякий
    soup = BeautifulSoup(text, "html.parser")
    text = soup.get_text(separator=" ")

    # удаляем лишние пробелы и переносы
    text = re.sub(r"\s+", " ", text)

    # нормализация Unicode (например, ё → е)
    text = unicodedata.normalize("NFC", text)

    # удаление ссылки вида [1], [2]
    text = re.sub(r"\[\d+\]", "", text)

    return text.strip()


def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:

        for line in f_in:
            record = json.loads(line)
            text_raw = record.get("text_raw", "")
            if not text_raw:
                continue

            text_clean = clean_text(text_raw)

            new_record = {
                "article_id": record["article_id"],
                "url": record["url"],
                "title": record["title"],
                "text": text_clean
            }

            f_out.write(json.dumps(new_record, ensure_ascii=False) + "\n")

            print(f"Processed: {record['article_id']}")

if __name__ == "__main__":
    main()