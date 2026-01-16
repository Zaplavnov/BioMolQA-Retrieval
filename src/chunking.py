import json
from pathlib import Path

INPUT_PATH = Path("data/processed/articles_clean.jsonl")
OUTPUT_PATH = Path("data/processed/chunks.jsonl")
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

CHUNK_SIZE = 256    
CHUNK_OVERLAP = 50   

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    tokens = text.split()  # простая токенизация
    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunks.append(" ".join(chunk_tokens))
        start += chunk_size - overlap  # шаг с overlap
        if start < 0:  
            start = 0
    return chunks

def main():
    with open(INPUT_PATH, "r", encoding="utf-8") as f_in, \
         open(OUTPUT_PATH, "w", encoding="utf-8") as f_out:

        for record in f_in:
            data = json.loads(record)
            text = data["text"]
            chunks = chunk_text(text)
            for idx, chunk in enumerate(chunks):
                chunk_record = {
                    "chunk_id": f"{data['article_id']}_{idx:03d}",
                    "article_id": data["article_id"],
                    "url": data["url"],
                    "text": chunk,
                    "token_count": len(chunk.split())
                }
                f_out.write(json.dumps(chunk_record, ensure_ascii=False) + "\n")
            print(f"Article {data['article_id']} -> {len(chunks)} chunks")

if __name__ == "__main__":
    main()
