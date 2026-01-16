import json
from pathlib import Path
from rank_bm25 import BM25Okapi

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
TOP_K = 3

class BM25Retrieval:
    def __init__(self):
        self.tokenized_chunks = []   # список токенов каждого чанка
        self.chunk_ids = []
        self.article_mapping = {}    # article_id -> {title, url}
        self.bm25 = None

    def fit(self, chunks_file=CHUNKS_PATH, articles_file=Path("data/processed/articles_clean.jsonl")):
        # Загружаем article_id -> title, url
        self.article_mapping = {}
        with open(articles_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.article_mapping[data["article_id"]] = {
                    "title": data["title"],
                    "url": data["url"]
                }

        # Загружаем чанки
        self.tokenized_chunks = []
        self.chunk_ids = []

        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.chunk_ids.append(data["chunk_id"])
                tokens = data["text"].split()
                self.tokenized_chunks.append(tokens)

        # Инициализация BM25
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def query(self, question, top_k=TOP_K):
        # Токенизация вопроса
        q_tokens = question.split()

        # BM25 scores
        scores = self.bm25.get_scores(q_tokens)

        # Топ-100 чанков для агрегации
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:100]

        # Агрегация по article_id
        article_scores = {}
        article_texts = {}
        article_titles = {}
        article_urls = {}

        for i in top_idx:
            chunk_id = self.chunk_ids[i]
            # Извлекаем article_id: все части chunk_id кроме последней (индекса чанка)
            article_id = "_".join(chunk_id.split("_")[:-1])
            score = scores[i]

            if article_id not in article_scores or score > article_scores[article_id]:
                article_scores[article_id] = score
                # text_preview пока просто первая часть чанка
                article_texts[article_id] = " ".join(self.tokenized_chunks[i])
                article_titles[article_id] = self.article_mapping.get(article_id, {}).get("title", "")
                article_urls[article_id] = self.article_mapping.get(article_id, {}).get("url", None)

        # Сортировка и топ-k
        sorted_articles = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for article_id, score in sorted_articles:
            results.append({
                "article_id": article_id,
                "score": score,
                "title": article_titles[article_id],
                "text_preview": article_texts[article_id],
                "url": article_urls[article_id]
            })

        return results