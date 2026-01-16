import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.sparse as sp
import joblib

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
ARTICLES_PATH = Path("data/processed/articles_clean.jsonl")
# для кэширования
VECTOR_PATH = Path("data/processed/tfidf/tfidf_vectors.npz")
CHUNK_IDS_PATH = Path("data/processed/tfidf/chunk_ids.json")
VECTORIZER_PATH = Path("data/processed/tfidf/vectorizer.joblib")
CHUNK_TEXTS_PATH = Path("data/processed/tfidf/chunk_texts.json")
TOP_K = 3

class TfidfRetrieval:
    def __init__(self):
        self.vectorizer = None
        self.chunk_texts = []
        self.chunk_ids = []
        self.tfidf_matrix = None
        self.article_mapping = {}  # article_id -> {title, url}

    def fit(self, chunks_file=CHUNKS_PATH, articles_file=ARTICLES_PATH, force_recompute=False):
        # Загружаем article_id -> title, url
        self.article_mapping = {}
        with open(articles_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.article_mapping[data["article_id"]] = {
                    "title": data["title"],
                    "url": data["url"]
                }

        # Если кэш есть
        if (VECTOR_PATH.exists() and CHUNK_IDS_PATH.exists() and 
            VECTORIZER_PATH.exists() and CHUNK_TEXTS_PATH.exists() and not force_recompute):
            self.tfidf_matrix = sp.load_npz(VECTOR_PATH)
            with open(CHUNK_IDS_PATH, "r", encoding="utf-8") as f:
                self.chunk_ids = json.load(f)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            with open(CHUNK_TEXTS_PATH, "r", encoding="utf-8") as f:
                self.chunk_texts = json.load(f)
            return

        # Загружаем чанки
        self.chunk_texts = []
        self.chunk_ids = []

        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.chunk_ids.append(data["chunk_id"])
                self.chunk_texts.append(data["text"])

        # TF-IDF векторы
        self.vectorizer = TfidfVectorizer(max_features=50000)
        self.tfidf_matrix = self.vectorizer.fit_transform(self.chunk_texts)

        # Сохраняем в кэш
        VECTOR_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHUNK_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        sp.save_npz(VECTOR_PATH, self.tfidf_matrix)
        with open(CHUNK_IDS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunk_ids, f, ensure_ascii=False)
        joblib.dump(self.vectorizer, VECTORIZER_PATH)
        with open(CHUNK_TEXTS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunk_texts, f, ensure_ascii=False)

        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")

    def query(self, question, top_k=TOP_K):
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF matrix not built. Call fit() first.")

        # Векторизуем вопрос
        q_vec = self.vectorizer.transform([question])

        # Косинусная близость
        sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()

        # Сначала топ-100 чанков
        top_idx = np.argsort(sims)[::-1][:100]

        # Агрегация по article_id
        article_scores = {}
        article_texts = {}
        article_titles = {}
        article_urls = {}

        for i in top_idx:
            chunk_id = self.chunk_ids[i]
            # Извлекаем article_id: все части chunk_id кроме последней (индекса чанка)
            article_id = "_".join(chunk_id.split("_")[:-1])
            score = float(sims[i])

            # Сохраняем лучший score для статьи
            if article_id not in article_scores or score > article_scores[article_id]:
                article_scores[article_id] = score
                article_texts[article_id] = self.chunk_texts[i]
                article_titles[article_id] = self.article_mapping.get(article_id, {}).get("title", "")
                article_urls[article_id] = self.article_mapping.get(article_id, {}).get("url", None)

        # Сортировка статей по max score
        sorted_articles = sorted(article_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        # Формируем результат
        results = []
        for article_id, score in sorted_articles:
            results.append({
                "article_id": article_id,
                "score": score,
                "title": article_titles.get(article_id, ""),
                "text_preview": article_texts[article_id],
                "url": article_urls.get(article_id)
            })

        return results

