import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

CHUNKS_PATH = Path("data/processed/chunks.jsonl")
ARTICLES_PATH = Path("data/processed/articles_clean.jsonl")
EMBEDDINGS_PATH = Path("data/processed/dense_retrieval/dense_embeddings.npz")
CHUNK_IDS_PATH = Path("data/processed/dense_retrieval/dense_chunk_ids.json")
TOP_K = 3
MODEL_NAME = "sentence-transformers/multi-qa-mpnet-base-dot-v1"  # можно заменить на ruBERT-tiny или multilingual-e5


class DenseRetrieval:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = None  # Lazy loading - загрузим только когда нужно
        self.chunk_texts = []
        self.chunk_ids = []
        self.chunk_embeddings = None
        self.article_mapping = {}
    
    def _get_model(self):
        """Ленивая загрузка модели - только когда она действительно нужна"""
        if self.model is None:
            print(f"Loading model {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
        return self.model

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

        # Загружаем чанки (нужны всегда, даже при использовании кэша)
        self.chunk_texts = []
        self.chunk_ids = []
        with open(chunks_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                self.chunk_ids.append(data["chunk_id"])
                self.chunk_texts.append(data["text"])

        # Кэширование
        if EMBEDDINGS_PATH.exists() and CHUNK_IDS_PATH.exists() and not force_recompute:
            print("Loading cached embeddings...")
            self.chunk_embeddings = np.load(EMBEDDINGS_PATH)["arr_0"]
            print(f"Loaded embeddings shape: {self.chunk_embeddings.shape}")
            return

        # Вычисляем эмбеддинги
        print(f"Encoding {len(self.chunk_texts)} chunks with {self.model_name} ...")
        model = self._get_model()
        # На CPU 32-64 batch_size
        batch_size =  64
        print(f"Using batch_size: {batch_size}")
        self.chunk_embeddings = model.encode(
            self.chunk_texts, 
            show_progress_bar=True, 
            convert_to_numpy=True,
            batch_size=batch_size,
            normalize_embeddings=True  # Нормализация для лучшей производительности
        )

        # Сохраняем кэш
        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        CHUNK_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_IDS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.chunk_ids, f, ensure_ascii=False)

        print(f"Embeddings saved. Shape: {self.chunk_embeddings.shape}")

    def query(self, question, top_k=TOP_K):
        # Вычисляем эмбеддинг вопроса
        model = self._get_model()
        q_emb = model.encode([question], convert_to_numpy=True)

        # Косинусная близость
        sims = cosine_similarity(q_emb, self.chunk_embeddings).flatten()

        # Топ-100 чанков для агрегации
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

            if article_id not in article_scores or score > article_scores[article_id]:
                article_scores[article_id] = score
                article_texts[article_id] = self.chunk_texts[i]
                article_titles[article_id] = self.article_mapping.get(article_id, {}).get("title", "")
                article_urls[article_id] = self.article_mapping.get(article_id, {}).get("url", None)

        # Сортировка и top-k
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