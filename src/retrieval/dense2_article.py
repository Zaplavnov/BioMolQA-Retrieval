import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ARTICLES_PATH = Path("data/processed/articles_clean.jsonl")
EMBEDDINGS_PATH = Path("data/processed/dense_article/dense_article_embeddings.npz")
ARTICLE_IDS_PATH = Path("data/processed/dense_article/dense_article_ids.json")
TOP_K = 3
# лёгкая и быстрая мультилингвальная модель
MODEL_NAME = "intfloat/multilingual-e5-small"  

class DenseArticleRetrieval:
    def __init__(self, model_name=MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.article_texts = []
        self.article_ids = []
        self.article_titles = {}
        self.article_urls = {}
        self.article_embeddings = None

    def _get_model(self):
        if self.model is None:
            print(f"Loading model {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
        return self.model

    def fit(self, articles_file=ARTICLES_PATH, force_recompute=False):
        # загрузить статьи
        self.article_texts = []
        self.article_ids = []
        self.article_titles = {}
        self.article_urls = {}

        with open(articles_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                aid = data["article_id"]
                self.article_ids.append(aid)
                # используем весь текст статьи; можно обрезать до первых 1000 токенов при желании
                self.article_texts.append(data["text"])
                self.article_titles[aid] = data.get("title", "")
                self.article_urls[aid] = data.get("url", None)

        # кэширование
        if EMBEDDINGS_PATH.exists() and ARTICLE_IDS_PATH.exists() and not force_recompute:
            print("Loading cached article embeddings...")
            self.article_embeddings = np.load(EMBEDDINGS_PATH)["arr_0"]
            print(f"Loaded embeddings shape: {self.article_embeddings.shape}")
            return

        # вычисляем эмбеддинги (быстро: количество ~число статей)
        model = self._get_model()
        print(f"Encoding {len(self.article_texts)} articles with {self.model_name} ...")
        # batch_size можно увеличить, но мелкая модель справится быстро
        self.article_embeddings = model.encode(
            self.article_texts,
            show_progress_bar=True,
            convert_to_numpy=True,
            batch_size=64,
            normalize_embeddings=True
        )

        # сохраняем кэш
        EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ARTICLE_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(EMBEDDINGS_PATH, self.article_embeddings)
        with open(ARTICLE_IDS_PATH, "w", encoding="utf-8") as f:
            json.dump(self.article_ids, f, ensure_ascii=False)

        print(f"Saved article embeddings, shape: {self.article_embeddings.shape}")

    def query(self, question, top_k=TOP_K):
        if self.article_embeddings is None:
            raise ValueError("Call fit() first to build embeddings.")

        model = self._get_model()
        q_emb = model.encode([question], convert_to_numpy=True)

        sims = cosine_similarity(q_emb, self.article_embeddings).flatten()
        top_idx = np.argsort(sims)[::-1][:top_k]

        results = []
        for i in top_idx:
            aid = self.article_ids[i]
            results.append({
                "article_id": aid,
                "score": float(sims[i]),
                "title": self.article_titles.get(aid, ""),
                "url": self.article_urls.get(aid),
                # preview: короткая часть статьи
                "text_preview": self.article_texts[i][:400]
            })
        return results