from retrieval.tfidf import TfidfRetrieval
from retrieval.bm25 import BM25Retrieval
from retrieval.dense import DenseRetrieval

question = "Что такое CRISPR?"

tfidf = TfidfRetrieval()
tfidf.fit()


bm25 = BM25Retrieval()
bm25.fit()

dense = DenseRetrieval()
dense.fit()

retrieval = dense
top_articles = retrieval.query(question)
for a in top_articles:
    print(f"{a['article_id']} ({a['score']:.3f}) - {a['title']}")
    print(f"{a['text_preview'][:150]}...")
    print(f"URL: {a['url']}\n")