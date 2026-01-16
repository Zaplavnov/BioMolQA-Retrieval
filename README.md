# BioMolQA-Retrieval

Retrieval-based question-answering system for Russian-language molecular biology content using articles from [biomolecula.ru](https://biomolecula.ru).

## Overview

This project implements and compares three retrieval methods for educational QA:
- **TF-IDF** (baseline) - classical keyword-based retrieval
- **BM25** - probabilistic ranking function
- **Dense Retrieval** - multilingual sentence transformers (multilingual-e5-small)

The system retrieves relevant articles from a corpus of 617 Russian-language articles on molecular biology and biotechnology.

## Results

Evaluation on 100 synthetic multi-article questions:

| Method | Recall@1 | Recall@3 | Recall@5 | MRR |
|--------|----------|-----------|----------|-----|
| TF-IDF | 0.35 | 0.51 | **0.60** | **0.44** |
| BM25 | 0.08 | 0.21 | 0.39 | 0.18 |
| Dense | 0.31 | 0.47 | 0.53 | 0.39 |

TF-IDF outperforms dense retrieval, likely due to strong lexical alignment between questions and articles.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Retrieval

```python
from src.retrieval.tfidf import TfidfRetrieval
from src.retrieval.bm25 import BM25Retrieval
from src.retrieval.dense2_article import DenseArticleRetrieval

# Initialize and fit a retriever
retriever = TfidfRetrieval()
retriever.fit()

# Query
question = "Что такое CRISPR?"
results = retriever.query(question, top_k=5)

for article in results:
    print(f"{article['title']} - {article['score']:.3f}")
    print(f"URL: {article['url']}\n")
```

### Evaluation

Run evaluation on the synthetic QA dataset:

```bash
python src/evaluate.py
```

This will evaluate all three methods and save results to `data/eval/`.

## Project Structure

```
├── src/
│   ├── retrieval/          # Retrieval method implementations
│   │   ├── tfidf.py        # TF-IDF retrieval
│   │   ├── bm25.py         # BM25 retrieval
│   │   └── dense2_article.py  # Dense retrieval
│   ├── scrape.py           # Web scraping from biomolecula.ru
│   ├── preprocessing.py    # Text cleaning
│   ├── chunking.py         # Document chunking
│   ├── evaluate.py          # Evaluation script
│   └── retrieve.py          # Example usage
├── data/
│   ├── raw/                 # Raw scraped articles
│   ├── processed/           # Cleaned articles and chunks
│   └── eval/                # Evaluation dataset and results
└── requirements.txt
```

## Dataset

- **Corpus**: 617 articles from biomolecula.ru/themes/techno
- **Evaluation**: 100 synthetic multi-article questions
- **Language**: Russian

## License

See LICENSE file.
