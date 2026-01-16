import json
from pathlib import Path
from time import time

from retrieval.tfidf import TfidfRetrieval
from retrieval.bm25 import BM25Retrieval
from retrieval.dense2_article import DenseArticleRetrieval

QA_PATH = Path("data/eval/qa_synthetic.jsonl")
OUT_DIR = Path("data/eval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TOP_KS = [1, 3, 5]          # метрики Recall@1,3,5
MAX_K = max(TOP_KS)        # сколько возвращаем у retriever (запросим 5)
METHODS = ["tfidf", "bm25", "dense"]

def load_qa(path):
    qs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            qs.append(json.loads(line))
    return qs

def compute_metrics_for_method(all_qs, retriever, method_name):
    recall_counts = {k: 0 for k in TOP_KS}
    mrr_sum = 0.0
    n_processed = 0

    per_question_results = []

    for q in all_qs:
        question = q.get("question") or q.get("question_text") or ""
        gt = set()
        if "required_articles" in q and q["required_articles"]:
            if "article_ids" in q and q["article_ids"]:
                try:
                    idxs = q["required_articles"]
                    mapped = [ q["article_ids"][i-1] for i in idxs if 1 <= i <= len(q["article_ids"]) ]
                    gt.update(mapped)
                except Exception:
                    gt.update(q.get("article_ids", []))
            else:
                gt.update(q.get("article_ids", []))
        else:
            gt.update(q.get("article_ids", []))

        if not gt:
            continue

        n_processed += 1

        try:
            ranked = retriever.query(question, top_k=MAX_K)
        except TypeError:
            ranked = retriever.query(question)
            ranked = ranked[:MAX_K]

        retrieved_ids = [r.get("article_id") for r in ranked if r.get("article_id")]

        for k in TOP_KS:
            topk = set(retrieved_ids[:k])
            if len(gt & topk) > 0:
                recall_counts[k] += 1

        rr = 0.0
        for rank, aid in enumerate(retrieved_ids, start=1):
            if aid in gt:
                rr = 1.0 / rank
                break
        mrr_sum += rr

        per_question_results.append({
            "question": question,
            "gt_article_ids": list(gt),
            "retrieved_ids": retrieved_ids,
            "reciprocal_rank": rr
        })

    # Aggregate
    if n_processed == 0:
        metrics = {f"Recall@{k}": 0.0 for k in TOP_KS}
        metrics["MRR"] = 0.0
        metrics["n_questions"] = 0
        metrics["n_processed"] = 0
    else:
        metrics = {}
        for k in TOP_KS:
            metrics[f"Recall@{k}"] = recall_counts[k] / n_processed
        metrics["MRR"] = mrr_sum / n_processed
        metrics["n_questions"] = len(all_qs)
        metrics["n_processed"] = n_processed

    return metrics, per_question_results

def main():
    print("Loading QA dataset...")
    qs = load_qa(QA_PATH)
    print(f"Loaded {len(qs)} questions.")

    print("Initializing retrievers and building indexes (may load caches)...")
    # TF-IDF
    tfidf = TfidfRetrieval()
    tfidf.fit()

    # BM25
    bm25 = BM25Retrieval()
    bm25.fit()

    # Dense article
    dense = DenseArticleRetrieval()
    dense.fit()

    retrievers = {"tfidf": tfidf, "bm25": bm25, "dense": dense}

    all_metrics = {}
    for method in METHODS:
        print(f"Evaluating method: {method}")
        start = time()
        metrics, per_q = compute_metrics_for_method(qs, retrievers[method], method)
        dur = time() - start
        print(f"Time for {method}: {dur:.2f}s")
        print("Metrics:", metrics)
        all_metrics[method] = metrics

        out_file = OUT_DIR / f"results_{method}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for item in per_q:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved per-question results to {out_file}")

    with open(OUT_DIR / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2, ensure_ascii=False)

    print("\nSUMMARY")
    for method, m in all_metrics.items():
        print(f"Method: {method}")
        for k in TOP_KS:
            print(f"  Recall@{k}: {m[f'Recall@{k}']:.3f}")
        print(f"  MRR: {m['MRR']:.3f}")
        print("")

if __name__ == "__main__":
    main()