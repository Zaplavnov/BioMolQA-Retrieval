import json
import time
import random
import sys
from pathlib import Path
from nltk import sent_tokenize

# Добавляем корневую директорию проекта в sys.path для импортов
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.prompt import PROMPT_TEMPLATE
from src.models.registry import get_model

ARTICLES_PATH = Path("data/processed/articles_clean.jsonl")
OUT_PATH = Path("data/eval/qa_synthetic.jsonl")
GROUP_SIZE = 5
N_GROUPS = 100  # количество групп вопросов
SNIPPET_MAX_CHARS = 2500  # максимум текста на статью (чтобы поместить в prompt)

# Конфигурация модели
MODEL_NAME = "Qwen/Qwen3-Coder-480B-A35B-Instruct"
PROVIDER = "cloud"
TEMPERATURE = 0.7


def load_articles(path):
    arts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            arts.append(json.loads(line))
    return arts

def make_snippet(text, max_chars=SNIPPET_MAX_CHARS):
    # Попробуем взять первые 3 предложения (nltk), иначе просто первые max_chars
    try:
        sents = sent_tokenize(text)
        if len(sents) >= 3:
            snippet = " ".join(sents[:4])  # 3-4 предложения
        else:
            snippet = text[:max_chars]
    except Exception:
        snippet = text[:max_chars]
    return snippet.replace("\n", " ").strip()


def extract_json_from_text(s):
    # Попробуем найти JSON-объект в тексте (по первым { ... })
    s = s.strip()
    # Наиболее частый случай: модель вернула ровно JSON
    try:
        return json.loads(s)
    except Exception:
        # попытаемся найти подстроку, начинающуюся с { и заканчивающуюся на }
        start = s.find("{")
        end = s.rfind("}")
        if start != -1 and end != -1 and end > start:
            sub = s[start:end+1]
            try:
                return json.loads(sub)
            except Exception:
                return None
        return None

def main():
    # Инициализируем модель
    print(f"Инициализация модели: {MODEL_NAME} (provider: {PROVIDER})")
    model = get_model(
        provider=PROVIDER,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE
    )
    print(f"Модель инициализирована: {model.info()}")
    
    articles = load_articles(ARTICLES_PATH)
    random.shuffle(articles)
    total = len(articles)
    max_groups = min(N_GROUPS, total // GROUP_SIZE)
    print(f"Total articles: {total}, will build {max_groups} groups of {GROUP_SIZE}")

    groups = []
    for i in range(max_groups):
        group = articles[i*GROUP_SIZE:(i+1)*GROUP_SIZE]
        groups.append(group)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for gi, group in enumerate(groups, start=1):
        # Build combined text (title + snippet per article)
        article_texts = []
        for j, art in enumerate(group, start=1):
            snippet = make_snippet(art.get("text", ""), SNIPPET_MAX_CHARS)
            article_texts.append(f"{art.get('title','')}\n{snippet}")

        prompt = PROMPT_TEMPLATE.format(
            article_1=article_texts[0],
            article_2=article_texts[1],
            article_3=article_texts[2],
            article_4=article_texts[3],
            article_5=article_texts[4],
        )

        # call model (with retry)
        for attempt in range(3):
            try:
                print(f"[Group {gi}/{max_groups}] Calling model, attempt {attempt+1}")
                raw = model.generate(prompt)
                parsed = extract_json_from_text(raw)
                if parsed is None:
                    print("Warning: could not parse JSON from model output; saving raw output for inspection.")
                    parsed = {"question": None, "required_articles": [], "explanation": "", "raw": raw}
                # ensure required_articles are indices 1..5 (convert if needed)
                ra = parsed.get("required_articles") or []
                # normalize to list of ints (if strings like "1,2,3")
                try:
                    ra = [int(x) for x in ra]
                except Exception:
                    ra = []
                # if <3 then we fallback to all 5
                if len(ra) < 3:
                    ra = list(range(1, GROUP_SIZE+1))

                out_record = {
                    "group_id": gi,
                    "article_ids": [art["article_id"] for art in group],
                    "question": parsed.get("question"),
                    "required_articles": ra,
                    "explanation": parsed.get("explanation",""),
                    "raw_model_output": raw
                }
                results.append(out_record)
                # be polite
                time.sleep(1.0)
                break
            except Exception as e:
                print(f"Error calling model: {e}")
                time.sleep(2 + attempt*2)
        else:
            # all retries failed
            results.append({
                "group_id": gi,
                "article_ids": [art["article_id"] for art in group],
                "question": None,
                "required_articles": [],
                "explanation": "",
                "raw_model_output": ""
            })

    # Save results
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Saved {len(results)} generated questions to {OUT_PATH}")

if __name__ == "__main__":
    main()