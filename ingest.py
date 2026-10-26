"""Data pipeline: TheMealDB API -> clean/enrich -> embed -> ChromaDB.

  python ingest.py            # uses cached data/meals_raw.json if present
  python ingest.py --refresh  # re-download everything from the API
"""
import argparse
import json
import pathlib
import re
import string
import time
from collections import Counter

import requests
from langchain_core.documents import Document

from retrieval import EMBED_MODEL, SEP, get_embeddings, get_store

API = "https://www.themealdb.com/api/json/v1/1/search.php?f={}"  # free public test key "1"
CACHE = pathlib.Path("data/meals_raw.json")

NON_VEG_CATEGORIES = {"Beef", "Chicken", "Lamb", "Pork", "Seafood", "Goat"}
NON_VEG_RE = re.compile(
    r"\b(chicken|beef|pork|lamb|mutton|bacon|ham|sausages?|prawns?|shrimps?|fish|salmon|tuna|"
    r"anchov(y|ies)|cod|mackerel|squid|crab|lobster|mussels?|clams?|oysters?|turkey|duck|goat|veal)\b"
)


def fetch_all() -> list:
    """Download every meal (a-z), with retries and de-duplication by id."""
    meals = {}
    for letter in string.ascii_lowercase:
        for _ in range(3):
            try:
                r = requests.get(API.format(letter), timeout=20)
                r.raise_for_status()
                for m in r.json().get("meals") or []:
                    meals[m["idMeal"]] = m
                break
            except Exception as e:
                print(f"  retrying '{letter}': {e}")
                time.sleep(2)
        time.sleep(0.3)  # be polite to a free API
        print(f"fetched '{letter}' -> {len(meals)} unique meals")
    return list(meals.values())


def diet_of(category: str, ingredients: list) -> str:
    """Heuristic diet label (category + ingredient keywords)."""
    if category in NON_VEG_CATEGORIES or NON_VEG_RE.search(" ".join(ingredients)):
        return "non-vegetarian"
    return "vegan" if category == "Vegan" else "vegetarian"


def to_document(m: dict) -> Document:
    names, detailed = [], []
    for i in range(1, 21):
        name = (m.get(f"strIngredient{i}") or "").strip().lower()
        measure = (m.get(f"strMeasure{i}") or "").strip()
        if name:
            names.append(name)
            detailed.append(f"{measure} {name}".strip())
    category, area = m.get("strCategory") or "", m.get("strArea") or ""
    text = f"{m['strMeal']}. {area} {category} dish. Ingredients: {', '.join(names)}."  # this is what gets embedded
    return Document(
        page_content=text,
        metadata={
            "id": m["idMeal"], "title": m["strMeal"], "category": category, "area": area,
            "diet": diet_of(category, names),
            "ingredients": SEP.join(names), "ingredients_detailed": SEP.join(detailed),
            "instructions": (m.get("strInstructions") or "").strip(),
            "thumb": m.get("strMealThumb") or "", "source": m.get("strSource") or "",
        },
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    if CACHE.exists() and not args.refresh:
        meals = json.loads(CACHE.read_text())
        print(f"Loaded {len(meals)} meals from cache")
    else:
        meals = fetch_all()
        CACHE.parent.mkdir(exist_ok=True)
        CACHE.write_text(json.dumps(meals))

    docs = [to_document(m) for m in meals]
    print("Diet labels:", dict(Counter(d.metadata["diet"] for d in docs)))

    dim = len(get_embeddings().embed_query("test"))
    print(f"Embedding {len(docs)} recipes with {EMBED_MODEL} ({dim}-dim vectors)...")
    store = get_store()
    for i in range(0, len(docs), 100):
        batch = docs[i:i + 100]
        store.add_documents(batch, ids=[d.metadata["id"] for d in batch])  # ids -> re-runs update, not duplicate
        print(f"  indexed {min(i + 100, len(docs))}/{len(docs)}")
    print("Done. Now run: python graph.py   or   streamlit run app.py")


if __name__ == "__main__":
    main()