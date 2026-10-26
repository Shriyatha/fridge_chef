"""The RAG core: embeddings, vector store, semantic search, metadata filtering, reranking."""
from functools import lru_cache
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 384-dimensional sentence embeddings. Runs locally and is free.
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHROMA_DIR = "chroma_db"
SEP = "|"  # separator for list fields stored in Chroma metadata (it only allows primitives)

PANTRY = {"salt", "water", "oil", "olive oil", "vegetable oil", "pepper", "black pepper",
          "sugar", "flour", "butter", "ground pepper", "sea salt"}

# Metadata filters: a hard constraint applied *inside* the vector search
DIET_FILTER = {
    "vegetarian": {"diet": {"$in": ["vegetarian", "vegan"]}},
    "vegan": {"diet": "vegan"},
}


@lru_cache(maxsize=1)
def get_embeddings():
    # normalize_embeddings=True -> dot product == cosine similarity
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL, encode_kwargs={"normalize_embeddings": True})


@lru_cache(maxsize=1)
def get_store():
    return Chroma(
        collection_name="meals",
        embedding_function=get_embeddings(),
        persist_directory=CHROMA_DIR,
        collection_metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine similarity
    )


def _norm(s: str) -> str:
    s = s.lower().strip()
    return s[:-1] if len(s) > 3 and s.endswith("s") else s  # crude singularisation


def _match(a: str, b: str) -> bool:
    a, b = _norm(a), _norm(b)
    return len(a) > 2 and len(b) > 2 and (a in b or b in a)


def is_pantry(ingredient: str) -> bool:
    return ingredient.lower().strip() in PANTRY or _norm(ingredient) in PANTRY


def vector_search(ingredients: List[str], diet: str = "none", k: int = 8, min_sim: float = 0.0) -> list:
    """Step 1 - retrieve: embed the query, find nearest recipes, apply the diet filter."""
    query = "Ingredients: " + ", ".join(ingredients)
    results = get_store().similarity_search_with_score(query, k=k, filter=DIET_FILTER.get(diet))
    hits = []
    for doc, dist in results:
        sim = round(1 - dist, 3)  # cosine distance -> similarity
        if sim >= min_sim:
            hits.append({**doc.metadata, "similarity": sim})
    return hits


def rerank(hits: list, user_ingredients: List[str], w_vec: float = 0.6) -> list:
    """Step 2 - rerank: blend semantic similarity with how many of the user's ingredients the recipe uses."""
    out = []
    for h in hits:
        recipe_ings = h["ingredients"].split(SEP)
        matched = [u for u in user_ingredients if any(_match(u, r) for r in recipe_ings)]
        missing = [r for r in recipe_ings
                   if not any(_match(u, r) for u in user_ingredients) and not is_pantry(r)]
        coverage = len(matched) / max(1, len(user_ingredients))
        score = w_vec * h["similarity"] + (1 - w_vec) * coverage
        out.append({**h, "coverage": round(coverage, 2), "missing": missing, "score": round(score, 3)})
    return sorted(out, key=lambda x: x["score"], reverse=True)