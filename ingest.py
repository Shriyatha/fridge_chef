"""Load recipes from a CSV into a local ChromaDB vector store.

Dataset: RecipeNLG or any Kaggle recipe CSV with columns
`title`, `ingredients`, `directions`. Save it as recipes.csv.
Run once:  python ingest.py
"""
import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

MAX_RECIPES = 10_000
BATCH = 500

df = pd.read_csv("recipes.csv").dropna(subset=["title", "ingredients", "directions"])
df = df.head(MAX_RECIPES)

docs = [
    Document(
        page_content=f"{r.title}. Ingredients: {r.ingredients}",  # what gets embedded
        metadata={
            "title": str(r.title),
            "ingredients": str(r.ingredients),
            "directions": str(r.directions),
        },
    )
    for r in df.itertuples()
]

store = Chroma(
    collection_name="recipes",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    persist_directory="chroma_db",
)

for i in range(0, len(docs), BATCH):
    store.add_documents(docs[i : i + BATCH])
    print(f"Indexed {min(i + BATCH, len(docs))}/{len(docs)}")