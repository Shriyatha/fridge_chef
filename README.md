# Fridge Chef

Agentic recipe assistant: LangChain + LangGraph + RAG (ChromaDB) + tool calling, with a Streamlit UI.

## Run
```
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key
# put a recipe CSV (title, ingredients, directions) at ./recipes.csv
python ingest.py
streamlit run app.py
```

## How it works
understand (structured output) -> search (RAG tool, retries with relaxed query) -> answer (LLM decides to call substitute_ingredient / get_nutrition)

## TODO (to make it resume-ready)
- [ ] Add 20 test fridges + check the recipe fits the diet; put results in a table here
- [ ] Add a GIF of the UI
- [ ] Note failure cases (e.g. no match found) and how the retry loop handles them