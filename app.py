import numpy as np
import streamlit as st

from graph import graph
from retrieval import get_embeddings, get_store, rerank, vector_search

st.set_page_config(page_title="Fridge Chef", page_icon="🍳", layout="wide")
st.title("🍳 Fridge Chef")
st.caption("An agentic RAG app: semantic search over real recipes + tool-calling + self-verification")

if get_store()._collection.count() == 0:
    st.error("The vector database is empty. Run `python ingest.py` first.")
    st.stop()

tab_chef, tab_lab, tab_how = st.tabs(["🍳 Chef", "🧪 Embeddings lab", "⚙️ How it works"])

# ---------------- Chef ----------------
with tab_chef:
    c_in, c_out = st.columns([1, 2])
    with c_in:
        fridge = st.text_area("What's in your fridge?", "eggs, spinach, rice, paneer", height=110)
        diet = st.selectbox("Diet", ["none", "vegetarian", "vegan"])
        go = st.button("Cook something!", type="primary", use_container_width=True)

    if go and fridge.strip():
        with st.spinner("Agent is working..."):
            res = graph.invoke({"user_input": f"{fridge}. Diet: {diet}."})
        with c_out:
            if res["ranked"] and res["ranked"][0]["thumb"]:
                st.image(res["ranked"][0]["thumb"], width=260)
            st.markdown(res["answer"])
            if res["verdict"].supported:
                st.success("✅ Verified: the answer is grounded in the retrieved recipes")
            else:
                st.warning(f"⚠️ Could not fully verify: {res['verdict'].issues}")

        st.subheader("Agent trace")
        for step in res["trace"]:
            st.write("•", step)
        st.write("**Tools called:**", ", ".join(res["tools_used"]))

        st.subheader("Retrieved recipes (after reranking)")
        st.dataframe([{"recipe": r["title"], "cuisine": r["area"], "diet": r["diet"],
                       "similarity": r["similarity"], "ingredient coverage": r["coverage"],
                       "final score": r["score"], "missing": ", ".join(r["missing"])} for r in res["ranked"]],
                     use_container_width=True)

# ---------------- Embeddings lab ----------------
with tab_lab:
    st.markdown("Embeddings turn text into vectors. **Similar meaning means vectors that point the same way.** Try it:")
    c1, c2 = st.columns(2)
    t1 = c1.text_input("Text A", "paneer")
    t2 = c2.text_input("Text B", "tofu")
    if t1 and t2:
        emb = get_embeddings()
        a, b = np.array(emb.embed_query(t1)), np.array(emb.embed_query(t2))
        st.metric("Cosine similarity", f"{float(a @ b):.3f}", help="1 = same meaning, 0 = unrelated")
        st.caption(f"Each text became a {len(a)}-dimensional vector. First 8 numbers of A: {np.round(a[:8], 3).tolist()}")

    st.divider()
    q = st.text_input("Semantic search (no LLM): type ingredients", "chickpeas, tomato, onion")
    if q:
        hits = vector_search([x.strip() for x in q.split(",") if x.strip()], k=5)
        for h in hits:
            st.write(f"**{h['title']}** ({h['area']}, {h['diet']})")
            st.progress(max(0.0, min(1.0, h["similarity"])), text=f"similarity {h['similarity']}")

# ---------------- How it works ----------------
with tab_how:
    st.markdown("""
**Pipeline:** TheMealDB API → clean + diet labelling → embed (MiniLM, 384-dim) → ChromaDB.

**Agent (LangGraph):** `understand` → `retrieve` → `rerank` → `answer` → `verify`
- **Loop 1:** weak match? relax the query and retrieve again (max 3 tries)
- **Loop 2:** verifier says the answer isn't grounded? regenerate once with feedback
- **Tools:** `search_recipes`, `substitute_ingredient`, `get_nutrition`
- **Guardrails:** hard diet filter in the vector search, capped loops, grounded-answer check
""")
    st.code(graph.get_graph().draw_mermaid(), language="text")