import streamlit as st

from graph import graph

st.set_page_config(page_title="Fridge Chef", page_icon="🍳", layout="wide")
st.title("🍳 Fridge Chef")
st.caption("Tell me what's in your fridge. I'll find a recipe and fill the gaps.")

col_in, col_out = st.columns([1, 2])

with col_in:
    fridge = st.text_area("What's in your fridge?", "eggs, spinach, rice, paneer", height=120)
    diet = st.selectbox("Diet", ["none", "vegetarian", "vegan", "gluten-free"])
    minutes = st.slider("Time (minutes)", 10, 90, 20, step=5)
    go = st.button("Cook something!", type="primary", use_container_width=True)

if go and fridge.strip():
    with st.spinner("Chef is thinking..."):
        result = graph.invoke(
            {"user_input": f"{fridge}. Diet: {diet}. Time limit: {minutes} minutes."}
        )
    with col_out:
        st.markdown(result["answer"])
    with st.sidebar:
        st.header("Agent trace")
        st.write("**Ingredients parsed:**", ", ".join(result["request"].ingredients))
        st.write("**Search attempts:**", result["attempts"])
        st.write("**Tools called:**")
        for t in result["tools_used"]:
            st.code(t)
        with st.expander("Retrieved recipes"):
            for r in result["recipes"]:
                st.write(f"{r['title']} (distance {r['distance']})")