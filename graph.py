"""Fridge Chef: LangChain tools + a 3-node LangGraph with one retry loop.

understand -> search -> (no good match? relax filters, search again) -> answer -> END
"""
import os
from typing import List, Optional, TypedDict

from langchain_anthropic import ChatAnthropic
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

# ---------- Models & stores ----------
llm = ChatAnthropic(model=os.getenv("MODEL", "claude-sonnet-5-5"), temperature=0)

store = Chroma(
    collection_name="recipes",
    embedding_function=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    persist_directory="chroma_db",
)

MAX_DISTANCE = 1.1  # lower = closer match. Tune this on your data.
MAX_SEARCH_ATTEMPTS = 3


# ---------- Structured output (LangChain) ----------
class Request(BaseModel):
    ingredients: List[str] = Field(description="Ingredients the user has")
    diet: Optional[str] = Field(None, description="e.g. vegetarian, vegan, none")
    max_minutes: Optional[int] = Field(None, description="Time limit in minutes")


# ---------- Tools ----------
@tool
def search_recipes(ingredients: str, k: int = 4) -> list:
    """Search the recipe database by a comma-separated ingredient list."""
    results = store.similarity_search_with_score(f"Ingredients: {ingredients}", k=k)
    return [
        {"title": d.metadata["title"], "ingredients": d.metadata["ingredients"],
         "directions": d.metadata["directions"], "distance": round(score, 3)}
        for d, score in results
        if score <= MAX_DISTANCE
    ]


SUBSTITUTES = {
    "butter": "olive oil or ghee",
    "egg": "mashed banana or flax egg",
    "milk": "oat milk or water + a little cream",
    "paneer": "firm tofu",
    "cream": "cashew paste or yogurt",
}


@tool
def substitute_ingredient(missing: str, diet: str = "none") -> str:
    """Suggest a substitute for a missing ingredient, respecting the diet."""
    key = missing.lower().strip()
    if key in SUBSTITUTES:
        return SUBSTITUTES[key]
    # Fallback: ask the LLM
    reply = llm.invoke(f"One short substitute for '{missing}' in a {diet} recipe. Answer in 6 words max.")
    return _text(reply)


NUTRITION = {  # per 100 g: (kcal, protein g). Swap for a real API (USDA FoodData) later.
    "egg": (143, 13), "spinach": (23, 3), "rice": (130, 3), "paneer": (265, 18),
    "tofu": (76, 8), "tomato": (18, 1), "onion": (40, 1), "potato": (77, 2),
}


@tool
def get_nutrition(ingredient: str) -> str:
    """Get approximate calories and protein per 100 g for an ingredient."""
    hit = NUTRITION.get(ingredient.lower().strip())
    return f"{hit[0]} kcal, {hit[1]} g protein per 100 g" if hit else "unknown"


# ---------- State ----------
class State(TypedDict):
    user_input: str
    request: Request
    recipes: list
    attempts: int
    answer: str
    tools_used: List[str]


def _text(msg) -> str:
    """Anthropic can return a list of content blocks; flatten to plain text."""
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") for b in c if isinstance(b, dict))


# ---------- Nodes ----------
def understand(state: State) -> dict:
    req = llm.with_structured_output(Request).invoke(
        f"Extract ingredients, diet and time limit from: {state['user_input']}"
    )
    return {"request": req, "attempts": 0, "tools_used": []}


def search(state: State) -> dict:
    req, attempt = state["request"], state["attempts"]
    # Each retry relaxes the query: full list -> first 3 ingredients -> first 2
    ings = req.ingredients[: max(2, len(req.ingredients) - attempt * 2)]
    recipes = search_recipes.invoke({"ingredients": ", ".join(ings)})
    return {"recipes": recipes, "attempts": attempt + 1,
            "tools_used": state["tools_used"] + ["search_recipes"]}


def route_after_search(state: State) -> str:
    if not state["recipes"] and state["attempts"] < MAX_SEARCH_ATTEMPTS:
        return "search"  # the loop: relax filters and try again
    return "answer"


def answer(state: State) -> dict:
    """Tool-calling loop: the LLM decides if it needs substitution/nutrition tools."""
    req = state["request"]
    tools = [substitute_ingredient, get_nutrition]
    by_name = {t.name: t for t in tools}
    agent = llm.bind_tools(tools)

    context = "\n\n".join(
        f"{r['title']}\nIngredients: {r['ingredients']}\nSteps: {r['directions']}"
        for r in state["recipes"]
    ) or "NO MATCHING RECIPES FOUND."

    msgs = [
        SystemMessage(content=(
            "You are Fridge Chef. Pick the best recipe from the CONTEXT only. "
            "If an ingredient is missing, call substitute_ingredient. "
            "Call get_nutrition for the main ingredients. "
            "Respect the diet and time limit. If there is no matching recipe, say so honestly. "
            "Format: title, ingredients, numbered steps, short nutrition note."
        )),
        HumanMessage(content=(
            f"I have: {', '.join(req.ingredients)}. Diet: {req.diet}. "
            f"Time limit: {req.max_minutes} min.\n\nCONTEXT:\n{context}"
        )),
    ]

    used = list(state["tools_used"])
    resp = None
    for _ in range(5):  # hard cap on tool-loop iterations (guardrail)
        resp = agent.invoke(msgs)
        msgs.append(resp)
        if not resp.tool_calls:
            break
        for call in resp.tool_calls:
            out = by_name[call["name"]].invoke(call["args"])
            used.append(call["name"])
            msgs.append(ToolMessage(content=str(out), tool_call_id=call["id"]))

    return {"answer": _text(resp), "tools_used": used}


# ---------- Graph ----------
builder = StateGraph(State)
builder.add_node("understand", understand)
builder.add_node("search", search)
builder.add_node("answer", answer)

builder.add_edge(START, "understand")
builder.add_edge("understand", "search")
builder.add_conditional_edges("search", route_after_search, {"search": "search", "answer": "answer"})
builder.add_edge("answer", END)

graph = builder.compile()


if __name__ == "__main__":
    out = graph.invoke({"user_input": "eggs, spinach, rice, paneer. vegetarian, 20 mins"})
    print(out["answer"])
    print("\nTools used:", out["tools_used"])