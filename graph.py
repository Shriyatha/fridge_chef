"""Fridge Chef agent (LangGraph, 5 nodes, 2 loops).

understand -> retrieve -> rerank --(weak match? relax query, retry)--> answer -> verify --(unsupported? retry once)--> END
"""
from typing import List, Literal, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from llm import get_llm
from retrieval import rerank, vector_search

llm = get_llm()

MIN_SIM = 0.15          # drop hits below this cosine similarity
MIN_COVERAGE = 0.4      # top recipe should use >= 40% of the user's ingredients
MAX_RETRIEVALS = 3
MAX_TOOL_STEPS = 5


# ---------- Structured outputs (LangChain + Pydantic) ----------
class Request(BaseModel):
    ingredients: List[str] = Field(description="Ingredients the user HAS: lowercase, singular nouns")
    diet: Literal["vegetarian", "vegan", "non-vegetarian", "none"] = "none"


class Verdict(BaseModel):
    supported: bool = Field(description="True if every ingredient/step is grounded in the context and the diet is respected")
    issues: str = Field(description="What is unsupported or wrong; empty if none")


# ---------- Tools ----------
@tool
def search_recipes(ingredients: List[str], diet: str = "none", k: int = 8) -> list:
    """Semantic vector search over the recipe database with an optional diet filter."""
    return vector_search(ingredients, diet=diet, k=k, min_sim=MIN_SIM)


SUBSTITUTES = {
    "butter": "olive oil or ghee", "egg": "mashed banana or flax egg", "eggs": "mashed banana or flax egg",
    "milk": "oat milk", "paneer": "firm tofu", "cream": "cashew paste or yogurt",
    "yogurt": "coconut yogurt", "sour cream": "plain yogurt", "parmesan": "nutritional yeast",
    "honey": "maple syrup", "buttermilk": "milk + a squeeze of lemon",
}


@tool
def substitute_ingredient(missing: str, diet: str = "none") -> str:
    """Suggest a substitute for a missing ingredient that respects the diet."""
    key = missing.lower().strip()
    if key in SUBSTITUTES:
        return SUBSTITUTES[key]
    return _text(llm.invoke(f"Give one short substitute for '{missing}' in a {diet} recipe. Max 6 words."))


NUTRITION = {  # approx per 100 g: (kcal, protein g)
    "egg": (143, 13), "eggs": (143, 13), "spinach": (23, 3), "rice": (130, 3), "paneer": (265, 18),
    "tofu": (76, 8), "tomato": (18, 1), "onion": (40, 1), "potato": (77, 2), "chicken": (165, 31),
    "lentils": (116, 9), "chickpeas": (164, 9), "cheese": (402, 25), "pasta": (131, 5), "beef": (250, 26),
}


@tool
def get_nutrition(ingredient: str) -> str:
    """Approximate calories and protein per 100 g for an ingredient."""
    hit = NUTRITION.get(ingredient.lower().strip())
    return f"{hit[0]} kcal, {hit[1]} g protein per 100 g" if hit else "unknown"


# ---------- State ----------
class State(TypedDict, total=False):
    user_input: str
    request: Request
    hits: list
    ranked: list
    attempts: int
    answer: str
    verdict: Verdict
    feedback: str
    verify_attempts: int
    tools_used: List[str]
    trace: List[str]


def _text(msg) -> str:
    c = msg.content
    return c if isinstance(c, str) else "".join(b.get("text", "") for b in c if isinstance(b, dict))


def _context(ranked: list) -> str:
    if not ranked:
        return "NO MATCHING RECIPES FOUND."
    return "\n\n".join(
        f"[{i}] {r['title']} ({r['area']}, {r['category']}, {r['diet']})\n"
        f"Ingredients: {', '.join(r['ingredients_detailed'].split('|'))}\n"
        f"Missing from the user's fridge: {', '.join(r['missing']) or 'nothing'}\n"
        f"Instructions: {r['instructions'][:1500]}"
        for i, r in enumerate(ranked, 1)
    )


# ---------- Nodes ----------
def understand(state: State) -> dict:
    req = llm.with_structured_output(Request).invoke(
        f"Extract the ingredients the user has and their diet from: {state['user_input']}"
    )
    return {"request": req, "attempts": 0, "verify_attempts": 0, "feedback": "", "tools_used": [],
            "trace": [f"understand -> ingredients={req.ingredients}, diet={req.diet}"]}


def retrieve(state: State) -> dict:
    req, n = state["request"], state["attempts"]
    ings = req.ingredients[: max(2, len(req.ingredients) - n)]  # each retry relaxes the query
    hits = search_recipes.invoke({"ingredients": ings, "diet": req.diet, "k": 8 + 4 * n})
    return {"hits": hits, "attempts": n + 1, "tools_used": state["tools_used"] + ["search_recipes"],
            "trace": state["trace"] + [f"retrieve (try {n + 1}): query with {len(ings)} ingredients, k={8 + 4 * n} -> {len(hits)} hits"]}


def rerank_node(state: State) -> dict:
    ranked = rerank(state["hits"], state["request"].ingredients)[:3]
    best = ranked[0] if ranked else None
    msg = f"rerank -> best: {best['title']} (coverage {best['coverage']}, score {best['score']})" if best else "rerank -> nothing found"
    return {"ranked": ranked, "trace": state["trace"] + [msg]}


def route_after_rerank(state: State) -> str:
    ranked = state["ranked"]
    weak = not ranked or ranked[0]["coverage"] < MIN_COVERAGE
    return "retrieve" if weak and state["attempts"] < MAX_RETRIEVALS else "answer"


def answer(state: State) -> dict:
    req = state["request"]
    tools = [substitute_ingredient, get_nutrition]
    by_name = {t.name: t for t in tools}
    agent = llm.bind_tools(tools)

    system = (
        "You are Fridge Chef. Choose the single best recipe from the CONTEXT and use ONLY information in it. "
        "For each important missing ingredient, call substitute_ingredient. "
        "Call get_nutrition for 1-2 main ingredients. Respect the diet. "
        "If there is no matching recipe, say so honestly. "
        "Format: title, why it fits, ingredients (mark substitutions), numbered steps, short nutrition note."
    )
    if state.get("feedback"):
        system += f"\nYour previous answer was rejected for: {state['feedback']}. Fix this."

    msgs = [SystemMessage(content=system),
            HumanMessage(content=f"I have: {', '.join(req.ingredients)}. Diet: {req.diet}.\n\nCONTEXT:\n{_context(state['ranked'])}")]

    used, resp = list(state["tools_used"]), None
    for _ in range(MAX_TOOL_STEPS):  # guardrail: capped tool loop
        resp = agent.invoke(msgs)
        msgs.append(resp)
        if not resp.tool_calls:
            break
        for call in resp.tool_calls:
            out = by_name[call["name"]].invoke(call["args"])
            used.append(call["name"])
            msgs.append(ToolMessage(content=str(out), tool_call_id=call["id"]))
    return {"answer": _text(resp), "tools_used": used,
            "trace": state["trace"] + [f"answer -> tool calls so far: {used}"]}


def verify(state: State) -> dict:
    """Self-check: is the answer grounded in the retrieved context and the diet respected?"""
    verdict = llm.with_structured_output(Verdict).invoke(
        f"CONTEXT:\n{_context(state['ranked'])}\n\nANSWER:\n{state['answer']}\n\n"
        f"Diet required: {state['request'].diet}. Is every ingredient and step in the ANSWER supported by the CONTEXT "
        "(ingredient substitutions that are clearly labelled as substitutions are allowed)?"
    )
    n = state["verify_attempts"] + 1
    return {"verdict": verdict, "verify_attempts": n, "feedback": verdict.issues,
            "trace": state["trace"] + [f"verify (check {n}) -> {'PASS' if verdict.supported else 'FAIL: ' + verdict.issues}"]}


def route_after_verify(state: State) -> str:
    return "answer" if (not state["verdict"].supported and state["verify_attempts"] < 2) else END


# ---------- Graph ----------
builder = StateGraph(State)
builder.add_node("understand", understand)
builder.add_node("retrieve", retrieve)
builder.add_node("rerank", rerank_node)
builder.add_node("answer", answer)
builder.add_node("verify", verify)

builder.add_edge(START, "understand")
builder.add_edge("understand", "retrieve")
builder.add_edge("retrieve", "rerank")
builder.add_conditional_edges("rerank", route_after_rerank, {"retrieve": "retrieve", "answer": "answer"})
builder.add_edge("answer", "verify")
builder.add_conditional_edges("verify", route_after_verify, {"answer": "answer", END: END})
graph = builder.compile()


if __name__ == "__main__":
    out = graph.invoke({"user_input": "eggs, spinach, rice, paneer. Diet: vegetarian."})
    print(out["answer"], "\n")
    print("\n".join(out["trace"]))