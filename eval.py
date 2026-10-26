"""Automatic retrieval evaluation: no manual labels, no LLM calls, free to run.

Idea ("hold-out recipe recall"): pick a recipe from the DB, pretend 4 of its main ingredients
are what's in the fridge, and check whether the system brings that recipe back.

  python eval.py --n 200
"""
import argparse
import random

from retrieval import SEP, get_store, is_pantry, rerank, vector_search


def make_queries(n: int, n_ing: int, seed: int):
    metas = get_store().get(include=["metadatas"])["metadatas"]
    rng = random.Random(seed)
    pool = []
    for m in metas:
        ings = [i for i in m["ingredients"].split(SEP) if not is_pantry(i)]
        if len(ings) >= n_ing:
            pool.append((m, rng.sample(ings, n_ing)))
    rng.shuffle(pool)
    return pool[:n]


def run(queries, w_vec, k=10):
    hit = {1: 0, 3: 0, 5: 0}
    rr = 0.0
    for m, ings in queries:
        hits = vector_search(ings, k=k)
        if w_vec is not None:
            hits = rerank(hits, ings, w_vec)
        ids = [h["id"] for h in hits]
        if m["id"] in ids:
            rank = ids.index(m["id"]) + 1
            rr += 1 / rank
            for c in hit:
                if rank <= c:
                    hit[c] += 1
    n = len(queries)
    return {f"Hit@{c}": v / n for c, v in hit.items()} | {"MRR": rr / n}


def diet_violations(queries, use_filter: bool):
    veg = [(m, i) for m, i in queries if m["diet"] in ("vegetarian", "vegan")]
    bad = total = 0
    for _, ings in veg:
        hits = vector_search(ings, diet="vegetarian" if use_filter else "none", k=3)
        total += len(hits)
        bad += sum(h["diet"] == "non-vegetarian" for h in hits)
    return bad / max(1, total), len(veg)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--ingredients", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    queries = make_queries(a.n, a.ingredients, a.seed)
    print(f"Running {len(queries)} auto-generated fridge queries...\n")

    configs = {"Vector search only": None, "Vector + rerank (w=0.6)": 0.6, "Vector + rerank (w=0.3)": 0.3}
    lines = ["| Config | Hit@1 | Hit@3 | Hit@5 | MRR |", "|---|---|---|---|---|"]
    for name, w in configs.items():
        r = run(queries, w)
        lines.append(f"| {name} | {r['Hit@1']:.2f} | {r['Hit@3']:.2f} | {r['Hit@5']:.2f} | {r['MRR']:.2f} |")

    off, nveg = diet_violations(queries, use_filter=False)
    on, _ = diet_violations(queries, use_filter=True)
    lines += ["", f"Diet safety on {nveg} vegetarian queries (share of top-3 results that were non-vegetarian):",
              f"- Without metadata filter: {off:.1%}", f"- With metadata filter: {on:.1%}"]

    report = "\n".join(lines)
    print(report)
    open("eval_results.md", "w").write(report + "\n")
    print("\nSaved to eval_results.md")