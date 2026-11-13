"""Generates the report table (5 questions x 3 modes) as Markdown. Edit QUESTIONS, then run: python eval_report.py"""
from app import IDX, init_index, retrieve

# (question, type note, expected source file or None). Replace with questions about YOUR documents.

QUESTIONS = [
    ("What was the CampusConnect project about?", "exact rare keyword (BM25 should win)", "05_project_writeup.md"),
    ("What do I do to unwind after a long day?", "paraphrase (dense should win)", "03_opinions_and_hobbies.md"),
    ("Where did you go to university?", "factual", "02_cv.md"),
    ("What is your hot take on learning new frameworks?", "opinion", "03_opinions_and_hobbies.md"),
    ("What is your salary expectation?", "out of scope (should say I don't know)", None),
]

def cell(hits):
    return "<br>".join(f"{h['chunk_id']} ({h['score']})" for h in hits[:3]) or "no match"


init_index()
print("| Question | BM25 result | Dense result | Hybrid result | Best mode and why |")
print("|---|---|---|---|---|")
hit_counts = {"bm25": 0, "dense": 0, "hybrid": 0}
labelled = 0
for q, note, expected in QUESTIONS:
    res = {m: retrieve(q, m, 3) for m in ("bm25", "dense", "hybrid")}
    print(f"| {q} _({note})_ | {cell(res['bm25'])} | {cell(res['dense'])} | {cell(res['hybrid'])} | *(write your verdict)* |")
    if expected:
        labelled += 1
        for m in hit_counts:
            hit_counts[m] += any(h["source"] == expected for h in res[m])
if labelled:
    print("\nhit@3:", {m: f"{c}/{labelled}" for m, c in hit_counts.items()})
