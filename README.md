# AI Clone of Hamda Tahir (Hybrid RAG)

**This is an AI clone, not the real person.**
Live link: [[FILL: https://your-service.onrender.com]]

A chatbot that answers in first person using only documents I wrote about myself. Pipeline: chunk -> BM25 index + sentence-embedding index -> retrieve (BM25 / Dense / Hybrid) -> Reciprocal Rank Fusion -> Groq LLM with a persona prompt -> answer shown with the retrieved chunks, sources and scores.

Stack: FastAPI + a single HTML page (instead of Gradio), rank_bm25, sentence-transformers (all-MiniLM-L6-v2), numpy cosine similarity (no vector database), Groq API for generation, hosted on Render. The Groq key is stored as a Render environment variable and is never in the repo.

## Run locally
```
python -m venv .venv && .venv\Scripts\activate      # Windows  (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
python build_embeddings.py                           # creates embeddings.npy + embeddings.json
set GROQ_API_KEY=your_key                            # Mac/Linux: export GROQ_API_KEY=your_key
uvicorn app:app --reload                             # open http://127.0.0.1:8000
```

---
# One-page report

## 1. Knowledge base and why
I used 6 documents total: `01_bio.md` (background and personality), `02_cv.md` (education and skills), `03_opinions_and_hobbies.md` (personal preferences), `04_faq.md` (25 direct Q&A pairs), `05_project_writeup.md` (a detailed technical writeup of my main project), and `06_study_habits.md` (my routines). I chose this mix to provide a solid foundation of facts (the CV), descriptive passages for semantic search (hobbies), and a detailed project write-up. 

The rare, exact keyword I included is **"CampusConnect"** (the name of my project), which appears in both the CV and the project write-up. I also included descriptive passages, like "I like to play strategy games on my PC, or just watch a documentary on Netflix," so dense search could match a paraphrased question like "What do I do to unwind?".
Chunking: ~200 words per chunk with a 30-word overlap; every chunk keeps its source file name and chunk ID.

## 2. Test questions (run `python eval_report.py` and paste the table here)
| Question | BM25 result | Dense result | Hybrid result | Best mode and why |
|---|---|---|---|---|
| What was the CampusConnect project about? *(exact rare keyword)* | 05_project_writeup-0 (1.69)<br>02_cv-0 (1.10) | 05_project_writeup-0 (0.35)<br>04_faq-2 (0.28) | 05_project_writeup-0 (0.03)<br>02_cv-0 (0.03) | **BM25** won because the exact word "CampusConnect" is a unique identifier. BM25 gives it a massive score boost, whereas dense search treats it similarly to other project words. |
| What do I do to unwind after a long day? *(paraphrase)* | 04_faq-2 (4.09)<br>04_faq-3 (1.59) | 01_bio-0 (0.21)<br>06_study_habits-0 (0.20) | 01_bio-0 (0.03)<br>04_faq-2 (0.03) | **Dense** won here because the word "unwind" doesn't appear in my notes. BM25 missed the context entirely, but dense search successfully matched the semantic meaning to my bio where I talk about relaxing. |
| Where did you go to university? *(factual)* | 01_bio-0 (3.49)<br>06_study_habits-0 (1.92) | 01_bio-0 (0.38)<br>04_faq-2 (0.31) | 01_bio-0 (0.03)<br>04_faq-2 (0.03) | **Hybrid** (or both). Both BM25 and Dense easily found the exact match in the bio. Hybrid merges them perfectly so the LLM gets the right context. |
| What is your hot take on learning new frameworks? *(opinion)* | 03_opinions_and_hobbies-0 (3.99)<br>04_faq-2 (1.59) | 03_opinions_and_hobbies-0 (0.45)<br>04_faq-2 (0.45) | 03_opinions_and_hobbies-0 (0.03)<br>04_faq-2 (0.03) | **BM25** won slightly on raw score due to the exact keyword "frameworks", but both models retrieved the correct chunk perfectly. |
| What is your salary expectation? *(out of scope)* | no match | 04_faq-1 (0.17)<br>04_faq-3 (0.14) | 04_faq-1 (0.01)<br>04_faq-3 (0.01) | **BM25** correctly returned "no match" (score 0). Dense search struggled because it just forced a match on random words, showing why BM25 is better for filtering out-of-scope questions. |

## 3. Fusion method
I used Reciprocal Rank Fusion, score = sum of 1 / (60 + rank) over the BM25 and dense rankings (top 20 of each). The alternative is a weighted score alpha * dense + (1 - alpha) * bm25 after min-max normalising each score; it needs tuning alpha and is sensitive to outliers, while RRF only uses ranks, so it needs no score normalisation.

## 4. Failure case
**Question:** "What is your salary expectation?"
**What happened:** The BM25 retriever correctly returned no matches (score 0). However, the Dense retriever forced a match and pulled up `04_faq-1` (my biggest weakness question) and `04_faq-3` (music preferences). The Hybrid mode then combined these, pushing irrelevant chunks to the LLM. 
**Why it happened:** Dense embeddings always return the "closest" vector, even if the text is entirely unrelated. Because the user asked about a job/salary, it pulled FAQ chunks about jobs and weakness. The LLM had to read these irrelevant chunks and finally say "I don't have that in my notes," which is correct, but the retrieval process was messy. This shows the danger of relying purely on dense search for out-of-scope questions.

## 5. One improvement
A great improvement would be adding a **Cross-Encoder Reranker** (like `ms-marco-MiniLM-L-6-v2`). Right now, the RRF simply merges ranks. A cross-encoder would take the top 10 chunks from the hybrid retrieval and actually read the text alongside the question to score them more accurately. This would fix the failure case above, as it would easily score the irrelevant FAQ chunks as a 0.
