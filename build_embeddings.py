"""Run ONCE on your laptop (and again whenever you edit data/). Creates embeddings.npy + embeddings.json."""
import json
import sys

import numpy as np
from sentence_transformers import SentenceTransformer

from app import EMB_PATH, EMBED_MODEL, META_PATH, corpus_hash, load_chunks

chunks, docs = load_chunks()
if not chunks:
    sys.exit("No documents found in data/. Add your .md/.txt/.pdf files first.")
if any("[[FILL" in c["text"] for c in chunks) and "--force" not in sys.argv:
    sys.exit("Placeholders [[FILL ...]] are still in data/. Replace them with real text (or use --force to test).")

print(f"{len(chunks)} chunks from {docs} documents. Embedding with {EMBED_MODEL} ...")
model = SentenceTransformer(EMBED_MODEL, device="cpu")
emb = model.encode([c["text"] for c in chunks], normalize_embeddings=True, show_progress_bar=True)
np.save(EMB_PATH, emb.astype("float32"))
META_PATH.write_text(json.dumps({"hash": corpus_hash(chunks), "model": EMBED_MODEL, "chunks": len(chunks)}))
print(f"Saved {EMB_PATH.name} {emb.shape} and {META_PATH.name}. Commit both files to GitHub.")
