import os, glob, re, json
import numpy as np
import faiss
import litellm
from config import MODEL
from sentence_transformers import SentenceTransformer
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200
TOP_K = 4
SIM_THRESHOLD = 0.25


def read_corpus(pattern="data/*"):
   docs = []
   for path in glob.glob(pattern):
       with open(path, "r", encoding="utf-8", errors="ignore") as f:
           text = f.read().strip()
           docs.append({"path": path, "text": text})
   return docs


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
   sents = re.split(r"(?<=[.!?])\s+", text)
   chunks, buf = [], ""
   for s in sents:
       if len(buf) + len(s) + 1 <= size:
           buf = (buf + " " + s).strip()
       else:
           if buf:
               chunks.append(buf)
           start = max(0, len(buf) - overlap)
           carry = buf[start:]
           buf = (carry + " " + s).strip()
   if buf:
       chunks.append(buf)
   return chunks


def embed_texts(texts, model=EMBED_MODEL):
   embs = []
   for t in texts:
      # r = litellm.embedding(model=model, input=t)
       embeding_model = SentenceTransformer(EMBED_MODEL)
       embedding = embeding_model.encode(t,normalize_embeddings=True)
       embs.append(np.array(embedding, dtype="float32"))


   return np.vstack(embs)


def build_index(chunks):
   embs = embed_texts([c["text"] for c in chunks])
   faiss.normalize_L2(embs)
   index = faiss.IndexFlatIP(embs.shape[1])
   index.add(embs)
   return index


class RAG:
   def __init__(self):
       self.docs = None
       self.chunks = None
       self.index = None


   def ingest(self, pattern="data/*"):
       self.docs = read_corpus(pattern)
       self.chunks = []
       for d in self.docs:
           for i, ch in enumerate(chunk_text(d["text"])):
               self.chunks.append({
                   "doc": d["path"],
                   "chunk_id": i,
                   "text": ch,
               })
       self.index = build_index(self.chunks)
       return len(self.chunks)


   def retrieve(self, query, k=TOP_K):
       q_emb = embed_texts([query])
       faiss.normalize_L2(q_emb)
       D, I = self.index.search(q_emb, k)
       results = []
       for score, idx in zip(D[0].tolist(), I[0].tolist()):
           ch = self.chunks[idx]
           results.append({
               "score": float(score),
               "doc": ch["doc"],
               "chunk_id": ch["chunk_id"],
               "text": ch["text"],
           })
       return results


   def answer(self, query, k=TOP_K, threshold=SIM_THRESHOLD):
       hits = self.retrieve(query, k=k)
       filtered = [h for h in hits if h["score"] >= threshold]
       if not filtered:
           return {"answer": "I couldn't find reliable context."}
       context = "\n\n".join(
           [f"[{i+1}] {h['text']}" for i, h in enumerate(filtered)]
       )
       prompt = f"Use only the context to answer. If not found, say so.\n\nContext:\n{context}\n\nQuestion: {query}"
       r = litellm.completion(
           model=MODEL,
           messages=[{"role":"user","content": prompt}],
           max_tokens=300,
           temperature=0.2,
       )
       return {"answer": r.choices[0].message["content"], "hits": filtered}
