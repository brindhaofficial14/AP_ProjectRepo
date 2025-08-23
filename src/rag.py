# src/rag.py
from __future__ import annotations
import os, typing as T
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd

class RAGIndex:
    def __init__(self, persist_dir: str = "rag_index",
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection("prompts")
        self.embedder = SentenceTransformer(model_name)

    def build_from_csv(self, csv_path: str, text_col="text", label_col="label",
                       batch_size: int = 256):
        df = pd.read_csv(csv_path)
        docs = df[text_col].astype(str).tolist()
        labels = df[label_col].astype(int).tolist()
        ids = [f"id_{i}" for i in range(len(docs))]

        # chunked upsert
        for i in range(0, len(docs), batch_size):
            chunk_docs = docs[i:i+batch_size]
            chunk_labels = labels[i:i+batch_size]
            chunk_ids = ids[i:i+batch_size]
            embs = self.embedder.encode(chunk_docs, batch_size=128, normalize_embeddings=True).tolist()
            metas = [{"label": int(l)} for l in chunk_labels]
            self.collection.upsert(ids=chunk_ids, embeddings=embs, documents=chunk_docs, metadatas=metas)

    def query(self, text: str, k: int = 5) -> T.List[dict]:
        emb = self.embedder.encode([text], normalize_embeddings=True).tolist()
        res = self.collection.query(query_embeddings=emb, n_results=k,
                                    include=["documents","metadatas","distances"])
        out = []
        for i in range(len(res["documents"][0])):
            out.append({
                "text": res["documents"][0][i],
                "label": int(res["metadatas"][0][i]["label"]),
                "distance": float(res["distances"][0][i]),
                "similarity": float(1.0 / (1.0 + res["distances"][0][i]))  # crude sim
            })
        return out
