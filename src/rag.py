# src/rag.py
from __future__ import annotations  # enable | type hints on older Python
import os, typing as T  # filesystem, typing alias
import chromadb  # vector DB (persistent)
from sentence_transformers import SentenceTransformer  # embedding model
import pandas as pd  # CSV loading

class RAGIndex:
    """Lightweight retrieval index for nearest labeled examples."""

    def __init__(
        self,
        persist_dir: str = "rag_index",                          # directory to persist DB
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"  # default SBERT encoder
    ):
        os.makedirs(persist_dir, exist_ok=True)                 # ensure dir
        self.client = chromadb.PersistentClient(path=persist_dir)  # persistent client
        self.collection = self.client.get_or_create_collection("prompts")  # named collection
        self.embedder = SentenceTransformer(model_name)         # load encoder

    def build_from_csv(
        self,
        csv_path: str,         # dataset CSV path
        text_col="text",       # text column name
        label_col="label",     # label column name
        batch_size: int = 256  # batch size for upsert
    ):
        """Build the index from a CSV of (text,label)."""
        df = pd.read_csv(csv_path)                               # load data
        docs = df[text_col].astype(str).tolist()                 # list of texts
        labels = df[label_col].astype(int).tolist()              # labels as ints
        ids = [f"id_{i}" for i in range(len(docs))]              # unique ids

        # Upsert in chunks to control memory usage
        for i in range(0, len(docs), batch_size):
            chunk_docs = docs[i:i+batch_size]                    # chunk texts
            chunk_labels = labels[i:i+batch_size]                # chunk labels
            chunk_ids = ids[i:i+batch_size]                      # chunk ids
            embs = self.embedder.encode(
                chunk_docs, batch_size=128, normalize_embeddings=True  # vectorize (normalized)
            ).tolist()
            metas = [{"label": int(l)} for l in chunk_labels]     # metadata with labels
            self.collection.upsert(
                ids=chunk_ids, embeddings=embs, documents=chunk_docs, metadatas=metas  # write to DB
            )

    def query(self, text: str, k: int = 5) -> T.List[dict]:
        """Return top-k nearest examples with labels and crude similarity."""
        emb = self.embedder.encode([text], normalize_embeddings=True).tolist()  # encode query
        res = self.collection.query(
            query_embeddings=emb, n_results=k, include=["documents","metadatas","distances"]  # fetch
        )
        out = []  # formatted results
        for i in range(len(res["documents"][0])):  # iterate over hits
            out.append({
                "text": res["documents"][0][i],                         # neighbor text
                "label": int(res["metadatas"][0][i]["label"]),          # neighbor label
                "distance": float(res["distances"][0][i]),              # distance (Chroma)
                "similarity": float(1.0 / (1.0 + res["distances"][0][i]))  # crude sim transform
            })
        return out  # list of dicts
