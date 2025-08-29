# src/rag.py
from __future__ import annotations
import os, hashlib, typing as T
import chromadb
from sentence_transformers import SentenceTransformer
import pandas as pd
from chromadb.config import Settings

def _hash_id(text: str) -> str:
    # stable id to dedupe identical rows across rebuilds
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

class RAGIndex:
    """Lightweight retrieval index for nearest labeled examples using ChromaDB + SBERT."""

    def __init__(
        self,
        persist_dir: str = "rag_index",
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        space: str = "cosine",                  # "cosine" | "l2" | "ip"
    ):
        os.makedirs(persist_dir, exist_ok=True)
        self.space = space
        self.client = chromadb.PersistentClient(
        path=persist_dir,
        settings=Settings(anonymized_telemetry=False)
)
        #self.client = chromadb.PersistentClient(path=persist_dir)
        # ensure cosine space (so dist = 1 - cos_sim when embeddings normalized)
        self.collection = self.client.get_or_create_collection(
            name="prompts",
            metadata={"hnsw:space": self.space}
        )
        self.embedder = SentenceTransformer(model_name)

    # ---------------- Build / Add ----------------

    def build_from_csv(
        self,
        csv_path: str,
        text_col: str = "text",
        label_col: str = "label",
        batch_size: int = 256,
        clear_existing: bool = False,
        dropna: bool = True,
        limit: int | None = None,
    ):
        """Build index from CSV of (text, label)."""
        df = pd.read_csv(csv_path)
        cols = [text_col, label_col]
        if dropna:
            df = df[cols].dropna()
        else:
            df = df[cols]
        if limit:
            df = df.iloc[:limit]

        if clear_existing:
            try:
                self.client.delete_collection("prompts")
            except Exception:
                pass
            self.collection = self.client.create_collection(
                name="prompts",
                metadata={"hnsw:space": self.space}
            )

        texts = df[text_col].astype(str).tolist()
        labels = [int(x) for x in df[label_col].tolist()]
        self.add_texts(texts, labels, batch_size=batch_size)

    def add_texts(
        self,
        texts: list[str],
        labels: list[int] | None = None,
        batch_size: int = 256,
    ):
        if labels is None:
            labels = [0] * len(texts)

        # Prepare IDs and embeddings in chunks
        for i in range(0, len(texts), batch_size):
            chunk_docs = [str(t) for t in texts[i:i+batch_size]]
            chunk_labels = [int(l) for l in labels[i:i+batch_size]]
            ids = [f"id_{_hash_id(t)}" for t in chunk_docs]

            embs = self.embedder.encode(
                chunk_docs, batch_size=128, normalize_embeddings=True
            ).tolist()

            metas = [{"label": l} for l in chunk_labels]
            self.collection.upsert(
                ids=ids, embeddings=embs, documents=chunk_docs, metadatas=metas
            )

    # ---------------- Query ----------------

    def query(self, text: str, k: int = 5) -> T.List[dict]:
        """Return top-k neighbors with label and similarity (cosine if configured)."""
        emb = self.embedder.encode([text], normalize_embeddings=True).tolist()
        res = self.collection.query(
            query_embeddings=emb, n_results=max(1, k),
            include=["documents", "metadatas", "distances"]
        )
        docs = res.get("documents") or [[]]
        metas = res.get("metadatas") or [[]]
        dists = res.get("distances") or [[]]
        if not docs or not docs[0]:
            return []

        out: list[dict] = []
        for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
            # If space == "cosine", distance = 1 - cosine_similarity
            if self.space == "cosine":
                sim = 1.0 - float(dist)
            else:
                # fallback monotonic transform; still usable for weighting
                sim = 1.0 / (1.0 + float(dist))
            out.append({
                "text": doc,
                "label": int(meta.get("label", 0)),
                "distance": float(dist),
                "similarity": float(sim),
                "meta": dict(meta),
            })
        return out

    def vote(self, text: str, k: int = 5) -> tuple[float | None, list[dict]]:
        """
        Similarity-weighted unsafe probability from top-k neighbors.
        Returns (rag_vote, neighbors). rag_vote=None if no neighbors/weight.
        """
        sims = self.query(text, k=k)
        if not sims:
            return None, []
        wsum = sum(max(0.0, s.get("similarity", 0.0)) for s in sims)
        if wsum <= 0:
            return None, sims
        unsafe_sum = sum(
            s["similarity"] * (1.0 if int(s.get("label", 0)) == 1 else 0.0)
            for s in sims
        )
        return unsafe_sum / wsum, sims
