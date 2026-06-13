"""Simple RAG over past projects — each agent retrieves similar prior cases."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from .config import EMBED_MODEL


class ProjectRetriever:
    """Builds one FAISS index per agent view, keyed by composed text."""

    def __init__(self, df: pd.DataFrame, cols_by_agent: dict[str, list[str]]):
        self.df = df.reset_index(drop=True)
        self.encoder = SentenceTransformer(EMBED_MODEL)
        self.indices: dict[str, faiss.IndexFlatIP] = {}
        self.texts: dict[str, list[str]] = {}
        for agent, cols in cols_by_agent.items():
            texts = self.df[cols].fillna("").astype(str).agg(" | ".join, axis=1).tolist()
            emb = self.encoder.encode(texts, normalize_embeddings=True, batch_size=32,
                                      show_progress_bar=False)
            idx = faiss.IndexFlatIP(emb.shape[1])
            idx.add(np.asarray(emb, dtype="float32"))
            self.indices[agent] = idx
            self.texts[agent] = texts

    def search(self, agent: str, query: str, k: int = 3, exclude_idx: int | None = None) -> list[str]:
        q = self.encoder.encode([query], normalize_embeddings=True)
        scores, ids = self.indices[agent].search(np.asarray(q, dtype="float32"), k + 1)
        out = []
        for i in ids[0]:
            if i == exclude_idx:
                continue
            out.append(self.texts[agent][i])
            if len(out) >= k:
                break
        return out
