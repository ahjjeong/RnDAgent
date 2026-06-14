"""Simple RAG over past projects — each agent retrieves similar prior cases."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
from .config import (
    EMBED_MODEL,
    ID_COL,
    RAG_BATCH_SIZE,
    RAG_CACHE_DIR,
    RAG_CACHE_ENABLED,
    RAG_CANDIDATE_POOL,
    RAG_DEVICE,
    RAG_SHOW_PROGRESS,
)


def _to_int_year(value) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    try:
        return int(float(value))
    except (TypeError, ValueError):
        text = str(value).strip()
        return int(text[:4]) if len(text) >= 4 and text[:4].isdigit() else None


class ProjectRetriever:
    """Builds one FAISS index per agent view, keyed by composed text."""

    def __init__(self, df: pd.DataFrame, cols_by_agent: dict[str, list[str]]):
        self.df = df.reset_index(drop=True)
        self.encoder = SentenceTransformer(EMBED_MODEL, device=RAG_DEVICE)
        self.indices: dict[str, faiss.IndexFlatIP] = {}
        self.texts: dict[str, list[str]] = {}
        self.row_ids = self.df.get("__dataset_row_id", pd.Series(range(len(self.df)))).tolist()
        self.project_ids = self.df.get(ID_COL, pd.Series([None] * len(self.df))).astype(str).tolist()
        self.years = [
            _to_int_year(row.get("과제수행연도"))
            or _to_int_year(row.get("제출년도"))
            or _to_int_year(row.get("종료연도"))
            for _, row in self.df.iterrows()
        ]
        for agent, cols in cols_by_agent.items():
            existing = [c for c in cols if c in self.df.columns]
            if not existing:
                continue
            texts = self.df[existing].fillna("").astype(str).agg(" | ".join, axis=1).tolist()
            idx = self._load_index(agent, existing, texts)
            if idx is None:
                emb = self.encoder.encode(
                    texts,
                    normalize_embeddings=True,
                    batch_size=RAG_BATCH_SIZE,
                    show_progress_bar=RAG_SHOW_PROGRESS,
                )
                idx = faiss.IndexFlatIP(emb.shape[1])
                idx.add(np.asarray(emb, dtype="float32"))
                self._save_index(agent, existing, texts, idx)
            self.indices[agent] = idx
            self.texts[agent] = texts

    def _cache_path(self, agent: str, cols: list[str], texts: list[str]):
        if not RAG_CACHE_ENABLED:
            return None
        digest = hashlib.sha256()
        digest.update(EMBED_MODEL.encode("utf-8"))
        digest.update(str(len(texts)).encode("utf-8"))
        digest.update("\0".join(cols).encode("utf-8"))
        for text in texts:
            digest.update(b"\0")
            digest.update(text.encode("utf-8", errors="ignore"))
        safe_agent = "".join(ch if ch.isalnum() else "_" for ch in agent)
        RAG_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        return RAG_CACHE_DIR / f"{safe_agent}_{digest.hexdigest()[:24]}.faiss"

    def _load_index(self, agent: str, cols: list[str], texts: list[str]):
        path = self._cache_path(agent, cols, texts)
        if path is None or not path.exists():
            return None
        try:
            idx = faiss.read_index(str(path))
        except Exception as exc:
            print(f"[RAG] Cache load failed, rebuilding: {path} ({exc})")
            return None
        if idx.ntotal != len(texts):
            print(f"[RAG] Cache row-count mismatch, rebuilding: {path}")
            return None
        print(f"[RAG] Loaded cached index: {path}")
        return idx

    def _save_index(self, agent: str, cols: list[str], texts: list[str], idx):
        path = self._cache_path(agent, cols, texts)
        if path is None:
            return
        try:
            faiss.write_index(idx, str(path))
            print(f"[RAG] Saved index cache: {path}")
        except Exception as exc:
            print(f"[RAG] Cache save skipped: {path} ({exc})")

    def search(
        self,
        agent: str,
        query: str,
        k: int = 3,
        exclude_idx: int | None = None,
        max_year: int | None = None,
        exclude_project_id: str | None = None,
    ) -> list[str]:
        if agent not in self.indices:
            if "공통" in self.indices:
                agent = "공통"
            elif self.indices:
                agent = next(iter(self.indices))
            else:
                return []
        q = self.encoder.encode([query], normalize_embeddings=True)
        candidate_k = min(self.indices[agent].ntotal, max(k + 1, RAG_CANDIDATE_POOL))
        scores, ids = self.indices[agent].search(np.asarray(q, dtype="float32"), candidate_k)
        out = []
        for i in ids[0]:
            if i < 0:
                continue
            if i == exclude_idx:
                continue
            if exclude_idx is not None and self.row_ids[i] == exclude_idx:
                continue
            if exclude_project_id is not None and self.project_ids[i] == str(exclude_project_id):
                continue
            if max_year is not None and self.years[i] is not None and self.years[i] > max_year:
                continue
            out.append(self.texts[agent][i])
            if len(out) >= k:
                break
        return out
