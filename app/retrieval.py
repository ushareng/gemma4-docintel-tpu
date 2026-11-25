"""BM25 retrieval over page-tagged chunks, so every answer can cite its page."""
from __future__ import annotations

import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from . import config
from .documents import Document

_STOP = set("""a an and are as at be by for from has have in is it its of on or that the
this to was were will with what which who whom how when where why do does did i you we
""".split())
_TOKEN = re.compile(r"[a-z0-9]+(?:[.,/-][a-z0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


@dataclass
class Chunk:
    page: int
    idx: int
    text: str


def chunk_document(doc: Document, words: int = config.CHUNK_WORDS,
                   overlap: int = config.CHUNK_OVERLAP) -> list[Chunk]:
    chunks: list[Chunk] = []
    step = max(1, words - overlap)
    for page in doc.pages:
        toks = page.text.split()
        if not toks:
            continue
        for start in range(0, max(1, len(toks) - overlap), step):
            piece = " ".join(toks[start:start + words])
            if piece:
                chunks.append(Chunk(page=page.number, idx=len(chunks), text=piece))
    return chunks


class BM25Index:
    def __init__(self, doc: Document):
        self.chunks = chunk_document(doc)
        corpus = [tokenize(c.text) or ["_empty_"] for c in self.chunks]
        self.bm25 = BM25Okapi(corpus) if corpus else None

    def search(self, query: str, k: int = config.TOP_K) -> list[tuple[Chunk, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        hits = [(c, float(s)) for c, s in ranked[:k] if s > 0]
        return hits or [(c, float(s)) for c, s in ranked[:k]]  # fall back to top-k
