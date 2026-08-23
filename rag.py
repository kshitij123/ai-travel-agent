"""
RAG — retrieval over local travel knowledge documents.

Pipeline:
  1. Load .md / .txt files from data/documents/
  2. Split into chunks (by markdown sections)
  3. Index with TF-IDF vectors (pure Python, no extra deps)
  4. search(query) returns top-k matching chunks

Used by the search_travel_docs tool when users ask factual travel
questions not covered by the flight/hotel database (e.g. JR Pass prices).
"""

import math
import re
from dataclasses import dataclass
from pathlib import Path

from config import DOCUMENTS_DIR, RAG_MIN_SCORE, RAG_TOP_K


@dataclass(frozen=True)
class DocumentChunk:
    source: str
    title: str
    text: str


@dataclass
class SearchResult:
    source: str
    title: str
    text: str
    score: float

    def to_dict(self) -> dict[str, str | float]:
        return {
            "source": self.source,
            "title": self.title,
            "text": self.text,
            "score": round(self.score, 4),
        }


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    words = _WORD_RE.findall(text.lower())
    tokens: list[str] = []
    for word in words:
        tokens.append(word)
        if word.endswith("s") and len(word) > 3:
            tokens.append(word[:-1])
    return tokens


def _term_freq(tokens: list[str]) -> dict[str, float]:
    counts: dict[str, int] = {}
    for token in tokens:
        counts[token] = counts.get(token, 0) + 1
    total = len(tokens) or 1
    return {token: count / total for token, count in counts.items()}


def _cosine_similarity(a: dict[str, float], b: dict[str, float]) -> float:
    if not a or not b:
        return 0.0
    shared = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in shared)
    norm_a = math.sqrt(sum(v * v for v in a.values()))
    norm_b = math.sqrt(sum(v * v for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _split_markdown(path: Path, content: str) -> list[DocumentChunk]:
    """Split a markdown file into chunks by ## sections."""
    chunks: list[DocumentChunk] = []
    sections = re.split(r"(?=^## )", content, flags=re.MULTILINE)
    for section in sections:
        section = section.strip()
        if not section:
            continue
        lines = section.splitlines()
        title = lines[0].lstrip("#").strip() if lines else path.stem
        chunks.append(DocumentChunk(source=path.name, title=title, text=section))
    return chunks


class TravelDocIndex:
    def __init__(self, documents_dir: Path) -> None:
        self._documents_dir = documents_dir
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[dict[str, float]] = []
        self._idf: dict[str, float] = {}
        self._build()

    def _build(self) -> None:
        self._chunks = []
        if not self._documents_dir.exists():
            return

        for path in sorted(self._documents_dir.glob("*")):
            if path.suffix.lower() not in {".md", ".txt"}:
                continue
            content = path.read_text(encoding="utf-8")
            self._chunks.extend(_split_markdown(path, content))

        doc_freq: dict[str, int] = {}
        tokenized = [_tokenize(chunk.text) for chunk in self._chunks]
        for tokens in tokenized:
            for token in set(tokens):
                doc_freq[token] = doc_freq.get(token, 0) + 1

        n_docs = len(self._chunks) or 1
        self._idf = {token: math.log((1 + n_docs) / (1 + freq)) + 1 for token, freq in doc_freq.items()}
        self._vectors = [self._tfidf(tokens) for tokens in tokenized]

    def _tfidf(self, tokens: list[str]) -> dict[str, float]:
        tf = _term_freq(tokens)
        return {token: weight * self._idf.get(token, 0.0) for token, weight in tf.items()}

    def search(self, query: str, top_k: int = RAG_TOP_K) -> list[SearchResult]:
        if not self._chunks:
            return []

        query_vec = self._tfidf(_tokenize(query))
        scored = [
            SearchResult(
                source=chunk.source,
                title=chunk.title,
                text=chunk.text,
                score=_cosine_similarity(query_vec, vec),
            )
            for chunk, vec in zip(self._chunks, self._vectors, strict=True)
        ]
        scored.sort(key=lambda r: r.score, reverse=True)
        return [r for r in scored[:top_k] if r.score >= RAG_MIN_SCORE]

    @property
    def document_count(self) -> int:
        return len({c.source for c in self._chunks})

    @property
    def chunk_count(self) -> int:
        return len(self._chunks)


_index: TravelDocIndex | None = None


def get_index() -> TravelDocIndex:
    global _index
    if _index is None:
        _index = TravelDocIndex(DOCUMENTS_DIR)
    return _index


def search_travel_docs(query: str) -> dict[str, object]:
    """Search indexed travel documents for information relevant to the query."""
    index = get_index()
    results = index.search(query)
    return {
        "query": query,
        "documents_indexed": index.document_count,
        "chunks_indexed": index.chunk_count,
        "results": [r.to_dict() for r in results],
    }
