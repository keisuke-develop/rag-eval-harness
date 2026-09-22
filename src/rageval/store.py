"""ベクトルの格納と近傍検索。

差し替え可能な4点のうち「Vector DB」がここにある（ADR 0002）。
実装を特定製品に固定していないことを示すために2つ置く。

`InMemoryStore`（kind: memory）
    総当たりのコサイン類似度。近似を挟まないので順位が必ず再現する。
    実験と CI の既定。コーパスが数百チャンク規模なら速度も問題にならない。

`ChromaStore`（kind: chroma）
    Chroma の HNSW を使う。**既定では chromadb を入れていない**ので、
    使うには `uv sync --extra chroma` が要る。外してある理由は pyproject.toml を参照。

    近似最近傍なので、同じベクトルでも `InMemoryStore` と順位が一致しないことがある。
    これは実装の差ではなく近似探索の性質なので、条件を比較するときは store を混ぜないこと。
    比較の道具としては、近似のぶれが条件の差に混ざらない `memory` のほうが素直。
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from rageval.config import StoreConfig
from rageval.embed import Vector
from rageval.ingest import Chunk


@dataclass(frozen=True)
class SearchHit:
    """検索結果1件。`rank` は1始まり。"""

    chunk: Chunk
    score: float
    rank: int


class VectorStore(Protocol):
    """差し替え点：Vector DB。"""

    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None: ...

    def search(self, query: Vector, k: int) -> list[SearchHit]: ...


def _normalize_rows(matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


@dataclass
class InMemoryStore:
    """総当たりのコサイン類似度。順位の同点は chunk_id の昇順で割る。"""

    chunks: list[Chunk] = field(default_factory=list)
    _matrix: NDArray[np.float32] | None = field(default=None, init=False, repr=False)

    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"チャンク {len(chunks)} 件に対してベクトルが {len(vectors)} 件")
        self.chunks = list(chunks)
        self._matrix = _normalize_rows(np.asarray(vectors, dtype=np.float32))

    def search(self, query: Vector, k: int) -> list[SearchHit]:
        if self._matrix is None or not self.chunks:
            return []
        vector = np.asarray(query, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector = vector / norm
        scores = self._matrix @ vector

        # 同点の並びが実行ごとに変わらないよう、スコア降順・chunk_id 昇順で決め切る。
        order = sorted(
            range(len(self.chunks)),
            key=lambda i: (-float(scores[i]), self.chunks[i].chunk_id),
        )
        return [
            SearchHit(chunk=self.chunks[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(order[:k], start=1)
        ]


@dataclass
class ChromaStore:
    """Chroma に載せる実装。コレクションは実行ごとに使い捨てる。"""

    collection_name: str = "rageval"
    _collection: Any = field(default=None, init=False, repr=False)
    _by_id: dict[str, Chunk] = field(default_factory=dict, init=False, repr=False)

    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"チャンク {len(chunks)} 件に対してベクトルが {len(vectors)} 件")
        # chromadb は既定で入れていないうえ import も重いので、使うときだけ読み込む。
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError(
                "chromadb が入っていない。store.kind: chroma を使うには "
                "`uv sync --extra chroma` で入れること。"
                "既定から外している理由は pyproject.toml のコメントを参照"
            ) from exc

        client = chromadb.EphemeralClient()
        # 同名のコレクションが残っていれば捨てる。無ければ何もしなくてよい。
        with contextlib.suppress(Exception):
            client.delete_collection(self.collection_name)
        self._collection = client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
            embedding_function=None,
        )
        self._by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self._collection.add(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[list(v) for v in vectors],
        )

    def search(self, query: Vector, k: int) -> list[SearchHit]:
        if self._collection is None:
            return []
        result = self._collection.query(query_embeddings=[list(query)], n_results=k)
        ids: list[str] = result["ids"][0]
        distances: list[float] = result["distances"][0]
        # Chroma のコサイン距離は 1 - 類似度。指標の向きを他の store と揃える。
        return [
            SearchHit(chunk=self._by_id[chunk_id], score=1.0 - float(distance), rank=rank)
            for rank, (chunk_id, distance) in enumerate(zip(ids, distances, strict=True), start=1)
        ]


def build_store(config: StoreConfig) -> VectorStore:
    """条件から Vector DB を組み立てる。"""
    if config.kind == "memory":
        return InMemoryStore()
    return ChromaStore()
