"""クエリに対する文脈の取得。top-k の適用と再ランクの切り替え。

差し替え可能な4点のうち「再ランカー」がここにある（ADR 0002）。実験004の軸。

`candidates` を指定しない場合、再ランクは **top-k で取った集合をそのまま並べ替える**。
このとき Recall@k は定義上変わらず、動くのは MRR だけになる。これは
docs/03_experiment_plan.md の実験004の仮説そのものなので、既定はこの形にしてある。
候補を広げて集合ごと入れ替えたい場合だけ `candidates` を top_k より大きくする。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from rageval.config import RetrieveConfig
from rageval.embed import Embedder, character_ngrams, normalize_text
from rageval.store import SearchHit, VectorStore


class Reranker(Protocol):
    """差し替え点：再ランカー。"""

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]: ...


@dataclass(frozen=True)
class LexicalReranker:
    """クエリとチャンクの文字 n-gram の重なりで並べ替える。

    外部APIに出ない決定的な再ランカー。埋め込みの距離が「だいたい似ている」で
    拾ってきた候補を、表層の一致で締め直す。交差エンコーダのような賢さは無いが、
    再ランクという段が入ることで順位がどう動くかは観察できる。
    """

    sizes: tuple[int, ...] = (2, 3)

    def score(self, query: str, text: str) -> float:
        query_grams = set(character_ngrams(normalize_text(query), self.sizes))
        if not query_grams:
            return 0.0
        text_grams = set(character_ngrams(normalize_text(text), self.sizes))
        return len(query_grams & text_grams) / len(query_grams)

    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]:
        scored = [(self.score(query, hit.chunk.text), hit) for hit in hits]
        # 同点は元の順位を尊重し、それも同じなら chunk_id で決め切る。
        scored.sort(key=lambda pair: (-pair[0], pair[1].rank, pair[1].chunk.chunk_id))
        return [
            SearchHit(chunk=hit.chunk, score=score, rank=rank)
            for rank, (score, hit) in enumerate(scored, start=1)
        ]


def build_reranker(config: RetrieveConfig) -> Reranker | None:
    return LexicalReranker() if config.rerank else None


@dataclass
class Retriever:
    """埋め込み・Vector DB・再ランカーをつないで、質問から文脈を引く。"""

    embedder: Embedder
    store: VectorStore
    config: RetrieveConfig
    reranker: Reranker | None = None

    def retrieve(self, question: str) -> list[SearchHit]:
        depth = self.config.candidates or self.config.top_k
        vector = self.embedder.embed([question])[0]
        hits = self.store.search(vector, depth)
        if self.reranker is not None:
            hits = self.reranker.rerank(question, hits)
        # 再ランク後に top_k へ絞り、順位を振り直す。評価はこの順位で見る。
        return [
            SearchHit(chunk=hit.chunk, score=hit.score, rank=rank)
            for rank, hit in enumerate(hits[: self.config.top_k], start=1)
        ]
