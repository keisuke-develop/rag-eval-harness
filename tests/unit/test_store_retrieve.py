"""Vector DB と検索。

順位が実行のたびに変わらないこと、再ランクが top-k と噛み合っていることを見る。
"""

from __future__ import annotations

from typing import Any

import pytest

from rageval.config import RetrieveConfig, StoreConfig
from rageval.embed import HashingEmbedder, Vector
from rageval.ingest import Chunk
from rageval.retrieve import LexicalReranker, Retriever, build_reranker
from rageval.store import ChromaStore, InMemoryStore, SearchHit, build_store


def chunk(name: str, text: str = "本文") -> Chunk:
    return Chunk(chunk_id=name, doc_id="d", index=0, start=0, end=len(text), text=text)


def test_search_orders_by_cosine_similarity() -> None:
    store = InMemoryStore()
    store.upsert(
        [chunk("a"), chunk("b"), chunk("c")],
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
    )
    hits = store.search([1.0, 0.0], k=3)
    assert [hit.chunk.chunk_id for hit in hits] == ["a", "c", "b"]
    assert [hit.rank for hit in hits] == [1, 2, 3]


def test_ties_are_broken_by_chunk_id_so_the_order_is_reproducible() -> None:
    store = InMemoryStore()
    store.upsert([chunk("b"), chunk("a")], [[1.0, 0.0], [1.0, 0.0]])
    assert [hit.chunk.chunk_id for hit in store.search([1.0, 0.0], k=2)] == ["a", "b"]


def test_search_returns_at_most_k() -> None:
    store = InMemoryStore()
    store.upsert([chunk(f"c{i}") for i in range(5)], [[1.0, 0.0]] * 5)
    assert len(store.search([1.0, 0.0], k=2)) == 2


def test_an_empty_store_returns_nothing() -> None:
    assert InMemoryStore().search([1.0], k=3) == []


def test_vectors_are_normalised_so_length_does_not_decide_the_order() -> None:
    store = InMemoryStore()
    store.upsert([chunk("long"), chunk("short")], [[10.0, 0.0], [0.0, 1.0]])
    hits = store.search([0.0, 1.0], k=2)
    assert hits[0].chunk.chunk_id == "short"


def test_mismatched_lengths_are_rejected() -> None:
    with pytest.raises(ValueError, match="ベクトルが"):
        InMemoryStore().upsert([chunk("a")], [[1.0], [2.0]])


def test_build_store() -> None:
    assert isinstance(build_store(StoreConfig(kind="memory")), InMemoryStore)
    assert isinstance(build_store(StoreConfig(kind="chroma")), ChromaStore)


@pytest.mark.integration
def test_chroma_finds_the_nearest_vector() -> None:
    # chromadb は既定では入れていない（未修正の脆弱性があり、どの実験でも使っていない）。
    # 入れている環境でだけ確かめる。
    pytest.importorskip("chromadb", reason="uv sync --extra chroma で入る")
    store = ChromaStore(collection_name="test_rageval")
    store.upsert(
        [chunk("a"), chunk("b")],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    hits = store.search([1.0, 0.0], k=1)
    assert hits[0].chunk.chunk_id == "a"
    # コサイン距離を類似度に直しているので、向きが揃っていれば 1 に近い
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)


# ---- 検索と再ランク ------------------------------------------------------


def build_retriever(**kwargs: object) -> Retriever:
    embedder = HashingEmbedder(dim=256)
    chunks = [
        chunk("c1", "富士山の最高地点の標高は3776.12メートルである"),
        chunk("c2", "TCPのヘッダ長は20バイトから60バイトである"),
        chunk("c3", "和紙は繊維が長いため薄くても強い"),
    ]
    store = InMemoryStore()
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    config = RetrieveConfig.model_validate({"top_k": 2, **kwargs})
    return Retriever(embedder=embedder, store=store, config=config, reranker=build_reranker(config))


def test_retrieve_returns_top_k_with_ranks_from_one() -> None:
    hits = build_retriever().retrieve("富士山の標高はいくつか")
    assert len(hits) == 2
    assert [hit.rank for hit in hits] == [1, 2]
    assert hits[0].chunk.chunk_id == "c1"


def test_candidates_widens_the_pool_before_reranking() -> None:
    """candidates を広げると、再ランクが集合そのものを入れ替えられる。"""
    retriever = build_retriever(rerank=True, candidates=3)
    hits = retriever.retrieve("和紙の繊維")
    assert len(hits) == 2
    assert hits[0].chunk.chunk_id == "c3"


def test_reranker_reorders_without_changing_the_set_by_default() -> None:
    """candidates を指定しなければ、集合は変わらず順位だけが動く（実験004の前提）。"""
    plain = build_retriever()
    reranked = build_retriever(rerank=True)
    question = "和紙はなぜ強いのか"
    assert {h.chunk.chunk_id for h in plain.retrieve(question)} == {
        h.chunk.chunk_id for h in reranked.retrieve(question)
    }


def test_lexical_reranker_scores_containment() -> None:
    reranker = LexicalReranker()
    assert reranker.score("あいうえお", "あいうえお") == pytest.approx(1.0)
    assert reranker.score("あいうえお", "かきくけこ") == 0.0
    assert reranker.score("", "なんでも") == 0.0


def test_lexical_reranker_renumbers_ranks() -> None:
    hits = [
        SearchHit(chunk=chunk("a", "まったく無関係"), score=0.9, rank=1),
        SearchHit(chunk=chunk("b", "富士山の標高"), score=0.1, rank=2),
    ]
    reranked = LexicalReranker().rerank("富士山の標高", hits)
    assert [hit.chunk.chunk_id for hit in reranked] == ["b", "a"]
    assert [hit.rank for hit in reranked] == [1, 2]


def test_build_reranker_is_off_by_default() -> None:
    assert build_reranker(RetrieveConfig(top_k=5)) is None
    assert build_reranker(RetrieveConfig(top_k=5, rerank=True)) is not None


def test_embedder_protocol_accepts_a_stub() -> None:
    """Protocol なので、継承なしのダミーを差し込める（ADR 0002）。"""

    class Stub:
        def embed(self, texts: list[str]) -> list[Vector]:
            return [[1.0, 0.0] for _ in texts]

    store = InMemoryStore()
    store.upsert([chunk("a")], [[1.0, 0.0]])
    retriever = Retriever(embedder=Stub(), store=store, config=RetrieveConfig(top_k=1))
    assert retriever.retrieve("なんでも")[0].chunk.chunk_id == "a"


def test_top_k_larger_than_the_index_returns_everything() -> None:
    """top_k がチャンク数を超えても落ちず、あるだけ返すこと。"""
    embedder = HashingEmbedder(dim=64)
    chunks = [chunk("c1", "ひとつめ"), chunk("c2", "ふたつめ")]
    store = InMemoryStore()
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    retriever = Retriever(embedder=embedder, store=store, config=RetrieveConfig(top_k=100))
    hits = retriever.retrieve("なんでも")
    assert len(hits) == 2
    assert [h.rank for h in hits] == [1, 2], "順位は詰めて振り直す"


def test_top_k_of_one() -> None:
    embedder = HashingEmbedder(dim=64)
    chunks = [chunk("c1", "富士山の標高"), chunk("c2", "TCPのヘッダ")]
    store = InMemoryStore()
    store.upsert(chunks, embedder.embed([c.text for c in chunks]))
    retriever = Retriever(embedder=embedder, store=store, config=RetrieveConfig(top_k=1))
    assert len(retriever.retrieve("富士山の高さ")) == 1


# ---- ChromaStore のアダプタ部分 ------------------------------------------
#
# chromadb は既定で入れない（未修正の脆弱性があり、どの実験でも使っていない）。
# それでも「距離を類似度に直す」「IDからチャンクに戻す」といった
# こちら側の変換は、chromadb の有無と関係なく壊れうる。
# 偽の chromadb を差し込んで、その変換だけを確かめる。


class _FakeCollection:
    def __init__(self) -> None:
        self.ids: list[str] = []
        self.embeddings: list[list[float]] = []
        self.queried: list[list[float]] = []

    def add(self, ids: list[str], embeddings: list[list[float]]) -> None:
        self.ids = ids
        self.embeddings = embeddings

    # chromadb の戻り値は "ids" が文字列、"distances" が浮動小数で、値の型が揃わない。
    # Any を使う理由はそこにある。本物の型は chromadb 側にあり、ここでは真似できない。
    def query(self, query_embeddings: list[list[float]], n_results: int) -> dict[str, Any]:
        self.queried = query_embeddings
        # 近い順に返ってくる前提。距離は 1 - コサイン類似度。
        return {
            "ids": [self.ids[:n_results]],
            "distances": [[0.0, 0.25][:n_results]],
        }


class _FakeClient:
    def __init__(self) -> None:
        self.collection = _FakeCollection()
        self.deleted: list[str] = []

    def delete_collection(self, name: str) -> None:
        self.deleted.append(name)

    def create_collection(
        self, name: str, metadata: dict[str, str], embedding_function: object
    ) -> _FakeCollection:
        self.metadata = metadata
        return self.collection


def _install_fake_chromadb(monkeypatch: pytest.MonkeyPatch) -> _FakeClient:
    import sys
    import types

    client = _FakeClient()
    module = types.ModuleType("chromadb")
    module.EphemeralClient = lambda: client  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "chromadb", module)
    return client


def test_chroma_adapter_turns_distance_into_similarity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Chroma は距離を返す。他の store と指標の向きを揃えてから外に出すこと。"""
    _install_fake_chromadb(monkeypatch)
    store = ChromaStore()
    store.upsert([chunk("a"), chunk("b")], [[1.0, 0.0], [0.0, 1.0]])
    hits = store.search([1.0, 0.0], k=2)
    assert [hit.chunk.chunk_id for hit in hits] == ["a", "b"]
    assert [hit.score for hit in hits] == [1.0, 0.75], "score = 1 - distance"
    assert [hit.rank for hit in hits] == [1, 2]


def test_chroma_adapter_asks_for_cosine(monkeypatch: pytest.MonkeyPatch) -> None:
    """距離の定義を既定任せにしない。既定は L2 で、指標の意味が変わる。"""
    client = _install_fake_chromadb(monkeypatch)
    ChromaStore().upsert([chunk("a")], [[1.0, 0.0]])
    assert client.metadata == {"hnsw:space": "cosine"}


def test_chroma_adapter_drops_a_leftover_collection(monkeypatch: pytest.MonkeyPatch) -> None:
    """前の実験のコレクションが残っていると、条件の違う結果が混ざる。"""
    client = _install_fake_chromadb(monkeypatch)
    ChromaStore(collection_name="mine").upsert([chunk("a")], [[1.0, 0.0]])
    assert client.deleted == ["mine"]


def test_chroma_search_before_upsert_returns_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_chromadb(monkeypatch)
    assert ChromaStore().search([1.0, 0.0], k=3) == []


def test_chroma_rejects_mismatched_lengths(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_chromadb(monkeypatch)
    with pytest.raises(ValueError, match="ベクトルが"):
        ChromaStore().upsert([chunk("a")], [[1.0], [2.0]])


def test_chroma_explains_how_to_install_when_it_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """既定から外しているので、使おうとした人に入れ方を伝えること。"""
    import sys

    monkeypatch.setitem(sys.modules, "chromadb", None)
    with pytest.raises(RuntimeError, match="uv sync --extra chroma"):
        ChromaStore().upsert([chunk("a")], [[1.0, 0.0]])
