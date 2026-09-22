"""チャンク分割と、文字オフセットの保存。

オフセットがずれると評価セットの正解根拠と突き合わせられなくなり、
比較実験そのものが成立しなくなる。境界を重点的に見る。
"""

from __future__ import annotations

import itertools
from pathlib import Path

import pytest
from pydantic import ValidationError

from rageval.config import ChunkConfig
from rageval.ingest import Chunk, Document, FixedSizeChunker, chunk_documents, load_corpus


def test_offsets_point_back_at_the_original_text() -> None:
    doc = Document(doc_id="d", text="あいうえおかきくけこさしすせそ")
    chunks = FixedSizeChunker(size=6, overlap=2).split(doc)
    for chunk in chunks:
        assert doc.text[chunk.start : chunk.end] == chunk.text


def test_chunks_advance_by_size_minus_overlap() -> None:
    doc = Document(doc_id="d", text="0123456789")
    chunks = FixedSizeChunker(size=4, overlap=1).split(doc)
    assert [(c.start, c.end) for c in chunks] == [(0, 4), (3, 7), (6, 10)]


def test_the_last_chunk_is_not_padded_and_the_loop_ends() -> None:
    doc = Document(doc_id="d", text="01234567")
    chunks = FixedSizeChunker(size=5, overlap=2).split(doc)
    assert chunks[-1].end == len(doc.text)
    assert chunks[-1].text == doc.text[chunks[-1].start :]


def test_a_document_shorter_than_one_chunk_yields_one_chunk() -> None:
    doc = Document(doc_id="d", text="みじかい")
    chunks = FixedSizeChunker(size=512, overlap=64).split(doc)
    assert len(chunks) == 1
    assert chunks[0].start == 0
    assert chunks[0].end == 4


def test_an_empty_document_yields_no_chunks() -> None:
    assert FixedSizeChunker(size=8, overlap=2).split(Document(doc_id="d", text="")) == []


def test_chunk_ids_are_stable_and_ordered() -> None:
    doc = Document(doc_id="doc01", text="0123456789")
    chunks = FixedSizeChunker(size=4, overlap=1).split(doc)
    assert [c.chunk_id for c in chunks] == ["doc01#c000", "doc01#c001", "doc01#c002"]
    assert [c.index for c in chunks] == [0, 1, 2]


def test_overlap_makes_a_span_land_in_two_chunks() -> None:
    """重複幅があると、1つの根拠が複数チャンクにまたがる。評価はその両方を正解とみなす。"""
    doc = Document(doc_id="d", text="0123456789")
    chunks = FixedSizeChunker(size=4, overlap=2).split(doc)
    covering = [c.chunk_id for c in chunks if c.covers("d", 3, 4)]
    assert len(covering) == 2


def test_covers_is_exclusive_at_the_boundary() -> None:
    chunk = Chunk(chunk_id="d#c000", doc_id="d", index=0, start=10, end=20, text="x" * 10)
    assert chunk.covers("d", 19, 20) is True
    assert chunk.covers("d", 20, 21) is False, "end は含まない"
    assert chunk.covers("d", 9, 10) is False
    assert chunk.covers("d", 9, 11) is True


def test_covers_requires_the_same_document() -> None:
    chunk = Chunk(chunk_id="a#c000", doc_id="a", index=0, start=0, end=10, text="x" * 10)
    assert chunk.covers("b", 0, 10) is False


def test_from_config() -> None:
    chunker = FixedSizeChunker.from_config(ChunkConfig(size=256, overlap=32))
    assert (chunker.size, chunker.overlap) == (256, 32)


def test_load_corpus_is_sorted_by_filename(tmp_path: Path) -> None:
    """並び順が環境で変わるとチャンクIDがずれて、再現性が壊れる。"""
    for name in ("doc02", "doc01", "doc03"):
        (tmp_path / f"{name}.txt").write_text(f"{name} の本文", encoding="utf-8")
    assert [d.doc_id for d in load_corpus(tmp_path)] == ["doc01", "doc02", "doc03"]


def test_load_corpus_rejects_an_empty_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"コーパスに \.txt がない"):
        load_corpus(tmp_path)


def test_chunk_documents_covers_every_document() -> None:
    docs = [Document(doc_id="a", text="0" * 10), Document(doc_id="b", text="1" * 10)]
    chunks = chunk_documents(docs, FixedSizeChunker(size=4, overlap=0))
    assert {c.doc_id for c in chunks} == {"a", "b"}


# ---- 境界の位相（phase） -------------------------------------------------


def test_phase_zero_keeps_the_previous_behaviour() -> None:
    """既定では、phase を入れる前とまったく同じ切り方になること。"""
    doc = Document(doc_id="d", text="0123456789")
    assert FixedSizeChunker(size=4, overlap=1, phase=0).split(doc) == FixedSizeChunker(
        size=4, overlap=1
    ).split(doc)


def test_phase_shifts_every_boundary() -> None:
    doc = Document(doc_id="d", text="0123456789")
    plain = [(c.start, c.end) for c in FixedSizeChunker(size=4, overlap=0).split(doc)]
    shifted = [(c.start, c.end) for c in FixedSizeChunker(size=4, overlap=0, phase=1).split(doc)]
    assert plain == [(0, 4), (4, 8), (8, 10)]
    # 最初だけ短くなり、以降の境界が1文字ずつずれる
    assert shifted == [(0, 3), (3, 7), (7, 10)]


def test_phase_keeps_full_coverage_and_offsets() -> None:
    """位相をずらしても、本文を取りこぼさず、オフセットも元の文書を指していること。"""
    doc = Document(doc_id="d", text="あいうえおかきくけこさしすせそたちつてと")
    # phase は size - overlap 未満でないと分割が進まない（ChunkConfig が弾く条件）
    for phase in range(0, 6 - 2):
        chunks = FixedSizeChunker(size=6, overlap=2, phase=phase).split(doc)
        assert chunks[0].start == 0
        assert chunks[-1].end == len(doc.text)
        for chunk in chunks:
            assert doc.text[chunk.start : chunk.end] == chunk.text
        # 隣り合うチャンクの間に穴が開いていない
        for previous, following in itertools.pairwise(chunks):
            assert following.start <= previous.end


def test_phase_changes_which_chunk_holds_a_span() -> None:
    """位相が変われば、同じ根拠を含むチャンクも変わる。ノイズ幅を測る土台。"""
    doc = Document(doc_id="d", text="0123456789" * 5)
    a = [
        c.chunk_id for c in FixedSizeChunker(size=10, overlap=0).split(doc) if c.covers("d", 9, 11)
    ]
    b = [
        c.chunk_id
        for c in FixedSizeChunker(size=10, overlap=0, phase=5).split(doc)
        if c.covers("d", 9, 11)
    ]
    assert a != b


def test_phase_must_be_smaller_than_size() -> None:
    with pytest.raises(ValidationError, match="phase"):
        ChunkConfig(size=64, overlap=8, phase=64)


def test_phase_must_leave_room_for_the_overlap() -> None:
    with pytest.raises(ValidationError, match="overlap"):
        ChunkConfig(size=64, overlap=32, phase=40)


# ---- 境界値と文字コード --------------------------------------------------


def test_a_corpus_file_with_a_bom_is_rejected(tmp_path: Path) -> None:
    """BOM は1文字として残り、その文書の正解根拠のオフセットが全部ずれる。

    黙って剥がすと、剥がす前に作った評価セットと辻褄が合わなくなる。止めるのが正しい。
    """
    (tmp_path / "bom.txt").write_bytes("\ufeff本文".encode())
    with pytest.raises(ValueError, match="BOM"):
        load_corpus(tmp_path)


def test_an_empty_corpus_file_is_rejected(tmp_path: Path) -> None:
    """空の文書はチャンクを生まないので、黙って無視すると入れ忘れに気づけない。"""
    (tmp_path / "ok.txt").write_text("本文がある", encoding="utf-8")
    (tmp_path / "empty.txt").write_text("   \n", encoding="utf-8")
    with pytest.raises(ValueError, match="が空"):
        load_corpus(tmp_path)


def test_the_largest_allowed_overlap_still_advances() -> None:
    """overlap = size - 1 は許される上限。ここで分割が止まらないこと。"""
    doc = Document(doc_id="d", text="0123456789" * 3)
    chunks = FixedSizeChunker(size=10, overlap=9).split(doc)
    assert chunks[-1].end == len(doc.text), "最後まで覆えていない"
    assert all(doc.text[c.start : c.end] == c.text for c in chunks)


def test_a_document_exactly_one_chunk_long() -> None:
    doc = Document(doc_id="d", text="0123456789")
    chunks = FixedSizeChunker(size=10, overlap=2).split(doc)
    assert len(chunks) == 1
    assert (chunks[0].start, chunks[0].end) == (0, 10)


def test_surrogate_pairs_keep_the_offsets_consistent() -> None:
    """𠮷 や絵文字は UTF-8 で4バイトだが、オフセットは Python の文字単位で数える。

    日本語のコーパスには稀な漢字が混じりうる。バイト単位と取り違えるとずれる。
    """
    doc = Document(doc_id="d", text="𠮷野家🍜のラーメン")
    assert len(doc.text) == 9, "Python の長さは符号位置の数"
    assert len(doc.text.encode("utf-8")) == 29, "バイト数とは一致しない"
    chunks = FixedSizeChunker(size=5, overlap=0).split(doc)
    for chunk in chunks:
        assert doc.text[chunk.start : chunk.end] == chunk.text
    assert chunks[-1].end == len(doc.text)


def test_a_span_at_the_very_start_and_end_is_covered() -> None:
    doc = Document(doc_id="d", text="0123456789")
    chunks = FixedSizeChunker(size=4, overlap=0).split(doc)
    assert any(c.covers("d", 0, 1) for c in chunks), "先頭の根拠"
    assert any(c.covers("d", 9, 10) for c in chunks), "末尾の根拠"
