"""指標の算出。

指標の定義がずれると、過去の実験と比較できなくなる。
docs/02_evaluation.md に書いた定義そのものを、ここで固定する。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rageval.evaluate import (
    GoldSpan,
    Prediction,
    QaItem,
    gold_chunk_ids,
    load_qa_set,
    score,
)
from rageval.generate import Answer
from rageval.ingest import Chunk, Document, FixedSizeChunker


def item(qid: str, *, answerable: bool = True, span: tuple[int, int] | None = None) -> QaItem:
    spans = [GoldSpan(doc="d", start=span[0], end=span[1])] if span else []
    return QaItem(qid=qid, question=f"{qid} の質問", gold_spans=spans, answerable=answerable)


def prediction(
    qid: str,
    retrieved: list[str],
    *,
    cited: list[str] | None = None,
    abstained: bool = False,
    failed: bool = False,
) -> Prediction:
    return Prediction(
        qid=qid,
        retrieved_chunk_ids=retrieved,
        answer=Answer(answer="", cited_chunk_ids=cited or [], abstained=abstained),
        generation_failed=failed,
    )


# ---- 正解チャンクの計算 --------------------------------------------------


def test_gold_chunks_are_recomputed_for_each_chunking() -> None:
    """同じ根拠でも、チャンクサイズが変われば正解チャンクは変わる。

    これが成り立つから、同じ評価セットでチャンクサイズを比較できる。
    """
    doc = Document(doc_id="d", text="0123456789" * 10)
    target = item("q1", span=(45, 55))

    coarse = gold_chunk_ids(target, FixedSizeChunker(size=100, overlap=0).split(doc))
    fine = gold_chunk_ids(target, FixedSizeChunker(size=10, overlap=0).split(doc))
    assert len(coarse) == 1
    assert len(fine) == 2, "細かく切ると根拠が2つのチャンクにまたがる"


def test_a_span_inside_an_overlap_belongs_to_both_chunks() -> None:
    doc = Document(doc_id="d", text="0123456789")
    chunks = FixedSizeChunker(size=6, overlap=4).split(doc)
    assert len(gold_chunk_ids(item("q", span=(4, 5)), chunks)) >= 2


def test_an_unanswerable_question_has_no_gold_chunk() -> None:
    chunks = [Chunk(chunk_id="d#c000", doc_id="d", index=0, start=0, end=10, text="x" * 10)]
    assert gold_chunk_ids(item("q", answerable=False), chunks) == set()


# ---- 指標 ---------------------------------------------------------------


def test_recall_counts_a_question_once_even_with_several_gold_chunks() -> None:
    result = score(
        [item("q1", span=(0, 1))],
        {"q1": prediction("q1", ["a", "b"])},
        {"q1": {"a", "b"}},
        k=5,
    )
    assert result.recall_at_k == 1.0


def test_recall_ignores_hits_below_k() -> None:
    result = score(
        [item("q1", span=(0, 1))],
        {"q1": prediction("q1", ["x", "y", "gold"])},
        {"q1": {"gold"}},
        k=2,
    )
    assert result.recall_at_k == 0.0


def test_mrr_uses_the_first_gold_rank() -> None:
    result = score(
        [item("q1", span=(0, 1)), item("q2", span=(0, 1))],
        {
            "q1": prediction("q1", ["gold", "x"]),
            "q2": prediction("q2", ["x", "y", "gold"]),
        },
        {"q1": {"gold"}, "q2": {"gold"}},
        k=5,
    )
    assert result.mrr == pytest.approx((1.0 + 1 / 3) / 2)


def test_mrr_is_zero_when_nothing_is_retrieved() -> None:
    result = score(
        [item("q1", span=(0, 1))], {"q1": prediction("q1", ["x"])}, {"q1": {"gold"}}, k=5
    )
    assert result.mrr == 0.0


def test_citation_matches_only_when_every_citation_is_gold() -> None:
    """正解を1つ挙げても、無関係なチャンクも挙げていれば不一致（docs/02_evaluation.md）。"""
    items = [item("q1", span=(0, 1)), item("q2", span=(0, 1)), item("q3", span=(0, 1))]
    predictions = {
        "q1": prediction("q1", ["gold"], cited=["gold"]),
        "q2": prediction("q2", ["gold", "junk"], cited=["gold", "junk"]),
        "q3": prediction("q3", ["gold"], cited=[]),
    }
    gold = {qid: {"gold"} for qid in ("q1", "q2", "q3")}
    assert score(items, predictions, gold, k=5).citation_match == pytest.approx(1 / 3)


def test_abstention_is_measured_only_on_unanswerable_questions() -> None:
    items = [
        item("q1", span=(0, 1)),
        item("q2", answerable=False),
        item("q3", answerable=False),
    ]
    predictions = {
        "q1": prediction("q1", ["gold"], cited=["gold"]),
        "q2": prediction("q2", [], abstained=True),
        "q3": prediction("q3", ["x"], cited=["x"]),
    }
    result = score(items, predictions, {"q1": {"gold"}}, k=5)
    assert result.abstention == pytest.approx(0.5)
    assert result.answerable == 1
    assert result.unanswerable == 2


def test_abstaining_on_an_answerable_question_is_counted_separately() -> None:
    result = score(
        [item("q1", span=(0, 1))],
        {"q1": prediction("q1", ["gold"], abstained=True)},
        {"q1": {"gold"}},
        k=5,
    )
    assert result.false_abstention == 1
    assert result.citation_match == 0.0
    assert result.recall_at_k == 1.0, "棄権しても検索は当たっている"


def test_generation_failures_are_counted() -> None:
    result = score(
        [item("q1", span=(0, 1))],
        {"q1": prediction("q1", ["gold"], failed=True)},
        {"q1": {"gold"}},
        k=5,
    )
    assert result.generation_failures == 1


def test_a_missing_prediction_is_an_error_not_a_zero() -> None:
    with pytest.raises(KeyError, match="q1"):
        score([item("q1", span=(0, 1))], {}, {}, k=5)


def test_scores_serialise() -> None:
    result = score(
        [item("q1", span=(0, 1))],
        {"q1": prediction("q1", ["gold"], cited=["gold"])},
        {"q1": {"gold"}},
        k=5,
    )
    assert result.as_dict()["recall_at_k"] == 1.0


# ---- 評価セットの読み込み ------------------------------------------------


def test_load_qa_set(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text(
        '{"qid": "q001", "question": "?", '
        '"gold_spans": [{"doc": "d", "start": 1, "end": 2}], '
        '"answerable": true, "factoid": true}\n'
        '\n{"qid": "q002", "question": "?", "gold_spans": [], "answerable": false, '
        '"factoid": false}\n',
        encoding="utf-8",
    )
    items = load_qa_set(path)
    assert [i.qid for i in items] == ["q001", "q002"]
    assert items[0].gold_spans[0].end == 2


def test_load_qa_set_rejects_an_unknown_field(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text('{"qid": "q", "question": "?", "note": "x"}\n', encoding="utf-8")
    with pytest.raises(Exception, match="note"):
        load_qa_set(path)


def test_load_qa_set_rejects_an_empty_file(tmp_path: Path) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="評価セットが空"):
        load_qa_set(path)


def test_an_answerable_question_without_gold_chunks_is_an_error() -> None:
    """オフセットがずれると正解チャンクが空になる。低いスコアとして紛れ込ませない。"""
    with pytest.raises(ValueError, match="正解チャンクが1つも無い"):
        score(
            [item("q1", span=(0, 1))],
            {"q1": prediction("q1", ["x"])},
            {"q1": set()},
            k=5,
        )


def test_the_error_points_at_the_check_command() -> None:
    with pytest.raises(ValueError, match=re.escape("build_qa_set.py --check")):
        score([item("q1", span=(0, 1))], {"q1": prediction("q1", [])}, {}, k=5)
