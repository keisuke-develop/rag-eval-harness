"""評価セットがコーパスと整合していることを確かめる。

正解根拠は文字オフセットで持つため、コーパスを1文字でも動かすと評価セットが
静かに壊れる。壊れたことに気づけるよう、オフセットが実際に意図した本文を
指しているかをここで検査する。docs/02_evaluation.md「チャンクIDの安定性」を参照。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

DATASETS_DIR = Path(__file__).resolve().parents[2] / "datasets"
CORPUS_DIR = DATASETS_DIR / "corpus"
QA_SET_PATH = DATASETS_DIR / "qa_set.jsonl"

MIN_QUESTIONS, MAX_QUESTIONS = 40, 60
MIN_UNANSWERABLE_RATIO, MAX_UNANSWERABLE_RATIO = 0.05, 0.20
MIN_FACTOID_RATIO = 0.50


def load_rows() -> list[dict[str, Any]]:
    lines = QA_SET_PATH.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_corpus() -> dict[str, str]:
    return {path.stem: path.read_text(encoding="utf-8") for path in CORPUS_DIR.glob("*.txt")}


@pytest.fixture(scope="module")
def rows() -> list[dict[str, Any]]:
    return load_rows()


@pytest.fixture(scope="module")
def corpus() -> dict[str, str]:
    return load_corpus()


def test_corpus_is_present(corpus: dict[str, str]) -> None:
    assert corpus, "コーパスが空。datasets/fetch_corpus.py を実行する"
    for doc_id, text in corpus.items():
        assert text.strip(), f"{doc_id} が空"


def test_question_count_is_in_range(rows: list[dict[str, Any]]) -> None:
    assert MIN_QUESTIONS <= len(rows) <= MAX_QUESTIONS


def test_qids_are_unique(rows: list[dict[str, Any]]) -> None:
    qids = [row["qid"] for row in rows]
    assert len(qids) == len(set(qids))


def test_questions_are_not_empty(rows: list[dict[str, Any]]) -> None:
    for row in rows:
        assert row["question"].strip(), f"{row['qid']} の質問文が空"


def test_gold_spans_point_at_real_text(rows: list[dict[str, Any]], corpus: dict[str, str]) -> None:
    """オフセットが範囲内にあり、空でない本文を指していること。"""
    for row in rows:
        for span in row["gold_spans"]:
            text = corpus.get(span["doc"])
            assert text is not None, f"{row['qid']}: 文書 {span['doc']} がコーパスにない"
            assert 0 <= span["start"] < span["end"] <= len(text), (
                f"{row['qid']}: オフセット {span['start']}-{span['end']} が "
                f"{span['doc']}（{len(text)} 字）の範囲外"
            )
            assert text[span["start"] : span["end"]].strip(), (
                f"{row['qid']}: 根拠の範囲が空白だけを指している"
            )


def test_answerable_questions_have_exactly_one_span(rows: list[dict[str, Any]]) -> None:
    """1問1正解根拠。崩すと Recall の解釈ができなくなる（docs/02_evaluation.md）。"""
    for row in rows:
        expected = 1 if row["answerable"] else 0
        assert len(row["gold_spans"]) == expected, (
            f"{row['qid']}: answerable={row['answerable']} なのに根拠が "
            f"{len(row['gold_spans'])} 個ある"
        )


def test_unanswerable_share_is_around_one_tenth(rows: list[dict[str, Any]]) -> None:
    ratio = sum(1 for row in rows if not row["answerable"]) / len(rows)
    assert MIN_UNANSWERABLE_RATIO <= ratio <= MAX_UNANSWERABLE_RATIO


def test_factoid_questions_are_at_least_half(rows: list[dict[str, Any]]) -> None:
    ratio = sum(1 for row in rows if row["factoid"]) / len(rows)
    assert ratio >= MIN_FACTOID_RATIO


def test_every_document_is_covered(rows: list[dict[str, Any]], corpus: dict[str, str]) -> None:
    """どの文書からも質問が出ていること。偏ると検索の実力が測れない。"""
    used = {span["doc"] for row in rows for span in row["gold_spans"]}
    assert used == set(corpus), f"質問のない文書がある: {sorted(set(corpus) - used)}"
