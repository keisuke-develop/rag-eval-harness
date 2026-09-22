"""評価セットの読み込みと指標の算出。

指標は差し替え式にしていない（ADR 0002）。定義が1箇所に集まるほうが、
何を測っているかを読み取りやすいため。増やすときはここを直接触る。

正解根拠は文字オフセットで持つので、**条件ごとに正解チャンクを計算し直す**。
チャンクサイズを変えれば、同じ質問でも正解とみなすチャンクは変わる。
ここが評価セットとパイプラインの接点であり、比較実験が成立する理由そのもの。

## 指標の定義

| 指標 | 母数 | 正解の条件 |
|---|---|---|
| Recall@k | 答えられる質問 | 上位k件に正解チャンクが1つでも入っている |
| MRR | 答えられる質問 | 正解チャンクが最初に現れた順位の逆数（入らなければ0） |
| 根拠一致率 | 答えられる質問 | 棄権せず、挙げた根拠がすべて正解チャンクである |
| 棄権率 | 文脈に答えが無い質問 | `abstained` が true |

根拠一致率で「挙げた根拠が**すべて**正解チャンク」を条件にしているのは、
docs/02_evaluation.md が「正解根拠のチャンクに含まれる情報**だけ**で
構成されているか」と定義しているため。正解を1つ挙げつつ無関係なチャンクも
挙げた場合は、文脈の外を使った可能性があるので不一致として扱う。
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from rageval.generate import Answer
from rageval.ingest import Chunk


class GoldSpan(BaseModel):
    """正解根拠の文字オフセット範囲（`end` は含まない）。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    doc: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)


class QaItem(BaseModel):
    """評価セットの1問。"""

    model_config = ConfigDict(extra="forbid", frozen=True)

    qid: str
    question: str
    gold_spans: list[GoldSpan] = Field(default_factory=list)
    answerable: bool = True
    factoid: bool = False


def load_qa_set(path: Path) -> list[QaItem]:
    """JSONL の評価セットを読む。"""
    items = [
        QaItem.model_validate(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not items:
        raise ValueError(f"評価セットが空: {path}")
    return items


def gold_chunk_ids(item: QaItem, chunks: Sequence[Chunk]) -> set[str]:
    """この条件のチャンク分割のもとで、正解根拠を含むチャンクのID。

    重複幅があると1つの根拠が複数のチャンクにまたがる。そのすべてを正解とみなす。
    """
    return {
        chunk.chunk_id
        for chunk in chunks
        for span in item.gold_spans
        if chunk.covers(span.doc, span.start, span.end)
    }


@dataclass(frozen=True)
class Prediction:
    """1問ぶんの実行結果。"""

    qid: str
    retrieved_chunk_ids: list[str]
    answer: Answer
    generation_failed: bool = False
    seconds: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "qid": self.qid,
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
            "answer": self.answer.model_dump(),
            "generation_failed": self.generation_failed,
            "seconds": round(self.seconds, 4),
        }


@dataclass(frozen=True)
class Scores:
    """1条件ぶんのスコア。"""

    k: int
    answerable: int
    unanswerable: int
    recall_at_k: float
    mrr: float
    citation_match: float
    abstention: float
    false_abstention: int
    generation_failures: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "k": self.k,
            "answerable": self.answerable,
            "unanswerable": self.unanswerable,
            "recall_at_k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "citation_match": round(self.citation_match, 4),
            "abstention": round(self.abstention, 4),
            "false_abstention": self.false_abstention,
            "generation_failures": self.generation_failures,
        }


def _ratio(hits: int, total: int) -> float:
    return hits / total if total else 0.0


def score(
    items: Iterable[QaItem],
    predictions: dict[str, Prediction],
    gold: dict[str, set[str]],
    k: int,
) -> Scores:
    """予測と正解を突き合わせて指標を出す。"""
    answerable = 0
    unanswerable = 0
    recall_hits = 0
    reciprocal_sum = 0.0
    citation_hits = 0
    abstention_hits = 0
    false_abstention = 0
    generation_failures = 0

    for item in items:
        prediction = predictions.get(item.qid)
        if prediction is None:
            raise KeyError(f"{item.qid} の予測が無い")
        if prediction.generation_failed:
            generation_failures += 1

        if not item.answerable:
            unanswerable += 1
            if prediction.answer.abstained:
                abstention_hits += 1
            continue

        answerable += 1
        gold_ids = gold.get(item.qid, set())
        if not gold_ids:
            # チャンクは文書を隙間なく覆うので、範囲内の根拠は必ずどれかに入る。
            # 空になるのは、オフセットが文書の外を指しているとき——つまり
            # 評価セットとコーパスがずれたとき。低いスコアとして紛れ込ませず、止める。
            raise ValueError(
                f"{item.qid}: 答えられる質問なのに正解チャンクが1つも無い。"
                "評価セットの文字オフセットとコーパスがずれている可能性がある。"
                "`uv run python datasets/build_qa_set.py --check` で突き合わせること"
            )
        retrieved = prediction.retrieved_chunk_ids[:k]

        if gold_ids & set(retrieved):
            recall_hits += 1
        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in gold_ids:
                reciprocal_sum += 1.0 / rank
                break

        cited = set(prediction.answer.cited_chunk_ids)
        if prediction.answer.abstained:
            false_abstention += 1
        elif cited and cited <= gold_ids:
            citation_hits += 1

    return Scores(
        k=k,
        answerable=answerable,
        unanswerable=unanswerable,
        recall_at_k=_ratio(recall_hits, answerable),
        mrr=reciprocal_sum / answerable if answerable else 0.0,
        citation_match=_ratio(citation_hits, answerable),
        abstention=_ratio(abstention_hits, unanswerable),
        false_abstention=false_abstention,
        generation_failures=generation_failures,
    )
