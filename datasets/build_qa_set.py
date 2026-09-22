"""qa_draft.yaml から qa_set.jsonl を生成し、docs/02_evaluation.md の作成基準を検査する。

正解根拠はチャンクIDではなく文字オフセットで持つ（docs/02_evaluation.md）。
オフセットを人が書くと必ずずれるため、原稿には「根拠として引用する本文」を書き、
このスクリプトがコーパスを引いてオフセットに変換する。

検査する内容:
    - 引用がコーパス本文に **ちょうど1回** 現れること（0回ならタイプミス、
      2回以上なら根拠が一意に定まらない）
    - 答えられない質問の割合が 1割前後であること
    - 固有名詞・数値を問う質問が半分以上あること
    - 質問文がコーパスの表現を写していないこと（連続一致の長さで見る）
    - 件数が 40〜60 件に収まっていること

使い方:
    uv run python datasets/build_qa_set.py          # 生成して保存
    uv run python datasets/build_qa_set.py --check  # 保存せず検査だけ行う
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from typing import Any

import yaml

DATASETS_DIR = Path(__file__).parent
CORPUS_DIR = DATASETS_DIR / "corpus"
DRAFT_PATH = DATASETS_DIR / "qa_draft.yaml"
QA_SET_PATH = DATASETS_DIR / "qa_set.jsonl"
META_PATH = DATASETS_DIR / "qa_set.meta.json"

#: 件数の範囲。docs/00_requirements.md「評価セットは 40〜60 件を目安とする」。
MIN_QUESTIONS, MAX_QUESTIONS = 40, 60

#: 答えられない質問の割合。docs/02_evaluation.md では 1割。
MIN_UNANSWERABLE_RATIO, MAX_UNANSWERABLE_RATIO = 0.05, 0.20

#: 固有名詞・数値を問う質問の下限。docs/02_evaluation.md「半分以上入れる」。
MIN_FACTOID_RATIO = 0.50

#: 質問文と根拠本文の間で許す連続一致の上限（文字数）。
#: 固有名詞の共有は避けられないが、これを超えると節ごと写している疑いが強い。
MAX_SHARED_RUN = 15


def longest_common_run(a: str, b: str) -> str:
    """a と b に共通する最長の連続部分文字列を返す。"""
    best_end, best_len = 0, 0
    previous = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        current = [0] * (len(b) + 1)
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                current[j] = previous[j - 1] + 1
                if current[j] > best_len:
                    best_len, best_end = current[j], i
        previous = current
    return a[best_end - best_len : best_end]


def load_corpus() -> dict[str, str]:
    corpus = {path.stem: path.read_text(encoding="utf-8") for path in CORPUS_DIR.glob("*.txt")}
    if not corpus:
        raise RuntimeError(f"コーパスが空: {CORPUS_DIR}。先に fetch_corpus.py を実行する。")
    return corpus


def build(draft: dict[str, Any], corpus: dict[str, str]) -> tuple[list[dict[str, Any]], list[str]]:
    """原稿を評価セットの行に変換する。戻り値は (行, 検査で見つかった問題)。"""
    rows: list[dict[str, Any]] = []
    problems: list[str] = []
    seen: set[str] = set()

    for item in draft["questions"]:
        qid = str(item["qid"])
        question = str(item["question"])
        answerable = bool(item.get("answerable", True))

        if qid in seen:
            problems.append(f"{qid}: qid が重複している")
        seen.add(qid)

        spans: list[dict[str, Any]] = []
        if answerable:
            doc_id = str(item["doc"])
            evidence = str(item["evidence"])
            text = corpus.get(doc_id)
            if text is None:
                problems.append(f"{qid}: 文書 {doc_id} がコーパスにない")
            else:
                hits = text.count(evidence)
                if hits == 0:
                    problems.append(f"{qid}: 引用が {doc_id} に見つからない -> {evidence[:40]}...")
                elif hits > 1:
                    problems.append(f"{qid}: 引用が {doc_id} に {hits} 回現れ、根拠が一意でない")
                else:
                    start = text.index(evidence)
                    spans.append({"doc": doc_id, "start": start, "end": start + len(evidence)})

                shared = longest_common_run(question, evidence)
                if len(shared) > MAX_SHARED_RUN:
                    problems.append(
                        f"{qid}: 質問が本文を {len(shared)} 文字そのまま写している -> 「{shared}」"
                    )
        else:
            if item.get("doc") or item.get("evidence"):
                problems.append(f"{qid}: answerable: false なのに根拠が書かれている")

        rows.append(
            {
                "qid": qid,
                "question": question,
                "gold_spans": spans,
                "answerable": answerable,
                "factoid": bool(item["factoid"]),
            }
        )

    problems.extend(check_composition(rows))
    return rows, problems


def check_composition(rows: list[dict[str, Any]]) -> list[str]:
    """件数・内訳が docs/02_evaluation.md の基準に収まっているかを見る。"""
    problems: list[str] = []
    total = len(rows)
    if not MIN_QUESTIONS <= total <= MAX_QUESTIONS:
        problems.append(f"件数 {total} が {MIN_QUESTIONS}〜{MAX_QUESTIONS} の範囲外")

    unanswerable = sum(1 for r in rows if not r["answerable"])
    ratio = unanswerable / total if total else 0.0
    if not MIN_UNANSWERABLE_RATIO <= ratio <= MAX_UNANSWERABLE_RATIO:
        problems.append(
            f"答えられない質問の割合 {ratio:.1%} が "
            f"{MIN_UNANSWERABLE_RATIO:.0%}〜{MAX_UNANSWERABLE_RATIO:.0%} の範囲外"
        )

    factoid_ratio = sum(1 for r in rows if r["factoid"]) / total if total else 0.0
    if factoid_ratio < MIN_FACTOID_RATIO:
        problems.append(
            f"固有名詞・数値を問う質問の割合 {factoid_ratio:.1%} が "
            f"{MIN_FACTOID_RATIO:.0%} を下回る"
        )
    return problems


def summarize(rows: list[dict[str, Any]], corpus: dict[str, str]) -> dict[str, Any]:
    per_doc: dict[str, int] = {doc_id: 0 for doc_id in sorted(corpus)}
    for row in rows:
        for span in row["gold_spans"]:
            per_doc[span["doc"]] += 1
    return {
        "total": len(rows),
        "answerable": sum(1 for r in rows if r["answerable"]),
        "unanswerable": sum(1 for r in rows if not r["answerable"]),
        "factoid": sum(1 for r in rows if r["factoid"]),
        "questions_per_doc": per_doc,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="評価セットを生成する")
    parser.add_argument("--check", action="store_true", help="保存せず検査だけ行う")
    args = parser.parse_args()

    draft = yaml.safe_load(DRAFT_PATH.read_text(encoding="utf-8"))
    corpus = load_corpus()
    rows, problems = build(draft, corpus)

    if problems:
        print(f"検査で {len(problems)} 件の問題が見つかった:\n")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    stats = summarize(rows, corpus)
    print(
        f"OK  {stats['total']} 件"
        f"（答えられる {stats['answerable']} / 答えられない {stats['unanswerable']}"
        f" / 固有名詞・数値 {stats['factoid']}）"
    )
    for doc_id, count in stats["questions_per_doc"].items():
        print(f"  {doc_id:<26} {count} 問")

    if args.check:
        return 0

    QA_SET_PATH.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
        newline="\n",
    )
    META_PATH.write_text(
        json.dumps(
            {
                "version": str(draft["version"]),
                "generated_at": dt.date.today().isoformat(),
                "corpus_documents": len(corpus),
                "corpus_chars": sum(len(t) for t in corpus.values()),
                **stats,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"\n{QA_SET_PATH} に書き出した。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
