"""実験1回ぶんの流れを組み立てる。

`config` が読んだ条件を受け取り、条件ごとに索引フェーズと評価フェーズを回して、
`runs/<run_id>/` に結果を残す（docs/01_architecture.md「全体像」「出力」）。

`config.snapshot.yaml` を残すのは、実験ファイルを後から編集しても
過去の結果の条件が変わらないようにするため。再現性の担保に要る。
"""

from __future__ import annotations

import contextlib
import datetime as dt
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rageval.config import ExperimentConfig, PipelineConfig
from rageval.embed import build_embedder
from rageval.evaluate import (
    Prediction,
    QaItem,
    Scores,
    gold_chunk_ids,
    load_qa_set,
    score,
)
from rageval.external import (
    ApiCaller,
    BedrockCaller,
    RateLimit,
    Usage,
    build_bedrock_client,
    build_http_client,
    read_api_key,
)
from rageval.generate import Answer, GenerationError, build_generator
from rageval.ingest import Document, FixedSizeChunker, chunk_documents, load_corpus
from rageval.retrieve import Retriever, build_reranker
from rageval.store import build_store


@dataclass(frozen=True)
class VariantResult:
    """条件1つぶんの結果。"""

    label: str
    pipeline: PipelineConfig
    scores: Scores
    usage: Usage
    chunks: int
    index_seconds: float
    query_seconds: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "config": self.pipeline.model_dump(mode="json"),
            "scores": self.scores.as_dict(),
            "usage": self.usage.as_dict(),
            "index": {"chunks": self.chunks, "seconds": round(self.index_seconds, 3)},
            "query": {"mean_seconds": round(self.query_seconds, 4)},
        }


@dataclass(frozen=True)
class RunResult:
    """実験1本ぶんの結果。`runs/<run_id>/result.json` の中身。"""

    run_id: str
    description: str
    generated_at: str
    axes: list[str]
    multi_axis_reason: str | None
    dataset: dict[str, Any]
    variants: list[VariantResult]
    #: 条件ごとの使用量を足したもの。実験1本でいくら使ったかは、
    #: 条件ごとの内訳より先に知りたい数字なので、run のレベルにも出す
    #: （docs/03_experiment_plan.md「記録する項目」）。
    total_usage: Usage = field(default_factory=Usage)
    #: 実験ファイルが宣言した指標。表に載せるのはこれだけにする。
    #: 条件によっては意味を持たない指標があり（たとえばダミー生成での棄権率）、
    #: 数字が並ぶと読む人は必ず比較してしまうため、宣言していないものは出さない。
    metrics: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "description": self.description,
            "generated_at": self.generated_at,
            "axes": self.axes,
            "multi_axis_reason": self.multi_axis_reason,
            "metrics": self.metrics,
            "total_usage": self.total_usage.as_dict(),
            "dataset": self.dataset,
            "variants": [variant.as_dict() for variant in self.variants],
        }


def _dataset_info(config: ExperimentConfig, items: list[QaItem]) -> dict[str, Any]:
    info: dict[str, Any] = {
        "corpus": str(config.dataset.corpus),
        "qa_set": str(config.dataset.qa_set),
        "questions": len(items),
        "answerable": sum(1 for item in items if item.answerable),
        "unanswerable": sum(1 for item in items if not item.answerable),
    }
    meta_path = config.dataset.qa_set.with_suffix(".meta.json")
    if meta_path.is_file():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        info["qa_set_version"] = meta.get("version")
    return info


def _uses(config: ExperimentConfig, provider: str) -> bool:
    return any(
        variant.pipeline.embedder.provider == provider
        or variant.pipeline.generate.provider == provider
        for variant in config.resolve()
    )


def _bedrock_region(config: ExperimentConfig) -> str:
    """実験ファイルが指したリージョン。条件ごとに違っていたら断る。

    条件によって呼び先が変わると、比べているものが「モデルの差」なのか
    「リージョンの差」なのか分からなくなる。
    """
    regions = set()
    for variant in config.resolve():
        if variant.pipeline.embedder.provider == "bedrock":
            regions.add(variant.pipeline.embedder.region)
        if variant.pipeline.generate.provider == "bedrock":
            regions.add(variant.pipeline.generate.region)
    if len(regions) > 1:
        raise ValueError(
            f"bedrock のリージョンが条件ごとに違う: {sorted(regions)}。"
            "呼び先が変わると、モデルの差とリージョンの差が混ざる"
        )
    return regions.pop() if regions else "ap-northeast-1"


def _build_caller() -> ApiCaller:
    # 鍵は読んだ時点で検証する。改行が混ざったままヘッダに載せない。
    api_key = read_api_key("OPENAI_API_KEY", fallback_provider="hashing")
    client = build_http_client(
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
    )
    # 既定は控えめ。上げるときは相手のレート上限を確認してからにする。
    return ApiCaller(client=client, rate_limit=RateLimit(requests_per_second=5.0, burst=5))


def _build_bedrock_caller(region: str) -> BedrockCaller:
    """Bedrock の呼び出し口。**鍵は読まない。**

    認証は手元なら AWS の認証情報、AWS 上ならタスクロールが持つ。
    `.env` に秘密を置く必要がないのが、この経路を選んだ理由の1つ。
    """
    return BedrockCaller(
        client=build_bedrock_client(region=region),
        region=region,
        rate_limit=RateLimit(requests_per_second=5.0, burst=5),
    )


def _refuse_to_clobber(config: ExperimentConfig, out_root: Path) -> None:
    """条件の違う過去の結果を、黙って消さない。

    同じ run_id で回し直すと出力先が同じになる。条件が同じなら結果も同じ
    （この道具は決定的なので）ため上書きして構わないが、**条件が違えば
    それは別の測定**で、消すと過去と比べられなくなる。
    比較できるようにするための道具が、比較の材料を消してはいけない。

    実行前に見る。回しきってから断られては時間の無駄なので。
    """
    snapshot_path = out_root / config.run_id / "config.snapshot.yaml"
    if not snapshot_path.is_file():
        return
    previous = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    if previous == config.snapshot():
        return
    raise FileExistsError(
        f"{out_root / config.run_id} に、条件の違う結果がすでにある。"
        "上書きすると過去の測定と比べられなくなる。"
        "run_id を変えるか、--out で別の場所に出すか、"
        "消えてよいなら --force を付けること"
    )


def run_experiment(
    config: ExperimentConfig,
    *,
    out_root: Path = Path("runs"),
    caller: ApiCaller | None = None,
    bedrock: BedrockCaller | None = None,
    force: bool = False,
) -> RunResult:
    """条件を順に回して結果を書き出す。"""
    config.check_paths()
    if not force:
        _refuse_to_clobber(config, out_root)
    documents = load_corpus(config.dataset.corpus)
    items = load_qa_set(config.dataset.qa_set)
    variants = config.resolve()

    with contextlib.ExitStack() as stack:
        if caller is None and _uses(config, "openai"):
            caller = _build_caller()
            stack.callback(caller.client.close)
        if bedrock is None and _uses(config, "bedrock"):
            bedrock = _build_bedrock_caller(_bedrock_region(config))

        results: list[VariantResult] = []
        prediction_rows: list[dict[str, Any]] = []
        total_usage = Usage()
        for variant in variants:
            result, rows = _run_variant(
                variant.label, variant.pipeline, documents, items, caller, bedrock
            )
            results.append(result)
            prediction_rows.extend(rows)
            total_usage.merge(result.usage)

    run = RunResult(
        run_id=config.run_id,
        description=config.description,
        generated_at=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        axes=config.axes(),
        multi_axis_reason=config.multi_axis_reason,
        dataset=_dataset_info(config, items),
        variants=results,
        total_usage=total_usage,
        metrics=list(config.metrics),
    )
    _write_outputs(config, run, prediction_rows, out_root)
    return run


def _run_variant(
    label: str,
    pipeline: PipelineConfig,
    documents: list[Document],
    items: list[QaItem],
    caller: ApiCaller | None,
    bedrock: BedrockCaller | None = None,
) -> tuple[VariantResult, list[dict[str, Any]]]:
    usage = Usage()
    # 同じ Usage を両方に持たせる。提供元を混ぜた条件でも合計が1つに集まる。
    if caller is not None:
        caller.usage = usage
    if bedrock is not None:
        bedrock.usage = usage

    # 索引フェーズ
    index_started = time.perf_counter()
    chunker = FixedSizeChunker.from_config(pipeline.chunk)
    chunks = chunk_documents(documents, chunker)
    embedder = build_embedder(pipeline.embedder, caller=caller, bedrock=bedrock)
    store = build_store(pipeline.store)
    store.upsert(chunks, embedder.embed([chunk.text for chunk in chunks]))
    index_seconds = time.perf_counter() - index_started

    # 評価フェーズ
    retriever = Retriever(
        embedder=embedder,
        store=store,
        config=pipeline.retrieve,
        reranker=build_reranker(pipeline.retrieve),
    )
    generator = build_generator(pipeline.generate, caller=caller, bedrock=bedrock)

    predictions: dict[str, Prediction] = {}
    gold: dict[str, set[str]] = {}
    rows: list[dict[str, Any]] = []
    query_started = time.perf_counter()
    for item in items:
        started = time.perf_counter()
        hits = retriever.retrieve(item.question)
        failed = False
        try:
            answer = generator(item.question, hits)
        except GenerationError:
            # 構造化出力が壊れた場合は失敗として記録する（ADR 0003）。
            # 黙って埋めると、生成が壊れていることがスコアに紛れて見えなくなる。
            answer = Answer(answer="", cited_chunk_ids=[], abstained=False)
            failed = True

        prediction = Prediction(
            qid=item.qid,
            retrieved_chunk_ids=[hit.chunk.chunk_id for hit in hits],
            answer=answer,
            generation_failed=failed,
            seconds=time.perf_counter() - started,
        )
        predictions[item.qid] = prediction
        gold[item.qid] = gold_chunk_ids(item, chunks)
        rows.append(
            {
                "variant": label,
                "question": item.question,
                "answerable": item.answerable,
                "gold_chunk_ids": sorted(gold[item.qid]),
                **prediction.as_dict(),
            }
        )
    query_seconds = (time.perf_counter() - query_started) / len(items)

    scores = score(items, predictions, gold, pipeline.retrieve.top_k)
    return (
        VariantResult(
            label=label,
            pipeline=pipeline,
            scores=scores,
            usage=usage,
            chunks=len(chunks),
            index_seconds=index_seconds,
            query_seconds=query_seconds,
        ),
        rows,
    )


def _write_outputs(
    config: ExperimentConfig,
    run: RunResult,
    prediction_rows: list[dict[str, Any]],
    out_root: Path,
) -> Path:
    out_dir = out_root / config.run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "result.json").write_text(
        json.dumps(run.as_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (out_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in prediction_rows),
        encoding="utf-8",
        newline="\n",
    )
    (out_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(config.snapshot(), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    return out_dir
