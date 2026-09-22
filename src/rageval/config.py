"""実験条件の型定義と検証。

YAML を読んで `ExperimentConfig` にし、**実行前に**条件の矛盾を弾く。
ここで落とせなかった誤りは、実験を1本回しきったあとに気づくことになる。

`variants` と `fixed` を分けているのは、「何を動かして何を固定したか」が
実験ファイルを見るだけで分かるようにするため（docs/01_architecture.md）。
`fixed` を土台に `variants` を深くマージして、条件ごとの完全な `PipelineConfig` を作る。

さらに「1実験につき動かす軸は1つ」（CLAUDE.md）を検証する。軸は section 単位ではなく
**葉のフィールド単位**で数える。`retrieve` セクションごと上書きしていても、
実際に値が変わっているのが `rerank` だけなら軸は1つと数える。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

#: パイプラインの構成要素。`fixed` と `variants` に書けるトップレベルのキー。
SECTIONS = ("chunk", "embedder", "store", "retrieve", "generate")

#: 算出できる指標。`metrics` に書けるもの。
KNOWN_METRICS = ("recall_at_k", "mrr", "citation_match", "abstention")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ChunkConfig(_Section):
    """チャンク分割の条件。"""

    size: int = Field(gt=0, description="1チャンクの文字数")
    overlap: int = Field(ge=0, description="隣接チャンクが重なる文字数")
    phase: int = Field(
        default=0,
        ge=0,
        description=(
            "チャンク境界をずらす文字数。最初のチャンクだけ短くして、"
            "以降の境界を全体にずらす。どこで切るかは本来どうでもよいはずの自由度なので、"
            "ここを振ったときのスコアの散らばりが、この測定系のノイズ幅そのものになる"
        ),
    )

    @model_validator(mode="after")
    def _overlap_must_be_smaller(self) -> ChunkConfig:
        if self.overlap >= self.size:
            raise ValueError(
                f"overlap({self.overlap}) は size({self.size}) より小さいこと。"
                "同じ以上だと分割が進まない"
            )
        if self.phase >= self.size:
            raise ValueError(f"phase({self.phase}) は size({self.size}) より小さいこと")
        if self.size - self.phase <= self.overlap:
            raise ValueError(
                f"size({self.size}) - phase({self.phase}) は overlap({self.overlap}) より"
                "大きいこと。最初のチャンクが重複幅に飲まれて分割が進まない"
            )
        return self


class EmbedderConfig(_Section):
    """埋め込みプロバイダの条件。"""

    provider: Literal["hashing", "openai"]
    model: str
    dim: int = Field(default=512, gt=0, description="hashing のときのベクトル次元")
    batch_size: int = Field(default=64, gt=0)


class StoreConfig(_Section):
    """Vector DB の条件。"""

    kind: Literal["memory", "chroma"] = "memory"


class RetrieveConfig(_Section):
    """検索の条件。"""

    top_k: int = Field(gt=0, description="評価の対象にする上位件数")
    rerank: bool = False
    candidates: int | None = Field(
        default=None,
        gt=0,
        description=(
            "再ランクにかける候補数。None なら top_k と同じで、並べ替えるだけで集合は変わらない"
        ),
    )

    @model_validator(mode="after")
    def _candidates_must_cover_top_k(self) -> RetrieveConfig:
        if self.candidates is not None and self.candidates < self.top_k:
            raise ValueError(f"candidates({self.candidates}) は top_k({self.top_k}) 以上であること")
        if self.candidates is not None and not self.rerank:
            raise ValueError("candidates は rerank: true のときだけ意味を持つ")
        return self


class GenerateConfig(_Section):
    """生成の条件。

    生成プロバイダは Protocol を切らずに実装を足す方針（ADR 0002）なので、
    ここでも `provider` の文字列で切り替える。
    """

    provider: Literal["quote", "openai"]
    model: str
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    seed: int = 42
    max_answer_chars: int = Field(default=160, gt=0)


class PipelineConfig(_Section):
    """`fixed` と1つの `variant` をマージした、実行可能な完全な条件。"""

    chunk: ChunkConfig
    embedder: EmbedderConfig
    store: StoreConfig
    retrieve: RetrieveConfig
    generate: GenerateConfig


class DatasetConfig(_Section):
    corpus: Path
    qa_set: Path


@dataclass(frozen=True)
class ResolvedVariant:
    """実行する条件1つぶん。"""

    label: str
    pipeline: PipelineConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """override を base に深くマージする。base は書き換えない。"""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        current = merged.get(key)
        if isinstance(current, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(current, value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    """入れ子の dict を `retrieve.top_k` のような葉のパスに潰す。"""
    if not isinstance(value, dict):
        return {prefix: value}
    flat: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        flat.update(_flatten(child, path))
    return flat


class ExperimentConfig(BaseModel):
    """実験ファイル1本ぶん。"""

    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(min_length=1)
    description: str = ""
    dataset: DatasetConfig
    fixed: dict[str, Any] = Field(default_factory=dict)
    variants: list[dict[str, Any]] = Field(min_length=1)
    metrics: list[str] = Field(default_factory=lambda: list(KNOWN_METRICS))
    multi_axis_reason: str | None = Field(
        default=None,
        description="軸を複数動かすときに理由を書く。書かないと検証で落ちる",
    )

    @model_validator(mode="after")
    def _validate(self) -> ExperimentConfig:
        self._check_section_names()
        self._check_metrics()
        self._check_axes()
        self._check_variants_are_distinct()
        return self

    def _check_variants_are_distinct(self) -> None:
        """条件どうしが区別できること。

        ここで `resolve()` を呼ぶので、値の誤り（overlap >= size など）も
        実験ファイルを読んだ時点で出る。実行してから気づくより早い。

        見るのは2つ。

        - ラベルの重複: predictions.jsonl でどの条件の行か分からなくなる
        - 条件そのものの重複: 同じ条件を2回回しても情報が増えず、表が読みにくくなるだけ
        """
        resolved = self.resolve()

        labels = [variant.label for variant in resolved]
        duplicated = sorted({label for label in labels if labels.count(label) > 1})
        if duplicated:
            raise ValueError(
                f"条件のラベルが重複している: {duplicated}。label を明示して区別すること"
            )

        seen: dict[str, str] = {}
        for variant in resolved:
            key = variant.pipeline.model_dump_json()
            if key in seen:
                raise ValueError(
                    f"同じ条件が2回書かれている:「{seen[key]}」と「{variant.label}」。"
                    "2回回しても結果は同じなので、どちらかを消すか、条件を変えること"
                )
            seen[key] = variant.label

    def _check_section_names(self) -> None:
        blocks: list[tuple[str, dict[str, Any]]] = [("fixed", self.fixed)]
        blocks += [(f"variants[{i}]", v) for i, v in enumerate(self.variants)]
        for name, block in blocks:
            unknown = set(block) - set(SECTIONS) - {"label"}
            if unknown:
                raise ValueError(
                    f"{name} に未知のキーがある: {sorted(unknown)}。"
                    f"書けるのは {list(SECTIONS)} と label のみ"
                )

    def _check_metrics(self) -> None:
        unknown = set(self.metrics) - set(KNOWN_METRICS)
        if unknown:
            raise ValueError(f"未知の指標: {sorted(unknown)}。算出できるのは {list(KNOWN_METRICS)}")

    def _check_axes(self) -> None:
        axes = self.axes()
        if len(axes) > 1 and not self.multi_axis_reason:
            raise ValueError(
                "1実験につき動かす軸は1つ。"
                f"いま動いているのは {axes}。"
                "実験を分けるか、複数軸にする理由を multi_axis_reason に書くこと"
            )

    def resolve(self) -> list[ResolvedVariant]:
        """`fixed` に各 `variant` を重ねて、実行可能な条件の一覧を作る。"""
        resolved: list[ResolvedVariant] = []
        for index, variant in enumerate(self.variants):
            override = {k: v for k, v in variant.items() if k != "label"}
            merged = _deep_merge(self.fixed, override)
            missing = [s for s in SECTIONS if s not in merged]
            if missing:
                raise ValueError(
                    f"variants[{index}] を fixed に重ねても {missing} が決まらない。"
                    "条件は fixed か variants のどちらかで必ず明示すること"
                )
            pipeline = PipelineConfig.model_validate(merged)
            label = variant.get("label") or self._derive_label(index, merged)
            resolved.append(ResolvedVariant(label=str(label), pipeline=pipeline))
        return resolved

    def axes(self) -> list[str]:
        """条件の間で実際に値が変わっている葉のパス。これが「動かした軸」。"""
        flattened = [
            _flatten(_deep_merge(self.fixed, {k: v for k, v in variant.items() if k != "label"}))
            for variant in self.variants
        ]
        if len(flattened) < 2:
            return []
        keys = sorted(set().union(*(set(f) for f in flattened)))
        return [key for key in keys if len({repr(f.get(key)) for f in flattened}) > 1]

    def _derive_label(self, index: int, merged: dict[str, Any]) -> str:
        axes = self.axes()
        if not axes:
            return f"variant{index + 1}"
        flat = _flatten(merged)
        return " / ".join(f"{axis}={flat.get(axis)}" for axis in axes)

    def check_paths(self) -> None:
        """データセットが実在するかを、実験を回す前に確かめる。"""
        if not self.dataset.corpus.is_dir():
            raise ValueError(f"コーパスのディレクトリがない: {self.dataset.corpus}")
        if not self.dataset.qa_set.is_file():
            raise ValueError(f"評価セットのファイルがない: {self.dataset.qa_set}")

    def snapshot(self) -> dict[str, Any]:
        """config.snapshot.yaml に残す内容。実験ファイルを後から編集しても結果は動かない。"""
        return self.model_dump(mode="json")


#: セクションごとに書けるキー。綴りを間違えたときの案内に使う。
ALLOWED_KEYS: dict[str, tuple[str, ...]] = {
    "chunk": tuple(ChunkConfig.model_fields),
    "embedder": tuple(EmbedderConfig.model_fields),
    "store": tuple(StoreConfig.model_fields),
    "retrieve": tuple(RetrieveConfig.model_fields),
    "generate": tuple(GenerateConfig.model_fields),
    "dataset": tuple(DatasetConfig.model_fields),
}


def load_experiment(path: Path) -> ExperimentConfig:
    """実験ファイルを読んで検証する。"""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"実験ファイルの中身が辞書でない: {path}")
    return ExperimentConfig.model_validate(raw)
