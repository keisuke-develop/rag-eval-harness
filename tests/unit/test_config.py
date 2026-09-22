"""実験条件の検証。

ここで落とせなかった誤りは、実験を1本回しきったあとに気づくことになる。
とくに「動かす軸が1つか」は、結果の表を読む人が条件の差分を推測せずに
済むかどうかを左右するので、機械で守らせる。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from rageval.config import ExperimentConfig, load_experiment

FIXED: dict[str, Any] = {
    "chunk": {"size": 512, "overlap": 64},
    "embedder": {"provider": "hashing", "model": "char-ngram", "dim": 128},
    "store": {"kind": "memory"},
    "retrieve": {"top_k": 5, "rerank": False},
    "generate": {"provider": "quote", "model": "top-hit-quote"},
}


def make(**overrides: Any) -> ExperimentConfig:
    base: dict[str, Any] = {
        "run_id": "t",
        "dataset": {"corpus": "datasets/corpus", "qa_set": "datasets/qa_set.jsonl"},
        "fixed": FIXED,
        "variants": [{}],
    }
    base.update(overrides)
    return ExperimentConfig.model_validate(base)


def test_variant_is_merged_onto_fixed() -> None:
    config = make(variants=[{"chunk": {"size": 256, "overlap": 32}}])
    pipeline = config.resolve()[0].pipeline
    assert pipeline.chunk.size == 256
    # 上書きしなかった条件は fixed のまま残る
    assert pipeline.retrieve.top_k == 5


def test_merge_is_deep_so_a_variant_need_not_restate_the_whole_section() -> None:
    config = make(variants=[{"retrieve": {"rerank": True}}])
    pipeline = config.resolve()[0].pipeline
    assert pipeline.retrieve.rerank is True
    assert pipeline.retrieve.top_k == 5


def test_single_axis_is_allowed() -> None:
    config = make(
        variants=[
            {"chunk": {"size": 256, "overlap": 64}},
            {"chunk": {"size": 512, "overlap": 64}},
        ]
    )
    assert config.axes() == ["chunk.size"]


def test_two_axes_without_a_reason_is_rejected() -> None:
    with pytest.raises(ValidationError, match="動かす軸は1つ"):
        make(
            variants=[
                {"chunk": {"size": 256, "overlap": 32}},
                {"chunk": {"size": 512, "overlap": 64}},
            ]
        )


def test_two_axes_with_a_stated_reason_is_allowed() -> None:
    config = make(
        variants=[
            {"chunk": {"size": 256, "overlap": 32}},
            {"chunk": {"size": 512, "overlap": 64}},
        ],
        multi_axis_reason="実験006で最良条件を組み合わせるため",
    )
    assert config.axes() == ["chunk.overlap", "chunk.size"]


def test_axis_is_counted_per_leaf_not_per_section() -> None:
    """セクションごと書いていても、実際に動いている値だけを軸と数える。"""
    config = make(
        variants=[
            {"retrieve": {"top_k": 5, "rerank": False}},
            {"retrieve": {"top_k": 5, "rerank": True}},
        ]
    )
    assert config.axes() == ["retrieve.rerank"]


def test_label_is_derived_from_the_axis() -> None:
    config = make(
        variants=[{"retrieve": {"top_k": 3}}, {"retrieve": {"top_k": 10}}],
    )
    assert [v.label for v in config.resolve()] == ["retrieve.top_k=3", "retrieve.top_k=10"]


def test_explicit_label_wins() -> None:
    config = make(variants=[{"label": "baseline"}])
    assert config.resolve()[0].label == "baseline"


def test_unknown_section_is_rejected() -> None:
    with pytest.raises(ValidationError, match="未知のキー"):
        make(variants=[{"chunking": {"size": 256}}])


def test_unknown_leaf_key_is_rejected() -> None:
    # 値の誤りは読み込んだ時点で出る（条件の重複検査が resolve を通すため）
    with pytest.raises(ValidationError):
        make(variants=[{"retrieve": {"topk": 3}}])


def test_unknown_metric_is_rejected() -> None:
    with pytest.raises(ValidationError, match="未知の指標"):
        make(metrics=["recall_at_k", "bleu"])


def test_a_missing_section_is_reported_at_load_time() -> None:
    with pytest.raises(ValidationError, match="generate"):
        make(fixed={k: v for k, v in FIXED.items() if k != "generate"})


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValidationError, match="より小さいこと"):
        make(variants=[{"chunk": {"size": 128, "overlap": 128}}])


def test_candidates_must_cover_top_k() -> None:
    with pytest.raises(ValidationError, match="top_k"):
        make(variants=[{"retrieve": {"top_k": 5, "rerank": True, "candidates": 3}}])


def test_candidates_without_rerank_is_rejected() -> None:
    with pytest.raises(ValidationError, match="rerank"):
        make(variants=[{"retrieve": {"top_k": 5, "rerank": False, "candidates": 20}}])


def test_check_paths_reports_a_missing_corpus(tmp_path: Path) -> None:
    config = make()
    config = config.model_copy(
        update={
            "dataset": config.dataset.model_copy(update={"corpus": tmp_path / "nope"}),
        }
    )
    with pytest.raises(ValueError, match="コーパスのディレクトリがない"):
        config.check_paths()


def test_ci_baseline_experiment_file_is_valid() -> None:
    """リポジトリに入っている実験ファイルが、実装と食い違っていないこと。"""
    config = load_experiment(Path("experiments/ci_baseline.yaml"))
    config.check_paths()
    pipeline = config.resolve()[0].pipeline
    assert pipeline.embedder.provider == "hashing", "CI で外部APIを叩いてはいけない"
    assert pipeline.generate.provider == "quote", "CI で外部APIを叩いてはいけない"


def test_snapshot_round_trips() -> None:
    config = make()
    assert ExperimentConfig.model_validate(config.snapshot()).run_id == config.run_id


def test_duplicated_labels_are_rejected() -> None:
    """ラベルが重複すると predictions.jsonl で条件を区別できなくなる。"""
    with pytest.raises(ValidationError, match="ラベルが重複"):
        make(variants=[{"label": "同じ"}, {"label": "同じ"}])


def test_the_same_condition_written_twice_is_rejected() -> None:
    """条件が同じなら自動ラベルは別々に付くので、条件そのものを比べて弾く。"""
    with pytest.raises(ValidationError, match="同じ条件が2回"):
        make(variants=[{"retrieve": {"top_k": 3}}, {"retrieve": {"top_k": 3}}])


def test_check_paths_reports_a_missing_qa_set(tmp_path: Path) -> None:
    config = make()
    config = config.model_copy(
        update={"dataset": config.dataset.model_copy(update={"qa_set": tmp_path / "nope.jsonl"})}
    )
    with pytest.raises(ValueError, match="評価セットのファイルがない"):
        config.check_paths()


def test_a_yaml_that_is_not_a_mapping_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.yaml"
    path.write_text("- これは配列\n- 辞書ではない\n", encoding="utf-8")
    with pytest.raises(ValueError, match="辞書でない"):
        load_experiment(path)
