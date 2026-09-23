"""パイプライン通しの確認。

小さな固定コーパスと少数の評価セットで、end-to-end が壊れていないことを見る
（docs/01_architecture.md「テスト方針」）。外部APIを使う経路も respx で
モックして通し、鍵が無い環境でも経路ごと壊れていないことを確かめる。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import yaml
from typer.testing import CliRunner

from rageval.cli import app
from rageval.config import load_experiment
from rageval.runner import run_experiment

DOCS = {
    "doc01": (
        "富士山\n\n"
        "富士山は日本の活火山である。最高地点の標高は3776.12メートルで、日本の最高峰である。\n"
        "山梨県と静岡県にまたがっており、2013年に世界文化遺産に登録された。\n"
    ),
    "doc02": (
        "和紙\n\n"
        "和紙は日本古来の紙である。楮や三椏を原料とし、繊維が長いため薄くても丈夫である。\n"
        "平安時代には流し漉きという日本独特の技法が確立された。\n"
    ),
    "doc03": (
        "光合成\n\n"
        "光合成は光エネルギーを化学エネルギーに変換する反応である。\n"
        "緑色植物では葉緑体のチラコイド膜で光化学反応が起こり、酸素が発生する。\n"
    ),
}

#: 質問と、根拠として引用する本文。オフセットは本文から機械的に求める。
QUESTIONS = [
    (
        "q001",
        "日本でいちばん高い山の高さは何メートルか。",
        "doc01",
        "最高地点の標高は3776.12メートル",
    ),
    (
        "q002",
        "和紙が薄くても丈夫なのはなぜか。",
        "doc02",
        "繊維が長いため薄くても丈夫である",
    ),
    (
        "q003",
        "緑色植物で光の反応が起こるのはどこか。",
        "doc03",
        "葉緑体のチラコイド膜で光化学反応が起こり",
    ),
]
UNANSWERABLE = ("q004", "富士山の山頂にある郵便局の営業期間はいつか。")


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for doc_id, text in DOCS.items():
        (corpus / f"{doc_id}.txt").write_text(text, encoding="utf-8", newline="\n")

    lines = []
    for qid, question, doc_id, evidence in QUESTIONS:
        start = DOCS[doc_id].index(evidence)
        lines.append(
            {
                "qid": qid,
                "question": question,
                "gold_spans": [{"doc": doc_id, "start": start, "end": start + len(evidence)}],
                "answerable": True,
                "factoid": True,
            }
        )
    lines.append(
        {
            "qid": UNANSWERABLE[0],
            "question": UNANSWERABLE[1],
            "gold_spans": [],
            "answerable": False,
            "factoid": True,
        }
    )
    qa_set = tmp_path / "qa_set.jsonl"
    qa_set.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in lines),
        encoding="utf-8",
        newline="\n",
    )
    return tmp_path


def write_experiment(workspace: Path, **overrides: Any) -> Path:
    config: dict[str, Any] = {
        "run_id": "test_run",
        "description": "パイプライン通しの確認",
        "dataset": {
            "corpus": str(workspace / "corpus"),
            "qa_set": str(workspace / "qa_set.jsonl"),
        },
        "variants": [{"chunk": {"size": 64, "overlap": 8}}, {"chunk": {"size": 128, "overlap": 8}}],
        "fixed": {
            "embedder": {"provider": "hashing", "model": "char-ngram", "dim": 512},
            "store": {"kind": "memory"},
            "retrieve": {"top_k": 3, "rerank": False},
            "generate": {"provider": "quote", "model": "top-hit-quote"},
        },
    }
    config.update(overrides)
    path = workspace / "experiment.yaml"
    path.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8", newline="\n")
    return path


@pytest.mark.integration
def test_the_whole_pipeline_runs_and_finds_the_evidence(workspace: Path) -> None:
    config = load_experiment(write_experiment(workspace))
    result = run_experiment(config, out_root=workspace / "runs")

    assert [v.label for v in result.variants] == ["chunk.size=64", "chunk.size=128"]
    assert result.axes == ["chunk.size"]
    for variant in result.variants:
        assert variant.chunks > 0
        assert variant.scores.answerable == 3
        assert variant.scores.unanswerable == 1
        assert variant.scores.recall_at_k > 0.0, "小さなコーパスで何も引けないのはおかしい"
        assert variant.scores.generation_failures == 0


@pytest.mark.integration
def test_chunk_size_changes_the_chunk_count_but_the_qa_set_still_works(workspace: Path) -> None:
    """正解根拠を文字オフセットで持っている効果を、通しで確かめる。"""
    result = run_experiment(load_experiment(write_experiment(workspace)), out_root=workspace / "r")
    small, large = result.variants
    assert small.chunks > large.chunks
    # チャンクIDは条件ごとに別物になるが、どちらの条件でも Recall は計算できている
    assert small.scores.recall_at_k > 0.0
    assert large.scores.recall_at_k > 0.0


@pytest.mark.integration
def test_outputs_are_written(workspace: Path) -> None:
    config = load_experiment(write_experiment(workspace))
    run_experiment(config, out_root=workspace / "runs")
    out = workspace / "runs" / "test_run"

    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    assert result["run_id"] == "test_run"
    assert result["axes"] == ["chunk.size"]
    assert len(result["variants"]) == 2

    rows = [
        json.loads(line)
        for line in (out / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 8, "4問 × 2条件"
    assert {row["variant"] for row in rows} == {"chunk.size=64", "chunk.size=128"}
    assert "gold_chunk_ids" in rows[0]

    snapshot = yaml.safe_load((out / "config.snapshot.yaml").read_text(encoding="utf-8"))
    assert snapshot["run_id"] == "test_run"


@pytest.mark.integration
def test_the_snapshot_keeps_the_conditions_even_if_the_yaml_changes(workspace: Path) -> None:
    experiment = write_experiment(workspace)
    run_experiment(load_experiment(experiment), out_root=workspace / "runs")
    experiment.write_text(
        experiment.read_text(encoding="utf-8").replace("size: 64", "size: 999"),
        encoding="utf-8",
    )
    snapshot = yaml.safe_load(
        (workspace / "runs" / "test_run" / "config.snapshot.yaml").read_text(encoding="utf-8")
    )
    assert snapshot["variants"][0]["chunk"]["size"] == 64


@pytest.mark.integration
def test_the_same_conditions_give_the_same_scores(workspace: Path) -> None:
    """再現性。同じ YAML と同じ評価セットから同じ結果が出ること（docs/00_requirements.md）。"""
    config = load_experiment(write_experiment(workspace))
    first = run_experiment(config, out_root=workspace / "a")
    second = run_experiment(config, out_root=workspace / "b")
    assert [v.scores.as_dict() for v in first.variants] == [
        v.scores.as_dict() for v in second.variants
    ]


@respx.mock
@pytest.mark.integration
def test_the_openai_path_runs_end_to_end_against_a_mock(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """鍵が無くても、実APIを使う経路が組み上がっていることを確かめる。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def embeddings(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [float(len(t) % 7), 1.0]}
                    for i, t in enumerate(inputs)
                ],
                "usage": {"prompt_tokens": 11},
            },
        )

    def chat(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        context = body["messages"][1]["content"]
        chunk_id = context.split("[", 1)[1].split("]", 1)[0]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {"answer": "モック", "cited_chunk_ids": [chunk_id]}
                            )
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10},
            },
        )

    respx.post("https://api.openai.com/v1/embeddings").mock(side_effect=embeddings)
    respx.post("https://api.openai.com/v1/chat/completions").mock(side_effect=chat)

    experiment = write_experiment(
        workspace,
        variants=[{"chunk": {"size": 128, "overlap": 8}}],
        fixed={
            "embedder": {"provider": "openai", "model": "text-embedding-3-small"},
            "store": {"kind": "memory"},
            "retrieve": {"top_k": 3, "rerank": False},
            "generate": {"provider": "openai", "model": "gpt-4o-mini"},
        },
    )
    result = run_experiment(load_experiment(experiment), out_root=workspace / "runs")
    variant = result.variants[0]
    assert variant.usage.requests > 0
    assert variant.usage.input_tokens > 0, "トークン数が記録されていない"
    assert variant.scores.generation_failures == 0


# ---- CLI -----------------------------------------------------------------


@pytest.mark.integration
def test_cli_run_and_report(workspace: Path) -> None:
    runner = CliRunner()
    experiment = write_experiment(workspace)
    runs = workspace / "runs"

    result = runner.invoke(app, ["run", str(experiment), "--out", str(runs)])
    assert result.exit_code == 0, result.output
    assert "chunk.size" in result.output

    report = runner.invoke(app, ["report", "--runs", str(runs)])
    assert report.exit_code == 0, report.output
    assert "test_run" in report.output
    assert "Recall@k" in report.output

    out_file = workspace / "report.md"
    written = runner.invoke(app, ["report", "--runs", str(runs), "--out", str(out_file)])
    assert written.exit_code == 0, written.output
    assert "| 条件 |" in out_file.read_text(encoding="utf-8")


@pytest.mark.integration
def test_cli_run_passes_the_regression_check(workspace: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(write_experiment(workspace)),
            "--out",
            str(workspace / "runs"),
            "--assert-recall-at-5",
            "0.0",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "回帰検証を通過" in result.output


@pytest.mark.integration
def test_cli_run_fails_when_the_score_drops(workspace: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "run",
            str(write_experiment(workspace)),
            "--out",
            str(workspace / "runs"),
            "--assert-recall-at-5",
            "1.01",
        ],
    )
    assert result.exit_code == 1
    assert "回帰検証に失敗" in result.output


@pytest.mark.integration
def test_cli_report_without_any_run(workspace: Path) -> None:
    result = CliRunner().invoke(app, ["report", "--runs", str(workspace / "empty")])
    assert result.exit_code == 1
    assert "result.json が無い" in result.output


@pytest.mark.integration
def test_cli_rejects_a_config_that_moves_two_axes(workspace: Path) -> None:
    experiment = write_experiment(
        workspace,
        variants=[
            {"chunk": {"size": 64, "overlap": 8}},
            {"chunk": {"size": 128, "overlap": 32}},
        ],
    )
    result = CliRunner().invoke(app, ["run", str(experiment), "--out", str(workspace / "runs")])
    assert result.exit_code == 1
    assert "動かす軸は1つ" in result.output
    assert "Traceback" not in result.output, "設定の誤りにトレースバックを出さない"


@pytest.mark.integration
def test_only_declared_metrics_appear_in_the_tables(workspace: Path) -> None:
    """宣言していない指標は表に出さない。

    条件によっては意味を持たない指標がある（ダミー生成での棄権率など）。
    数字が並ぶと読む人は必ず比較してしまうので、出さないことを仕組みで担保する。
    """
    runner = CliRunner()
    runs = workspace / "runs"
    experiment = write_experiment(workspace, metrics=["recall_at_k", "mrr"])

    result = runner.invoke(app, ["run", str(experiment), "--out", str(runs)])
    assert result.exit_code == 0, result.output
    assert "Recall@k" in result.output
    assert "MRR" in result.output
    assert "根拠一致率" not in result.output
    assert "棄権率" not in result.output

    report = runner.invoke(app, ["report", "--runs", str(runs)])
    assert report.exit_code == 0, report.output
    assert "Recall@k" in report.output
    assert "根拠一致率" not in report.output


@pytest.mark.integration
def test_declared_metrics_are_recorded_in_the_result(workspace: Path) -> None:
    experiment = write_experiment(workspace, metrics=["recall_at_k", "mrr"])
    run_experiment(load_experiment(experiment), out_root=workspace / "runs")
    result = json.loads(
        (workspace / "runs" / "test_run" / "result.json").read_text(encoding="utf-8")
    )
    assert result["metrics"] == ["recall_at_k", "mrr"]
    # 生の値そのものは残す。診断に使うため、表に出さないことと保存しないことは別。
    assert "citation_match" in result["variants"][0]["scores"]


@pytest.mark.integration
def test_all_experiment_files_in_the_repository_are_valid() -> None:
    """リポジトリに入っている実験ファイルが、実装と食い違っていないこと。"""
    paths = sorted(Path("experiments").rglob("*.yaml"))
    assert len(paths) >= 6
    for path in paths:
        config = load_experiment(path)
        config.check_paths()
        for variant in config.resolve():
            assert variant.pipeline.retrieve.top_k > 0
        offline = path.parent.name == "offline" or config.run_id == "ci_baseline"
        providers = {
            (v.pipeline.embedder.provider, v.pipeline.generate.provider) for v in config.resolve()
        }
        if offline:
            assert providers == {("hashing", "quote")}, f"{path} は鍵なしで回せる必要がある"
            assert set(config.metrics) <= {"recall_at_k", "mrr"}, (
                f"{path}: ダミー条件では生成側の指標を宣言しない"
            )


@pytest.mark.integration
def test_the_run_records_the_total_usage(workspace: Path) -> None:
    """実験1本でいくら使ったかを、条件ごとの内訳より先に見られること。"""
    run_experiment(load_experiment(write_experiment(workspace)), out_root=workspace / "runs")
    result = json.loads(
        (workspace / "runs" / "test_run" / "result.json").read_text(encoding="utf-8")
    )
    total = result["total_usage"]
    per_variant = [v["usage"] for v in result["variants"]]
    for key in ("requests", "input_tokens", "output_tokens", "retries"):
        assert total[key] == sum(v[key] for v in per_variant), f"{key} の合計が合わない"


@respx.mock
@pytest.mark.integration
def test_a_broken_generation_is_recorded_as_a_failure(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """生成の構造化出力が壊れたら、失敗として数える（ADR 0003）。

    黙って空の回答で埋めると、生成が壊れていることがスコアに紛れて見えなくなる。
    ここが通らないと、その仕組みが効いているか誰も知らないまま進むことになる。
    """
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    def embeddings(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        return httpx.Response(
            200,
            json={"data": [{"index": i, "embedding": [1.0, 0.0]} for i, _ in enumerate(inputs)]},
        )

    # JSON にならない文字列を返す。構造化出力の約束を破ったケース。
    respx.post("https://api.openai.com/v1/embeddings").mock(side_effect=embeddings)
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200, json={"choices": [{"message": {"content": "すみません、わかりません"}}]}
        )
    )

    experiment = write_experiment(
        workspace,
        variants=[{"chunk": {"size": 128, "overlap": 8}}],
        fixed={
            "embedder": {"provider": "openai", "model": "text-embedding-3-small"},
            "store": {"kind": "memory"},
            "retrieve": {"top_k": 3, "rerank": False},
            "generate": {"provider": "openai", "model": "gpt-4o-mini"},
        },
    )
    result = run_experiment(load_experiment(experiment), out_root=workspace / "runs")

    variant = result.variants[0]
    assert variant.scores.generation_failures == 4, "4問すべてで生成が壊れている"
    assert variant.scores.citation_match == 0.0, "壊れた出力を正解に数えてはいけない"
    assert variant.scores.recall_at_k > 0.0, "検索は動いているので、そちらは残る"

    rows = [
        json.loads(line)
        for line in (workspace / "runs" / "test_run" / "predictions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all(row["generation_failed"] for row in rows)


@pytest.mark.integration
def test_the_report_shows_the_reason_for_multiple_axes(workspace: Path) -> None:
    """複数軸を動かした実験は、その理由が表と一緒に出ること。"""
    runner = CliRunner()
    runs = workspace / "runs"
    experiment = write_experiment(
        workspace,
        variants=[
            {"chunk": {"size": 64, "overlap": 8}, "retrieve": {"top_k": 2}},
            {"chunk": {"size": 128, "overlap": 16}, "retrieve": {"top_k": 3}},
        ],
        multi_axis_reason="最良条件を組み合わせるため",
    )
    assert runner.invoke(app, ["run", str(experiment), "--out", str(runs)]).exit_code == 0
    report = runner.invoke(app, ["report", "--runs", str(runs)])
    assert "複数軸の理由: 最良条件を組み合わせるため" in report.output


@pytest.mark.integration
def test_the_report_names_the_providers(workspace: Path) -> None:
    """ダミー実装での結果と実APIでの結果を、表だけで区別できること。"""
    runner = CliRunner()
    runs = workspace / "runs"
    runner.invoke(app, ["run", str(write_experiment(workspace)), "--out", str(runs)])
    report = runner.invoke(app, ["report", "--runs", str(runs)])
    assert "埋め込み hashing / 生成 quote" in report.output


@pytest.mark.integration
def test_the_qa_set_version_is_recorded(workspace: Path) -> None:
    """評価セットの版を結果に残す。版の違う結果を混ぜないための手がかり（docs/02）。"""
    (workspace / "qa_set.meta.json").write_text(
        json.dumps({"version": "9.9", "total": 4}), encoding="utf-8"
    )
    run_experiment(load_experiment(write_experiment(workspace)), out_root=workspace / "runs")
    result = json.loads(
        (workspace / "runs" / "test_run" / "result.json").read_text(encoding="utf-8")
    )
    assert result["dataset"]["qa_set_version"] == "9.9"


# ---- 利用者の誤りに、読める形で応えるか ----------------------------------
#
# ここは「例外が上がること」ではなく「画面に何が出るか」を見る。
# 例外が上がるだけならテストは通るが、使う人が読むのは画面のほうで、
# そこに数十行のトレースバックが出れば本当の原因は埋もれる。


@pytest.mark.integration
def test_a_typo_in_a_key_names_the_key_and_the_alternatives(workspace: Path) -> None:
    experiment = write_experiment(workspace)
    experiment.write_text(
        experiment.read_text(encoding="utf-8").replace("top_k:", "top_K:"),
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["run", str(experiment), "--out", str(workspace / "runs")])
    assert result.exit_code == 1
    assert "retrieve.top_K" in result.output, "どのキーが違うかを指す"
    assert "top_k" in result.output, "正しい候補を挙げる"
    assert "Traceback" not in result.output
    assert len(result.output.splitlines()) <= 4, "長すぎる: " + result.output


@pytest.mark.integration
def test_a_missing_experiment_file_is_reported_plainly(workspace: Path) -> None:
    result = CliRunner().invoke(app, ["run", str(workspace / "nope.yaml")])
    assert result.exit_code == 1
    assert "実験ファイルが見つからない" in result.output
    assert "Traceback" not in result.output


@pytest.mark.integration
def test_a_missing_corpus_is_reported_plainly(workspace: Path) -> None:
    experiment = write_experiment(
        workspace,
        dataset={
            "corpus": str(workspace / "corpus_nope"),
            "qa_set": str(workspace / "qa_set.jsonl"),
        },
    )
    result = CliRunner().invoke(app, ["run", str(experiment), "--out", str(workspace / "runs")])
    assert result.exit_code == 1
    assert "コーパスのディレクトリがない" in result.output
    assert "Traceback" not in result.output


@pytest.mark.integration
def test_an_unknown_metric_names_what_can_be_used(workspace: Path) -> None:
    experiment = write_experiment(workspace, metrics=["recall_at_k", "bleu"])
    result = CliRunner().invoke(app, ["run", str(experiment), "--out", str(workspace / "runs")])
    assert result.exit_code == 1
    assert "未知の指標" in result.output
    assert "recall_at_k" in result.output, "使える指標を挙げる"
    assert "Traceback" not in result.output


@pytest.mark.integration
def test_a_typo_in_a_dataset_key_names_the_alternatives(workspace: Path) -> None:
    experiment = write_experiment(
        workspace,
        dataset={"corpus_path": str(workspace / "corpus"), "qa_set": str(workspace / "x.jsonl")},
    )
    result = CliRunner().invoke(app, ["run", str(experiment), "--out", str(workspace / "runs")])
    assert result.exit_code == 1
    assert "corpus, qa_set" in result.output, "使えるキーを挙げる"


# ---- 過去の測定を黙って消さないか ----------------------------------------


@pytest.mark.integration
def test_rerunning_the_same_conditions_is_allowed(workspace: Path) -> None:
    """同じ条件なら結果も同じなので、上書きして構わない。"""
    config = load_experiment(write_experiment(workspace))
    run_experiment(config, out_root=workspace / "runs")
    run_experiment(config, out_root=workspace / "runs")


@pytest.mark.integration
def test_rerunning_with_different_conditions_is_refused(workspace: Path) -> None:
    """条件が違えば別の測定。消すと過去と比べられなくなる。"""
    experiment = write_experiment(workspace)
    run_experiment(load_experiment(experiment), out_root=workspace / "runs")

    experiment.write_text(
        experiment.read_text(encoding="utf-8").replace("top_k: 3", "top_k: 2"),
        encoding="utf-8",
    )
    with pytest.raises(FileExistsError, match="条件の違う結果がすでにある"):
        run_experiment(load_experiment(experiment), out_root=workspace / "runs")


@pytest.mark.integration
def test_force_overwrites_on_purpose(workspace: Path) -> None:
    experiment = write_experiment(workspace)
    run_experiment(load_experiment(experiment), out_root=workspace / "runs")
    experiment.write_text(
        experiment.read_text(encoding="utf-8").replace("top_k: 3", "top_k: 2"),
        encoding="utf-8",
    )
    result = run_experiment(load_experiment(experiment), out_root=workspace / "runs", force=True)
    assert result.variants[0].pipeline.retrieve.top_k == 2


@pytest.mark.integration
def test_the_refusal_is_reported_before_running(workspace: Path) -> None:
    """回しきってから断られては時間の無駄なので、実行前に止まること。"""
    experiment = write_experiment(workspace)
    runner = CliRunner()
    runner.invoke(app, ["run", str(experiment), "--out", str(workspace / "runs")])
    experiment.write_text(
        experiment.read_text(encoding="utf-8").replace("top_k: 3", "top_k: 2"),
        encoding="utf-8",
    )
    result = runner.invoke(app, ["run", str(experiment), "--out", str(workspace / "runs")])
    assert result.exit_code == 1
    assert "条件の違う結果がすでにある" in result.output
    assert "--force" in result.output, "どうすればよいかを示す"
    assert "Recall@k" not in result.output, "実験を回す前に止まっていること"


@pytest.mark.integration
def test_a_broken_result_file_names_the_file(workspace: Path) -> None:
    """途中で書き込みが止まった結果が混じっても、どれが原因かを言うこと。"""
    runs = workspace / "runs"
    (runs / "broken").mkdir(parents=True)
    (runs / "broken" / "result.json").write_text("{ これは JSON では", encoding="utf-8")
    result = CliRunner().invoke(app, ["report", "--runs", str(runs)])
    assert result.exit_code == 1
    assert "結果が壊れている" in result.output
    assert "broken" in result.output, "どのディレクトリか分かること"
    assert "Traceback" not in result.output


@pytest.mark.integration
def test_a_yaml_that_is_not_a_mapping_is_reported_plainly(workspace: Path) -> None:
    """実験ファイルに配列を書いたときも、トレースバックにしない。"""
    experiment = workspace / "list.yaml"
    experiment.write_text("- これは配列\n- 辞書ではない\n", encoding="utf-8")
    result = CliRunner().invoke(app, ["run", str(experiment), "--out", str(workspace / "runs")])
    assert result.exit_code == 1
    assert "実験ファイルを読めない" in result.output
    assert "辞書でない" in result.output
    assert "Traceback" not in result.output


@pytest.mark.integration
def test_a_result_from_an_older_version_is_reported_plainly(workspace: Path) -> None:
    """項目が足りない古い result.json が混じっても、原因を言って止まること。

    形は JSON として正しいので `_load_results` は通る。表を組む段で落ちる。
    """
    runs = workspace / "runs"
    (runs / "old").mkdir(parents=True)
    (runs / "old" / "result.json").write_text('{"run_id": "old"}', encoding="utf-8")
    result = CliRunner().invoke(app, ["report", "--runs", str(runs)])
    assert result.exit_code == 1
    assert "結果の形が想定と違う" in result.output
    assert "古い実験の結果" in result.output, "どうすればよいかを示す"
    assert "Traceback" not in result.output


@pytest.mark.integration
def test_a_result_that_cannot_be_opened_names_the_path(workspace: Path) -> None:
    """読めない result.json（ここではディレクトリ）でも、どれが原因かを言うこと。

    ディレクトリを読もうとすると、Linux では IsADirectoryError、
    Windows では PermissionError が出る。どちらも OSError なので同じ経路を通る。
    """
    runs = workspace / "runs"
    (runs / "weird" / "result.json").mkdir(parents=True)
    result = CliRunner().invoke(app, ["report", "--runs", str(runs)])
    assert result.exit_code == 1
    assert "結果を読めない" in result.output
    assert "weird" in result.output
    assert "Traceback" not in result.output


@pytest.mark.integration
def test_report_refuses_to_overwrite_a_hand_written_file(workspace: Path) -> None:
    """`--out docs/04_results.md` のような指定で、書き溜めた考察を消さないこと。

    生成物には目印の行が入っている。それが無いファイルは人が書いたものとみなす。
    """
    experiment = write_experiment(workspace)
    runner = CliRunner()
    runner.invoke(app, ["run", str(experiment), "--out", str(workspace / "runs")])

    target = workspace / "handwritten.md"
    target.write_text("# 実験結果\n\n> 状態：未実施。ここに考察を書く。\n", encoding="utf-8")
    result = runner.invoke(app, ["report", "--runs", str(workspace / "runs"), "--out", str(target)])
    assert result.exit_code == 1
    assert "生成物ではない" in result.output
    assert "--force" in result.output, "どうすればよいかを示す"
    assert "考察を書く" in target.read_text(encoding="utf-8"), "中身が残っていること"


@pytest.mark.integration
def test_report_overwrites_its_own_output_without_asking(workspace: Path) -> None:
    """自分が作った表を更新するのは日常の操作なので、止めない。"""
    experiment = write_experiment(workspace)
    runner = CliRunner()
    runner.invoke(app, ["run", str(experiment), "--out", str(workspace / "runs")])
    target = workspace / "report.md"
    args = ["report", "--runs", str(workspace / "runs"), "--out", str(target)]
    assert runner.invoke(app, args).exit_code == 0
    assert runner.invoke(app, args).exit_code == 0, "2回目も通ること"


@pytest.mark.integration
def test_report_force_overwrites_on_purpose(workspace: Path) -> None:
    experiment = write_experiment(workspace)
    runner = CliRunner()
    runner.invoke(app, ["run", str(experiment), "--out", str(workspace / "runs")])
    target = workspace / "handwritten.md"
    target.write_text("消えてよい\n", encoding="utf-8")
    result = runner.invoke(
        app, ["report", "--runs", str(workspace / "runs"), "--out", str(target), "--force"]
    )
    assert result.exit_code == 0
    assert "消えてよい" not in target.read_text(encoding="utf-8")


@pytest.mark.integration
def test_report_refuses_a_write_target_it_cannot_read(workspace: Path) -> None:
    """読めないものを黙って消さない。ここではディレクトリを指定して確かめる。"""
    experiment = write_experiment(workspace)
    runner = CliRunner()
    runner.invoke(app, ["run", str(experiment), "--out", str(workspace / "runs")])
    target = workspace / "dir.md"
    target.mkdir()
    result = runner.invoke(app, ["report", "--runs", str(workspace / "runs"), "--out", str(target)])
    assert result.exit_code == 1
    assert "書き出し先を読めない" in result.output


# ---- .env の読み込み ------------------------------------------------------
#
# docs/00_requirements.md に「API キーは .env で与える」と書いてあり、
# .env.example も置いてあるのに、読む処理が無かった。
# 案内どおりに置いても効かない状態だったので、効くことをここで固定する。


@pytest.mark.integration
def test_the_key_in_a_dotenv_file_is_picked_up(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """リポジトリ直下の .env に置いた鍵が、実験を回すときに読まれること。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(workspace)
    # `.env` の中身を組み立てる必要があるので、この「鍵の名前=値」の形は避けられない。
    # detect-secrets はこの形そのものを検出するため、1件ずつ理由を書いて抑止する。
    # 値は実在しない固定の文字列で、テストの中だけで使い捨てる。
    line = "OPENAI_API_KEY=not-a-real-key-from-dotenv"  # pragma: allowlist secret
    (workspace / ".env").write_text(f"# コメント行は無視される\n{line}\n", encoding="utf-8")

    from rageval.cli import _load_env_file

    _load_env_file()
    import os

    assert os.environ["OPENAI_API_KEY"] == "not-a-real-key-from-dotenv"


@pytest.mark.integration
def test_an_existing_environment_variable_wins_over_the_dotenv_file(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CI や AWS では環境変数で渡す。ファイルが後から上書きしてはいけない。"""
    monkeypatch.setenv("OPENAI_API_KEY", "not-a-real-key-from-environment")
    monkeypatch.chdir(workspace)
    # 上と同じ理由。実在しない値で、テストの中だけで使い捨てる。
    (workspace / ".env").write_text(
        "OPENAI_API_KEY=not-a-real-key-from-dotenv\n",  # pragma: allowlist secret
        encoding="utf-8",
    )

    from rageval.cli import _load_env_file

    _load_env_file()
    import os

    assert os.environ["OPENAI_API_KEY"] == "not-a-real-key-from-environment"


@pytest.mark.integration
def test_no_dotenv_file_is_fine(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """鍵を使わない条件では .env が無いのが普通。落ちないこと。"""
    monkeypatch.chdir(workspace)
    assert not (workspace / ".env").exists()

    from rageval.cli import _load_env_file

    _load_env_file()


@pytest.mark.integration
def test_a_missing_key_explains_where_to_put_it(workspace: Path) -> None:
    """鍵が無いときの案内が、実際に効く置き方を指していること。"""
    from rageval.external import read_api_key

    with pytest.raises(ValueError) as excinfo:
        read_api_key("RAGEVAL_TEST_ABSENT_KEY", fallback_provider="hashing")
    message = str(excinfo.value)
    assert ".env" in message
    assert "環境変数" in message
    assert "hashing" in message, "鍵なしで動かす道も示す"


@pytest.mark.integration
@respx.mock
def test_a_rejected_key_is_reported_plainly(
    workspace: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """鍵を用意した直後にいちばん踏みやすい失敗。トレースバックにしない。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-wrong")
    respx.post("https://api.openai.com/v1/embeddings").mock(
        return_value=httpx.Response(401, json={"error": {"message": "Incorrect API key"}})
    )
    experiment = write_experiment(
        workspace,
        variants=[{"chunk": {"size": 128, "overlap": 8}}],
        fixed={
            "embedder": {"provider": "openai", "model": "text-embedding-3-small"},
            "store": {"kind": "memory"},
            "retrieve": {"top_k": 3, "rerank": False},
            "generate": {"provider": "quote", "model": "top-hit-quote"},
        },
    )
    result = CliRunner().invoke(app, ["run", str(experiment), "--out", str(workspace / "runs")])
    assert result.exit_code == 1
    assert "外部APIの呼び出しに失敗した" in result.output
    assert "作り直す" in result.output, "次の一手を示す"
    assert "Traceback" not in result.output
    assert "sk-wrong" not in result.output, "鍵そのものを出力に載せない"
