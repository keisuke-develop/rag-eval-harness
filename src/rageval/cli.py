"""`run` と `report` の2コマンド。

`run`
    実験ファイルを検証して回し、`runs/<run_id>/` に結果を残す。
    `--assert-recall-at-5` を付けると、下回ったときに終了コード1で落ちる。
    CI の回帰検証はこれを使う（docs/01_architecture.md）。

`report`
    `runs/` に溜まった結果を読み、条件 × スコアの表を Markdown で出す。
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
from dotenv import load_dotenv
from pydantic import ValidationError
from pydantic_core import ErrorDetails

from rageval.config import ALLOWED_KEYS, ExperimentConfig, load_experiment
from rageval.external import ExternalCallError
from rageval.runner import RunResult, run_experiment


def _load_env_file() -> None:
    """カレントディレクトリの `.env` を読む。

    `docs/00_requirements.md` に「API キーは `.env` で与える」と書いてあり、
    `.env.example` も置いてあるのに、**読む処理が無かった**。
    案内どおりに置いても効かない状態だったので入れた。

    - 探すのはカレントディレクトリだけ。実験ファイルのデータセットのパスも
      リポジトリ相対なので、この道具はもともとリポジトリ直下から実行する前提。
      探索範囲を広げると、どの `.env` が効いたのかが分からなくなる。
    - すでに環境変数があればそちらを優先する（`override=False`）。
      CI や AWS では環境変数で渡すので、ファイルが後から上書きしてはいけない。
    - ライブラリ側では読まない。副作用を持つのは入口だけにする。
    """
    load_dotenv(Path.cwd() / ".env", override=False)


app = typer.Typer(
    add_completion=False,
    help="RAG の設定を変えたときの精度変化を、同じ評価セットで比較する道具",
    # 例外が出たときにローカル変数を表示しない（typer の既定は表示する）。
    # 出力は CloudWatch などの共有先に流れることがあり、そこに実行時の変数が
    # 丸ごと載ると、鍵やコーパス本文が意図せず残る。原因の特定には
    # 例外の型・メッセージ・スタックで足りる。
    pretty_exceptions_show_locals=False,
)

#: 指標の表示名。実験ファイルが宣言した指標だけを、この順で表に出す。
_SCORE_LABELS = {
    "recall_at_k": "Recall@k",
    "mrr": "MRR",
    "citation_match": "根拠一致率",
    "abstention": "棄権率",
}


@app.callback()
def main() -> None:
    """どのコマンドより先に走る。鍵を読めるようにしてから本体に入る。"""
    _load_env_file()


def _columns(metrics: list[str]) -> list[tuple[str, str]]:
    """宣言された指標だけを列にする。宣言が空なら全部出す（後方互換）。"""
    if not metrics:
        return list(_SCORE_LABELS.items())
    return [(key, label) for key, label in _SCORE_LABELS.items() if key in metrics]


def _fail(headline: str, *details: str) -> NoReturn:
    """利用者の誤りを、トレースバックなしで伝えて終わる。

    トレースバックは実装の不具合を追うためのもので、設定の書き間違いに出すものではない。
    出すと、本当のメッセージが数十行のフレームに埋もれて読まれなくなる。
    """
    typer.echo(f"エラー: {headline}", err=True)
    for line in details:
        typer.echo(f"  {line}", err=True)
    raise typer.Exit(code=1)


def _describe(error: ErrorDetails) -> str:
    """pydantic の検証エラー1件を、日本語で読める1行にする。"""
    location = ".".join(str(part) for part in error["loc"])
    message = str(error.get("msg", ""))
    # 自前の検証で投げた ValueError は "Value error, " が前に付く。それは剥がす。
    message = message.removeprefix("Value error, ")

    if error.get("type") == "extra_forbidden":
        section = str(error["loc"][0]) if error["loc"] else ""
        allowed = ALLOWED_KEYS.get(section)
        hint = f"書けるのは {', '.join(allowed)}" if allowed else "綴りを確認すること"
        return f"{location}: 指定できないキー。{hint}"
    return f"{location}: {message}" if location else message


#: 応答の本文に出る識別子ごとの案内。状態コードより先に見る。
#: 429 は「残高が無い」と「速すぎる」の両方で返るが、**やることは正反対**なので
#: まとめて案内すると役に立たない。本文で区別できるときは区別する。
_API_ADVICE_BY_BODY = {
    "insufficient_quota": (
        "残高が無い。platform.openai.com の Billing でクレジットを追加すること（待っても直らない）"
    ),
    "credit_balance_exhausted": (
        "残高が無い。platform.openai.com の Billing でクレジットを追加すること（待っても直らない）"
    ),
    "rate_limit_exceeded": "呼び出しが速すぎる。retry の間隔を広げるか、条件数を減らすこと",
    "model_not_found": (
        "モデルが見つからないか、鍵に使う権限が無い。"
        "実験ファイルの embedder.model / generate.model と、鍵の権限を確認すること"
    ),
    "invalid_api_key": "鍵が違う。platform.openai.com の API keys で作り直すこと",
}

#: 本文で区別がつかないときの、状態コードごとの案内。
_API_ADVICE_BY_STATUS = {
    "401": "鍵が違うか失効している。platform.openai.com の API keys で作り直すこと",
    "403": "鍵にこのモデルを使う権限が無い。組織やプロジェクトの設定を確認すること",
    "429": "残高が無いか、呼び出しが速すぎる。まず Billing を確認すること",
    "404": "モデル名が違う。実験ファイルの embedder.model / generate.model を確認すること",
}


def _advise(exc: Exception) -> list[str]:
    """失敗の内容に、次の一手を添える。鍵そのものは載せない。"""
    message = str(exc)
    lines = [message.splitlines()[0][:300]]
    for marker, advice in _API_ADVICE_BY_BODY.items():
        if marker in message:
            lines.append(advice)
            return lines
    for code, advice in _API_ADVICE_BY_STATUS.items():
        if code in message:
            lines.append(advice)
            break
    else:
        lines.append("経路とモデル名を確認すること。鍵を入れ替えた直後なら .env の内容も見ること")
    return lines


def _load(path: Path) -> ExperimentConfig:
    """実験ファイルを読む。利用者の誤りは読める形にして終わる。"""
    try:
        return load_experiment(path)
    except FileNotFoundError:
        _fail(f"実験ファイルが見つからない: {path}")
    except ValidationError as exc:
        _fail(
            f"実験ファイルの条件が正しくない: {path}",
            *(_describe(error) for error in exc.errors()),
        )
    except ValueError as exc:
        _fail(f"実験ファイルを読めない: {path}", str(exc))


@app.command()
def run(
    experiment: Annotated[Path, typer.Argument(help="実験条件の YAML")],
    out: Annotated[Path, typer.Option(help="結果の出力先")] = Path("runs"),
    assert_recall_at_5: Annotated[
        float | None,
        typer.Option(help="回帰検証の閾値。下回ると失敗"),
    ] = None,
    force: Annotated[
        bool,
        typer.Option(help="条件の違う結果を上書きする"),
    ] = False,
) -> None:
    """実験を1本回す。"""
    config = _load(experiment)
    typer.echo(f"{config.run_id}: {config.description}")
    if config.axes():
        typer.echo(f"動かす軸: {', '.join(config.axes())}")
    if config.multi_axis_reason:
        typer.echo(f"複数軸の理由: {config.multi_axis_reason}")

    try:
        result = run_experiment(config, out_root=out, force=force)
    except FileExistsError as exc:
        _fail(str(exc))
    except ValueError as exc:
        # データセットが無い、評価セットとコーパスがずれている、など。
        # これも利用者が直せる誤りなので、トレースバックにしない。
        _fail("実験を回せなかった", str(exc))
    except ExternalCallError as exc:
        # 鍵が違う・残高が無い・レート制限に当たった、など。
        # 鍵を入れて最初に踏むのはここなので、トレースバックにしない。
        _fail("外部APIの呼び出しに失敗した", *_advise(exc))
    typer.echo("")
    typer.echo(_format_variant_table(result))
    typer.echo("")
    typer.echo(f"結果を {out / config.run_id} に書き出した。")

    if assert_recall_at_5 is None:
        return
    worst = min(variant.scores.recall_at_k for variant in result.variants)
    if worst < assert_recall_at_5:
        typer.echo(
            f"\n回帰検証に失敗: Recall@k の最低値 {worst:.3f} が "
            f"閾値 {assert_recall_at_5:.3f} を下回った。",
            err=True,
        )
        raise typer.Exit(code=1)
    typer.echo(f"回帰検証を通過: Recall@k の最低値 {worst:.3f} >= {assert_recall_at_5:.3f}")


#: 生成物であることの目印。`report` はこの行を持つファイルしか上書きしない。
GENERATED_MARKER = "<!-- rageval report が生成する。手で書いた考察は消えるので注意。 -->"


@app.command()
def report(
    runs: Annotated[Path, typer.Option(help="結果が入っているディレクトリ")] = Path("runs"),
    out: Annotated[Path | None, typer.Option(help="書き出し先。省略すると標準出力に出す")] = None,
    force: Annotated[bool, typer.Option(help="手で書いたファイルでも上書きする")] = False,
) -> None:
    """条件 × スコアの表を Markdown で出す。"""
    results = _load_results(runs)
    if not results:
        typer.echo(f"{runs} に result.json が無い。先に run を実行すること。", err=True)
        raise typer.Exit(code=1)

    try:
        markdown = _format_report(results)
    except (KeyError, TypeError) as exc:
        # 古い版の result.json が混じっていると、期待した項目が無いことがある。
        _fail("結果の形が想定と違う", str(exc), "古い実験の結果が混ざっていないか確認すること")
    if out is None:
        typer.echo(markdown)
        return
    _refuse_to_clobber(out, force=force)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(markdown, encoding="utf-8", newline="\n")
    typer.echo(f"{out} に書き出した。")


def _refuse_to_clobber(out: Path, *, force: bool) -> None:
    """手で書いた文書を黙って消さない。

    `docs/04_results.md` のように、考察を書き溜めるファイルを書き出し先に
    指定してしまうことがある。生成物には目印の行を入れてあるので、
    それが無いファイルは「人が書いたもの」とみなして止まる。
    """
    if force or not out.exists():
        return
    try:
        existing = out.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # 読めないものを消さないほうが安全。
        _fail(f"書き出し先を読めない: {out}", "中身を確かめること。上書きするなら --force")
    if GENERATED_MARKER in existing:
        return
    _fail(
        f"書き出し先が生成物ではない: {out}",
        "手で書いた内容が消える。別の名前にするか、消えてよければ --force",
    )


def _load_results(runs: Path) -> list[dict[str, Any]]:
    """`runs/` から結果を集める。壊れているものは、どれが壊れているかを言って止まる。

    途中で書き込みが止まった result.json が1つ混じると、`run` と同じく
    トレースバックが出て、どのファイルが原因か分からなくなる。
    """
    results = []
    for path in sorted(runs.glob("*/result.json")):
        try:
            results.append(json.loads(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            _fail(
                f"結果が壊れている: {path}",
                f"{exc}",
                "実験を回し直すか、このディレクトリを消すこと",
            )
        except OSError as exc:
            _fail(f"結果を読めない: {path}", str(exc))
    return results


def _format_variant_table(result: RunResult) -> str:
    columns = _columns(result.metrics)
    header = ["条件", *[label for _, label in columns], "チャンク数"]
    rows = [
        [
            variant.label,
            *[f"{variant.scores.as_dict()[key]:.3f}" for key, _ in columns],
            str(variant.chunks),
        ]
        for variant in result.variants
    ]
    return _render_table(header, rows)


def _format_report(results: list[dict[str, Any]]) -> str:
    lines = [
        "# 実験結果",
        "",
        GENERATED_MARKER,
        "",
    ]
    for result in results:
        lines.append(f"## {result['run_id']}")
        lines.append("")
        if result.get("description"):
            lines.append(result["description"])
            lines.append("")
        axes = result.get("axes") or []
        lines.append(f"- 動かした軸: {', '.join(axes) if axes else '（なし）'}")
        if result.get("multi_axis_reason"):
            lines.append(f"- 複数軸の理由: {result['multi_axis_reason']}")
        dataset = result.get("dataset", {})
        lines.append(
            f"- 評価セット: {dataset.get('questions')} 件"
            f"（答えられない質問 {dataset.get('unanswerable')} 件）"
            f" / v{dataset.get('qa_set_version', '—')}"
        )
        lines.append(f"- 実行: {result.get('generated_at')}")
        # どのプロバイダで測ったかを必ず出す。ダミー実装での結果と実APIでの結果が
        # 並んだとき、表だけ見て区別できないと「測っていないものを測ったように」読まれる。
        providers = sorted(
            {
                f"埋め込み {v['config']['embedder']['provider']}"
                f" / 生成 {v['config']['generate']['provider']}"
                for v in result["variants"]
            }
        )
        lines.append(f"- プロバイダ: {', '.join(providers)}")
        lines.append("")

        columns = _columns(result.get("metrics", []))
        header = ["条件", *[label for _, label in columns], "チャンク数", "索引(秒)"]
        rows = []
        for variant in result["variants"]:
            scores = variant["scores"]
            rows.append(
                [
                    variant["label"],
                    *[f"{scores[key]:.3f}" for key, _ in columns],
                    str(variant["index"]["chunks"]),
                    f"{variant['index']['seconds']:.1f}",
                ]
            )
        lines.append(_render_markdown_table(header, rows))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _display_width(text: str) -> int:
    """端末上の表示幅。日本語は1文字で2桁ぶん取るので、len() では桁が揃わない。"""
    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in text)


def _pad(text: str, width: int) -> str:
    return text + " " * max(0, width - _display_width(text))


def _render_table(header: list[str], rows: list[list[str]]) -> str:
    cells = [header, *rows]
    widths = [max(_display_width(row[i]) for row in cells) for i in range(len(header))]
    out = ["  ".join(_pad(cell, widths[i]) for i, cell in enumerate(header))]
    out.append("  ".join("-" * width for width in widths))
    out.extend("  ".join(_pad(cell, widths[i]) for i, cell in enumerate(row)) for row in rows)
    return "\n".join(out)


def _escape_cell(text: str) -> str:
    """Markdown の表のセルに入れられる形にする。

    条件のラベルは実験ファイル由来で、`|` や改行が入りうる。そのまま書くと
    列がずれて表が壊れ、読む人には別の条件の値に見える。
    制御文字も落とす（端末やビューアの表示を乗っ取られないため）。
    """
    flattened = " ".join(text.split())
    printable = "".join(c for c in flattened if c.isprintable())
    return printable.replace("\\", "\\\\").replace("|", "\\|")


def _render_markdown_table(header: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(_escape_cell(c) for c in header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    out.extend("| " + " | ".join(_escape_cell(c) for c in row) + " |" for row in rows)
    return "\n".join(out)
