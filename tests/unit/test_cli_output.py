"""表の出力が壊れないこと。

条件のラベルは実験ファイル由来で、`|` や改行、制御文字が入りうる。
そのまま Markdown の表に書くと列がずれ、読む人には別の条件の値に見える。

エスケープの期待値を読み違えないよう、この中では `chr()` と連結で
文字を組み立てている。`"\\|"` のような書き方は、テスト側の解釈を挟むぶん
何を確かめているのかが分かりにくくなる。
"""

from __future__ import annotations

from rageval.cli import _columns, _escape_cell, _render_markdown_table

BACKSLASH = chr(92)
PIPE = "|"


def test_a_pipe_in_a_cell_does_not_add_a_column() -> None:
    table = _render_markdown_table(["条件", "値"], [["a" + PIPE + "b", "1"]])
    row = table.splitlines()[2]
    # 区切りとして効く `|` は、直前が backslash でないものだけ。
    delimiters = sum(
        1 for i, c in enumerate(row) if c == PIPE and (i == 0 or row[i - 1] != BACKSLASH)
    )
    assert delimiters == 3, f"列が増えている: {row}"


def test_a_pipe_is_escaped() -> None:
    assert _escape_cell("a" + PIPE + "b") == "a" + BACKSLASH + PIPE + "b"


def test_a_backslash_is_escaped_first_so_the_pipe_stays_escaped() -> None:
    # 入力 a\|b は「backslash + pipe」。先に backslash を二重化しないと、
    # 出力の a\|b が「エスケープ済みの pipe」と読まれて列が消える。
    assert _escape_cell("a" + BACKSLASH + PIPE + "b") == (
        "a" + BACKSLASH * 2 + BACKSLASH + PIPE + "b"
    )


def test_newlines_are_flattened() -> None:
    assert _escape_cell("一行目" + chr(10) + "二行目") == "一行目 二行目"


def test_control_characters_are_dropped() -> None:
    """端末やビューアの表示を乗っ取られないように落とす。"""
    assert _escape_cell("正常" + chr(27) + "[31m赤" + chr(0)) == "正常[31m赤"


def test_a_plain_cell_is_untouched() -> None:
    assert _escape_cell("chunk.size=512") == "chunk.size=512"


def test_no_declared_metrics_means_show_everything() -> None:
    """metrics を書いていない古い実験ファイルでも、表が空にならないこと。"""
    keys = [key for key, _ in _columns([])]
    assert keys == ["recall_at_k", "mrr", "citation_match", "abstention"]


def test_declared_metrics_are_filtered_and_ordered() -> None:
    assert [key for key, _ in _columns(["mrr", "recall_at_k"])] == ["recall_at_k", "mrr"]
