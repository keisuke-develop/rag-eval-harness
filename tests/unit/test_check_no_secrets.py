"""見本ファイルに本物の鍵が混ざるのを止める検査。

`detect-secrets` は `.env.example` のような見本ファイルを検査対象から外す。
利用者には「`.env.example` を写して鍵を入れる」と案内しているので、
**間違えて見本のほうを書き換えるのがいちばん起こりやすい事故**にあたる。
実際に一度起きている（deploy/VERIFICATION.md ではなく docs/07_review_log.md に記録）。

ここで確かめるのは2つ。本物らしい鍵を落とすことと、
差し込み用の値を落とさないこと。**後者を外すと、検査は無視されるようになる。**
"""

from __future__ import annotations

import secrets
import string
from pathlib import Path

import pytest

from tools.check_no_secrets import entropy, find_in, looks_real, main

ALPHABET = string.ascii_letters + string.digits


def fake_key(prefix: str, length: int = 156) -> str:
    """本物と同じ形の、実在しない鍵。毎回変わるので値を書き写せない。"""
    return prefix + "".join(secrets.choice(ALPHABET) for _ in range(length))


@pytest.mark.parametrize(
    "prefix",
    ["sk-proj-", "sk-ant-", "sk-", "ghp_", "github_pat_", "xoxb-", "AKIA", "ASIA"],
)
def test_a_real_looking_key_is_found(prefix: str) -> None:
    found = find_in(f"OPENAI_API_KEY={fake_key(prefix)}")
    assert found, f"{prefix} を見逃した"
    assert found[0][0] == 1


def test_the_value_itself_is_never_reported() -> None:
    """検査の出力はログに残る。そこに鍵を書かない。"""
    key = fake_key("sk-proj-")
    found = find_in(f"OPENAI_API_KEY={key}")
    masked = found[0][2]
    assert key not in masked
    assert key[8:20] not in masked, "一部でも出さない"
    assert "156文字" in masked, "長さは出す。見分けがつくように"


@pytest.mark.parametrize(
    "placeholder",
    [
        "sk-your-key-here",
        "sk-...",
        "sk-proj-XXXXXXXXXXXXXXXXXXXX",
        "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
        "AKIA" + "IOSFODNN7EXAMPLE",  # AWS が文書で使う例示値。組み立てて literal を作らない
    ],
)
def test_a_placeholder_is_left_alone(placeholder: str) -> None:
    """差し込み用の値で落ちると、検査そのものが無視されるようになる。"""
    assert not find_in(f"OPENAI_API_KEY={placeholder}")


def test_the_shipped_example_file_passes() -> None:
    """リポジトリに入っている見本が、そのまま通ること。"""
    assert not find_in(Path(".env.example").read_text(encoding="utf-8"))


def test_entropy_separates_words_from_keys() -> None:
    assert entropy("your-key-here") < 3.0
    assert entropy(fake_key("", 100)) > 3.0


def test_a_short_tail_is_not_a_key() -> None:
    assert not looks_real("abc")
    assert not looks_real("")


def test_it_fails_on_a_file_with_a_key(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "config.env.example"
    target.write_text(f"OPENAI_API_KEY={fake_key('sk-proj-')}\n", encoding="utf-8")
    assert main(["prog", str(target)]) == 1
    captured = capsys.readouterr()
    assert "本物らしい鍵がある" in captured.err
    assert ".env に置く" in captured.err, "どうすればよいかを示す"


def test_it_passes_on_a_clean_file(tmp_path: Path) -> None:
    target = tmp_path / "clean.env.example"
    target.write_text("OPENAI_API_KEY=sk-your-key-here\n", encoding="utf-8")
    assert main(["prog", str(target)]) == 0


def test_binary_and_unreadable_files_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n" + fake_key("sk-proj-").encode())
    (tmp_path / "broken.txt").write_bytes(b"\xff\xfe not utf-8")
    assert main(["prog", str(tmp_path / "image.png"), str(tmp_path / "broken.txt")]) == 0


def test_a_missing_path_is_skipped(tmp_path: Path) -> None:
    assert main(["prog", str(tmp_path / "nope.txt")]) == 0


def test_several_keys_on_one_line_report_once_per_prefix() -> None:
    line = f"A={fake_key('sk-proj-')} B={fake_key('ghp_')}"
    assert len(find_in(line)) >= 2
