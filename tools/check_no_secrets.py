"""コミット対象のファイルに、本物らしい鍵が混ざっていないかを見る。

`detect-secrets` を CI で通しているが、**見本ファイルを検査対象から外す**。
`.env.example` に本物の鍵を書いても検出しない（同じ内容でも `.txt` / `.py` /
`.yaml` なら検出する）。

利用者には「`.env.example` を写して鍵を入れる」と案内しているので、
**間違えて見本のほうを書き換えるのは、いちばん起こりやすい事故**にあたる。
実際に一度起きた（2026-09-23）。そのときリポジトリを守ったものは何も無く、
コミット前にたまたま差分を見て気づいただけだった。

ここでは「本物らしさ」だけを見る。提供元ごとの接頭辞のあとに、
十分な長さと散らばりを持つ文字列が続いていたら、本物とみなして落とす。
`sk-your-key-here` のような差し込み用の値は、散らばりが足りないので通る。

    uv run python tools/check_no_secrets.py
"""

from __future__ import annotations

import math
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

#: 提供元ごとの接頭辞。ここに挙げたものだけを見る。
#: 何でも拾おうとすると誤検出が増えて、結局は無視されるようになる。
PREFIXES = (
    "sk-proj-",  # OpenAI（プロジェクト鍵）
    "sk-ant-",  # Anthropic
    "sk-",  # OpenAI（旧形式）。上の2つに当たらなかったものだけが来る
    "ghp_",  # GitHub（個人アクセストークン）
    "github_pat_",
    "gho_",  # GitHub（OAuth）
    "xoxb-",  # Slack（bot）
    "xoxp-",  # Slack（user）
    "AKIA",  # AWS（アクセスキーID）
    "ASIA",  # AWS（一時的な認証情報）
)

#: 接頭辞のあとに続く文字。鍵に使われる文字だけ。
TAIL = re.compile(r"[A-Za-z0-9_\-]+")

#: 本物とみなす下限。差し込み用の値（sk-your-key-here など）と分ける。
MIN_TAIL_LENGTH = 20
MIN_ENTROPY = 3.0

#: 中身が文字列でないものは見ない。
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf", ".zip", ".whl", ".lock"}


def entropy(text: str) -> float:
    """1文字あたりの情報量。低いものは人が書いた言葉で、鍵ではない。"""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    return -sum((n / total) * math.log2(n / total) for n in counts.values())


def looks_real(tail: str) -> bool:
    return len(tail) >= MIN_TAIL_LENGTH and entropy(tail) >= MIN_ENTROPY


def find_in(text: str) -> list[tuple[int, str, str]]:
    """(行番号, 接頭辞, 伏せた値) を返す。**値そのものは返さない。**"""
    found = []
    for number, line in enumerate(text.splitlines(), start=1):
        for prefix in PREFIXES:
            start = 0
            while (index := line.find(prefix, start)) != -1:
                start = index + len(prefix)
                match = TAIL.match(line, start)
                if match and looks_real(match.group()):
                    found.append((number, prefix, f"{prefix}…（{len(match.group())}文字）"))
                    break
    return found


def tracked_files() -> list[Path]:
    # 引数は固定で、外から来る値が混ざる余地が無い。`git` を絶対パスで持たないのは、
    # 手元・CI・コンテナで置き場所が違うため。シェルは通していない。
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [Path(name) for name in out.split()]


def main(argv: list[str]) -> int:
    paths = [Path(a) for a in argv[1:]] or tracked_files()
    problems = []
    for path in paths:
        if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for number, _prefix, masked in find_in(text):
            problems.append(f"{path.as_posix()}:{number} 本物らしい鍵がある: {masked}")

    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(
            "\nコミット対象のファイルに鍵を置かないこと。"
            "\n鍵は .env に置く（.gitignore に入っている）。"
            "\n.env.example には差し込み用の値だけを書くこと。",
            file=sys.stderr,
        )
        return 1
    print(f"本物らしい鍵は見つからなかった（{len(paths)} ファイル）")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
