"""CodeBuild に渡すソースの zip を作る。

手元に Docker が無いので、イメージのビルドは CodeBuild に任せている。
その入力になる zip をここで作る。`zip` コマンドは Windows に無いことが多いので
Python で組む。

**秘密情報を混ぜないことがこのスクリプトの主な仕事。** 除外リストに漏れがあると、
`.env` や認証情報がビルド環境と、場合によってはイメージの中まで運ばれる。
除外は「拡張子で弾く」ではなく「入れるものを選ぶ」方針にしてある。

    python deploy/package_source.py /tmp/src.zip
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

#: zip に入れるもの。ここに書いていないものは入らない。
INCLUDE = (
    "pyproject.toml",
    "uv.lock",
    "README.md",
    "LICENSE",
    "Dockerfile",
    ".dockerignore",
)
INCLUDE_DIRS = ("src", "datasets", "experiments", "deploy")

#: 上のディレクトリのなかでも入れないもの。
EXCLUDE_NAMES = frozenset({"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"})
EXCLUDE_SUFFIXES = frozenset({".pyc", ".pyo", ".pem", ".key", ".crt", ".p12", ".pfx"})

#: 秘密が入りうる名前。見つけたら黙って飛ばさず、失敗させる。
SECRET_HINTS = (".env", "credentials", "secret", "id_rsa", ".aws")


def is_secret(path: Path) -> bool:
    name = path.name.lower()
    return any(hint in name for hint in SECRET_HINTS) and name != ".env.example"


def collect(root: Path) -> list[Path]:
    files: list[Path] = []
    for name in INCLUDE:
        candidate = root / name
        if candidate.is_file():
            files.append(candidate)

    for directory in INCLUDE_DIRS:
        for path in sorted((root / directory).rglob("*")):
            if not path.is_file():
                continue
            if any(part in EXCLUDE_NAMES for part in path.parts):
                continue
            if path.suffix.lower() in EXCLUDE_SUFFIXES:
                continue
            files.append(path)
    return files


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    root = Path(__file__).resolve().parents[1]
    out = Path(argv[1])
    files = collect(root)

    leaked = [f for f in files if is_secret(f)]
    if leaked:
        print("秘密情報が混ざりかけた。除外リストを直すこと:", file=sys.stderr)
        for path in leaked:
            print(f"  {path.relative_to(root)}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root).as_posix())

    size = out.stat().st_size
    print(f"{len(files)} ファイル / {size / 1024:.0f} KiB を {out} に固めた。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
