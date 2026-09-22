"""実験の結果を S3 に置く。

コンテナの中でだけ使う。ライブラリ本体（`src/rageval`）からは参照しない。
boto3 もこのスクリプトのためだけに Docker イメージへ入れており、
pyproject.toml の依存には含めていない。

使い方:
    python deploy/upload_results.py <runs ディレクトリ> <バケット名>
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path

CONTENT_TYPES = {
    ".json": "application/json",
    ".jsonl": "application/x-ndjson",
    ".yaml": "application/yaml",
}


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__)
        return 2

    source = Path(argv[1])
    bucket = argv[2]
    if not source.is_dir():
        print(f"結果のディレクトリがない: {source}", file=sys.stderr)
        return 1

    import boto3

    client = boto3.client("s3")
    # 同じ run_id を何度回しても上書きせずに残るよう、実行時刻で仕切る。
    # 過去の結果が消えると条件間の比較ができなくなる。
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    prefix = os.environ.get("RESULTS_PREFIX", "runs").strip("/")

    uploaded = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        key = f"{prefix}/{stamp}/{path.relative_to(source).as_posix()}"
        extra = {"ContentType": CONTENT_TYPES.get(path.suffix, "text/plain; charset=utf-8")}
        client.upload_file(str(path), bucket, key, ExtraArgs=extra)
        print(f"s3://{bucket}/{key}")
        uploaded += 1

    print(f"{uploaded} ファイルを s3://{bucket}/{prefix}/{stamp}/ に置いた。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
