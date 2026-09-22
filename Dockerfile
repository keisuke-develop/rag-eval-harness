# 実験を1本回して結果を S3 に置くためのイメージ。
# 常駐しない。ECS の RunTask で起動し、終わったら止まる（deploy/README.md を参照）。
FROM python:3.12-slim-bookworm

# ベースイメージの OS パッケージにセキュリティ更新を当てる。
# ECR の ScanOnPush が拾う指摘はほぼここ（perl / openssl / util-linux / zlib）から出る。
# ビルドのたびに取得する最新版が変わるため、イメージはビット単位では再現しない。
# 「同じ入力から同じイメージ」より「脆弱性を持ち越さない」を優先している。
# ビルドした版は ECR のタグ（不変）とダイジェストで特定できるので、追跡はそちらで担保する。
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# uv はビルド時だけ使う。実行時の依存には入れない。
COPY --from=ghcr.io/astral-sh/uv:0.8.12 /uv /usr/local/bin/uv

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PROJECT_ENVIRONMENT=/usr/local

WORKDIR /app

# 依存だけを先に入れて、コードの変更でこの層が壊れないようにする。
# --locked を付けているので、uv.lock と pyproject.toml がずれていればここで落ちる。
# イメージに入るものを必ずロックで固定するため（再現性と、入った物の追跡のため）。
# LICENSE は pyproject の license-files が指している。
# 無くてもビルドは通ってしまうが、その場合イメージの中の配布物から
# ライセンス本文が静かに落ちる。MIT は本文の同梱を求めるので必ず入れる。
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --locked --no-dev --group deploy --no-install-project

COPY src/ ./src/
COPY datasets/ ./datasets/
COPY experiments/ ./experiments/
COPY deploy/entrypoint.sh deploy/upload_results.py ./deploy/
RUN uv sync --locked --no-dev --group deploy && chmod +x ./deploy/entrypoint.sh

# 書き込むのは /tmp だけ。ルートファイルシステムは読み取り専用で動く。
ENV RAGEVAL_OUT=/tmp/runs

# root で動かす必要がない。
RUN useradd --create-home --uid 10001 rageval
USER rageval

ENTRYPOINT ["./deploy/entrypoint.sh"]
CMD ["experiments/ci_baseline.yaml"]
