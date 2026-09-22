#!/bin/sh
# 実験を1本回して、結果を S3 に置く。
#
# 引数に実験ファイルを渡す。省略すると ci_baseline を回す。
# RESULTS_BUCKET が設定されていればアップロードし、なければローカルに残すだけ。
set -eu

EXPERIMENT="${1:-experiments/ci_baseline.yaml}"
OUT="${RAGEVAL_OUT:-/tmp/runs}"

echo "実験: ${EXPERIMENT}"
rageval run "${EXPERIMENT}" --out "${OUT}"

if [ -n "${RESULTS_BUCKET:-}" ]; then
  python deploy/upload_results.py "${OUT}" "${RESULTS_BUCKET}"
else
  echo "RESULTS_BUCKET が未設定。結果は ${OUT} に置いたままにする。"
fi
