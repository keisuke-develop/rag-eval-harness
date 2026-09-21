# Architecture Decision Record

設計上の判断と、その理由を残す場所。
「なぜこうしたか」はコードからは読み取れないため、判断した時点で書く。

| # | 決定 | 状態 |
|---|---|---|
| [0001](0001-evaluation-set-first.md) | 評価セットをパイプラインより先に作る | 採用 |
| [0002](0002-limit-pluggable-points.md) | 差し替え可能な箇所を4つに限定する | 採用 |
| [0003](0003-no-llm-judge.md) | 生成の評価に LLM-as-a-judge を使わない | 採用 |
