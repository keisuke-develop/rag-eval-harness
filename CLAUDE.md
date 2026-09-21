# CLAUDE.md

このリポジトリで作業するときの前提。

## このプロジェクトは何か

RAG の設定（チャンクサイズ・重複幅・埋め込みモデル・top-k・再ランク）を変えたときに、
検索と生成の精度がどう動くかを、**同じ評価セットで比較するための道具**。

RAG アプリではなく、RAG を測る装置。この区別を崩さないこと。
機能を足したくなったら、まず「それは比較の役に立つか」を問う。

## 設計判断（変更する前に読む）

- [docs/adr/0001](docs/adr/0001-evaluation-set-first.md) 評価セットをパイプラインより先に作る
- [docs/adr/0002](docs/adr/0002-limit-pluggable-points.md) 差し替え可能な箇所を4つに限定する
- [docs/adr/0003](docs/adr/0003-no-llm-judge.md) 生成の評価に LLM-as-a-judge を使わない

これらに反する変更を提案する場合は、新しい ADR を書いて理由を残すこと。
黙って覆さない。

## 実装の順序

設計は済んでいる。以下の順で進める。理由は ADR 0001。

1. コーパス選定（再配布可能なライセンスの公開文書のみ）
2. 評価セット 40〜60 件の作成（`datasets/qa_set.jsonl`）
3. `config.py` — 実験条件の型定義
4. `ingest.py` → `embed.py` → `store.py` → `retrieve.py` → `generate.py`
5. `evaluate.py` — 指標の算出
6. `cli.py` — `run` / `report`
7. 実験の実施と `docs/04_results.md` への記録
8. CI の回帰検証を有効化

**2 を飛ばして 3 以降に進まないこと。**

## 守ること

| 項目 | 内容 |
|---|---|
| 正解根拠 | チャンクIDではなく**文字オフセット**で持つ。IDで持つとチャンクサイズの比較実験が成立しない |
| 実験の軸 | 1実験につき動かす軸は1つ。`variants` と `fixed` を YAML で明示的に分ける |
| 外部呼び出し | タイムアウト・指数バックオフのリトライ・レート制限・トークン数記録を必ず通す |
| テスト | 外部 API は必ずモックする。CI で実際の API を叩かない |
| 型 | mypy strict を通す。`Any` を使うときは理由をコメントに書く |
| 秘密情報 | `.env` はコミットしない。サンプル値は `.env.example` に置く |
| ライセンス | コーパスに文書を足したら `datasets/corpus/LICENSE-NOTICE.md` に出典とライセンスを追記する |

## 書かないこと

- **他所の業務で書いたコードを持ち込まない。** このリポジトリは public。設計判断を再現した新規実装のみ
- 実験結果を良く見せるための後付けの仮説。外れた仮説は外れたと書く
- 測っていないものを測ったように書く（回答の自然さ・網羅性は評価対象外）

## コマンド

```bash
uv sync                  # 依存解決
uv run pytest            # テスト
uv run ruff check .      # Lint
uv run ruff format .     # フォーマット
uv run mypy              # 型チェック
uv run rageval run experiments/001_chunk_size.yaml   # 実験（未実装）
uv run rageval report --out docs/04_results.md       # 結果出力（未実装）
```

## ドキュメントの位置づけ

| ファイル | 役割 |
|---|---|
| `README.md` | 外部の読み手が最初に見る。結果の表をここに転記する |
| `docs/00_requirements.md` | 何を作り、何を作らないか |
| `docs/01_architecture.md` | モジュール構成とデータフロー |
| `docs/02_evaluation.md` | このリポジトリの中核。指標と評価セットの設計 |
| `docs/03_experiment_plan.md` | 実験6本の計画と仮説 |
| `docs/04_results.md` | 実験結果。実施後に記入 |

実装を変えたら、対応するドキュメントも同じコミットで更新する。
