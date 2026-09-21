# アーキテクチャ設計

作成日: 2026-09-21

---

## 1. 全体像

実験1回の流れ。左から右へ一方向に流れ、各段の出力は次の段の入力にしかならない。

```
experiments/001_chunk_size.yaml
        │
        ▼
   [ config ]  実験条件を Pydantic で読み込み、実行前に検証する
        │
        ├──────────── 索引フェーズ（条件が変われば作り直す） ────────────┐
        │                                                                  │
        ▼                    ▼                      ▼                      │
  [ ingest ]  ──chunks──▶ [ embed ]  ──vectors──▶ [ store ]               │
   分割戦略               プロバイダ抽象           Vector DB 抽象          │
        │                                                                  │
        └──────────────────────────────────────────────────────────────────┘
                                   │
        ┌──────────────── 評価フェーズ（評価セットを順に流す）─────────────┐
        │                                                                  │
        ▼                    ▼                      ▼                      │
  [ retrieve ] ──contexts─▶ [ generate ] ──answer─▶ [ evaluate ]          │
   top-k / 再ランク        構造化出力              指標の算出              │
        │                                                                  │
        └──────────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                         runs/<run_id>/result.json
                                   │
                                   ▼
                            docs/04_results.md
```

## 2. モジュールの責務

| モジュール | 責務 | 差し替え |
|---|---|---|
| `config.py` | 実験条件の型定義と検証。YAML を読んで `ExperimentConfig` にする | — |
| `ingest.py` | コーパスの読み込みとチャンク分割 | **チャンク戦略** |
| `embed.py` | チャンクとクエリのベクトル化。リトライ・レート制限・コスト記録 | **埋め込みプロバイダ** |
| `store.py` | ベクトルの格納と近傍検索 | **Vector DB** |
| `retrieve.py` | クエリに対する文脈の取得。top-k、再ランクの適用 | **再ランカー** |
| `generate.py` | 文脈つきプロンプトの組み立てと生成。出力の構造化と検証 | 生成プロバイダ |
| `evaluate.py` | 検索側・生成側の指標の算出 | — |
| `cli.py` | `run` と `report` の2コマンド | — |

**差し替え可能にするのは4点だけ**にした。比較したい軸以外を可変にすると、結果が動いたときに原因を特定できなくなる。詳細は [adr/0002-limit-pluggable-points.md](adr/0002-limit-pluggable-points.md)。

## 3. 抽象のかたち

差し替え点は Protocol（構造的部分型）で定義する。継承関係を強制せず、テスト時にダミー実装を差し込みやすくするため。

```python
class Chunker(Protocol):
    def split(self, doc: Document) -> list[Chunk]: ...

class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[Vector]: ...

class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[Vector]) -> None: ...
    def search(self, query: Vector, k: int) -> list[SearchHit]: ...

class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit]) -> list[SearchHit]: ...
```

## 4. 実験条件の型

```yaml
# experiments/001_chunk_size.yaml
run_id: "001_chunk_size"
description: "チャンクサイズが検索精度に与える影響を見る"

dataset:
  corpus: "datasets/corpus"
  qa_set: "datasets/qa_set.jsonl"

variants:                 # この軸だけを動かす
  - chunk: { size: 256, overlap: 32 }
  - chunk: { size: 512, overlap: 64 }
  - chunk: { size: 1024, overlap: 128 }

fixed:                    # 比較の邪魔をしないよう固定する
  embedder: { provider: "openai", model: "text-embedding-3-small" }
  store:    { kind: "chroma" }
  retrieve: { top_k: 5, rerank: false }
  generate: { provider: "openai", model: "gpt-4o-mini", temperature: 0.0, seed: 42 }
```

`variants` と `fixed` を分けたのは、**「何を動かして何を固定したか」が実験ファイルを見るだけで分かる**ようにするため。結果の表を読む人が、条件の差分を推測しなくて済む。

## 5. 外部呼び出しの扱い

埋め込みと生成は外部 API に出るため、以下を共通のデコレータで包む。

| 対策 | 内容 |
|---|---|
| タイムアウト | 接続・読み取りともに明示的に設定する |
| リトライ | 指数バックオフ。429 と 5xx のみ対象、4xx は即座に失敗させる |
| レート制限 | トークンバケットで秒あたりのリクエスト数を抑える |
| コスト記録 | 入出力トークン数を実行ごとに記録し、result.json に含める |
| ログ | リクエストIDと所要時間を残す。プロンプト本文は評価セット由来なので記録してよい |

## 6. 出力

```
runs/<run_id>/
├── result.json     # 条件・スコア・トークン数・所要時間
├── predictions.jsonl  # 質問ごとの検索結果と生成回答（後から見返せるように）
└── config.snapshot.yaml  # 実行時の条件（YAML の変更に影響されないよう固定）
```

`config.snapshot.yaml` を残すのは、実験ファイルを後で編集しても過去の結果の条件が変わらないようにするため。再現性の担保に必要。

## 7. テスト方針

| 層 | 対象 | 方針 |
|---|---|---|
| unit | 各モジュール単体 | 外部 API はダミー実装で差し替える。チャンク分割の境界、リトライの発火条件、指標の計算を中心に |
| integration | パイプライン通し | 小さな固定コーパスと3件の評価セットで、end-to-end が壊れていないことを確認する |
| CI | 全体 | pytest / ruff / mypy に加え、評価スコアの回帰検証を走らせる |

## 8. CI での回帰検証

固定の小さな評価セットで検索スコアを算出し、ベースラインを下回ったら落とす。

```
pytest → ruff → mypy → rageval run experiments/ci_baseline.yaml
                          → Recall@5 が 0.70 を下回ったら exit 1
```

これを入れるのは、**評価を手作業に残すと必ず止まる**ため。自動で回るところに置いて初めて、精度の劣化に気づける状態になる。

## 9. 想定する拡張

現時点では作らないが、抽象の切り方はこれらを見据えている。

- Vector DB を Chroma から Qdrant へ差し替える（`store.py` の実装追加のみで済む）
- 再ランカーの有無を実験軸に加える（`retrieve.py` は既に受け口を持つ）
- AWS 上での実行（Lambda もしくは ECS。`cli.py` を叩くだけなので構成は薄くて済む）
