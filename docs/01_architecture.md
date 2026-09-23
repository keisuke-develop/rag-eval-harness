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
| `evaluate.py` | 評価セットの読み込みと、検索側・生成側の指標の算出 | — |
| `external.py` | 外部API呼び出しの共通処理（タイムアウト・リトライ・レート制限・使用量記録） | — |
| `runner.py` | 実験1回ぶんの流れの組み立てと、`runs/` への書き出し | — |
| `cli.py` | `run` と `report` の2コマンド | — |

**差し替え可能にするのは4点だけ**にした。比較したい軸以外を可変にすると、結果が動いたときに原因を特定できなくなる。詳細は [adr/0002-limit-pluggable-points.md](adr/0002-limit-pluggable-points.md)。

`external.py` と `runner.py` は当初の8モジュールに無かったが、実装して必要になったので足した。どちらも差し替え点ではない。

- `external.py`：タイムアウト・リトライ・レート制限・トークン数記録は `embed.py` と `generate.py` の両方で要る。片方に置いてもう片方から呼ぶと依存が捻れるため、独立させた
- `runner.py`：索引フェーズと評価フェーズをつなぐ処理をどこかが持つ必要がある。`cli.py` に置くと、実験を回す処理を試すのに CLI 越しでしか触れなくなる

## 3. 抽象のかたち

差し替え点は Protocol（構造的部分型）で定義する。継承関係を強制せず、テスト時にダミー実装を差し込みやすくするため。Protocol は、それを使う側のモジュールに置いてある（`Chunker` は `ingest.py`、`Embedder` は `embed.py`、`VectorStore` は `store.py`、`Reranker` は `retrieve.py`）。

生成プロバイダには Protocol を切らない。ADR 0002 で「あえて固定する」と決めた箇所なので、`generate.py` の中で `provider` の文字列による明示的な分岐にしてある。差し替えたくなったら分岐を1つ足すという判断が要る。

**実際に1つ足した**（2026-09-23、`provider: bedrock`）。鍵を持たずに呼べること・前払いの残高に縛られないこと・比較の次元数を揃えられることが理由で、経緯は [ADR 0002 の追記](adr/0002-limit-pluggable-points.md)に残してある。分岐は3つになったが、**差し替え点の見た目は与えていない。**

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

実装にあたって、以下を足した。

| 項目 | 内容 |
|---|---|
| 深いマージ | `variants` はセクションごと置き換えるのではなく、`fixed` に深く重ねる。`retrieve: { rerank: true }` だけ書けば `top_k` は `fixed` のまま残る |
| 軸の検証 | `fixed` に重ねた結果を比べ、**実際に値が動いている葉のフィールド**を軸と数える。2つ以上動いていて `multi_axis_reason` が書かれていなければ、実行前に落とす |
| `multi_axis_reason` | 実験006のように意図して複数軸を動かす場合だけ書く。`result.json` にも残る |
| `label` | 条件の名前。省略すると軸から自動で付く（例 `chunk.size=256`） |
| `retrieve.candidates` | 再ランクにかける候補数。省略すると `top_k` と同じで、並べ替えるだけで集合は変わらない |

プロバイダは、外部APIに出ないものと出るものの2つずつを持つ。

| 種類 | 外部APIに出ない | 出る |
|---|---|---|
| 埋め込み | `hashing`（文字 n-gram のハッシュ化ベクトル） | `openai` |
| 生成 | `quote`（最上位ヒットをそのまま引用） | `openai` |

外部APIに出ない側は乱数ではなく、実際に文字の重なりを拾う。CI の回帰検証と、鍵が無い環境でのパイプライン全体の確認に使う。

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

**同じ `run_id` で回し直すと、出力先が同じになる。** この道具は決定的なので、条件が同じなら結果も同じで、上書きして構わない。しかし**条件が違えばそれは別の測定**であり、消すと過去と比べられなくなる。そのため、スナップショットを突き合わせて条件が違う場合は**実験を回す前に断る**。上書きしてよいときだけ `--force` を付ける。

`report --out` も同じ考えで守る。生成した表には目印の行を入れてあり、**その行を持たないファイルには書き出さない。**
`docs/04_results.md` のような、考察を書き溜めたファイルを書き出し先に指定してしまうことがあるため。
こちらも `--force` で上書きできる。

比較できるようにするための道具が、比較の材料も、それについて書いたことも、黙って消してはいけない。

### APIキーの受け渡し

要件（[00_requirements.md](00_requirements.md)）で「API キーは `.env` で与える」と決めている。
読むのは **CLI の入口で1回だけ**で、**カレントディレクトリの `.env`** に限る。

- ライブラリ側（`external.py` など）は `os.environ` しか見ない。副作用を持つのは入口だけにする
- すでに環境変数があればそちらを優先する。CI と AWS は環境変数で渡すので、ファイルが後から上書きしてはいけない
- 探索範囲をカレントディレクトリに限るのは、どの `.env` が効いたのか分からなくなるのを避けるため

`.env.example` を置いてありながら**読む処理が無かった**ことがある。
案内どおりに置いても効かない状態だったので、効くことをテストで固定した。

## 7. テスト方針

| 層 | 対象 | 方針 |
|---|---|---|
| unit | 各モジュール単体 | 外部 API はダミー実装で差し替える。チャンク分割の境界、リトライの発火条件、指標の計算を中心に |
| integration | パイプライン通し | 小さな固定コーパスと3件の評価セットで、end-to-end が壊れていないことを確認する |
| CI | 全体 | pytest / ruff / mypy に加え、評価スコアの回帰検証を走らせる |

## 8. CI での回帰検証

固定の評価セットで検索スコアを算出し、ベースラインを下回ったら落とす。

```
pytest → ruff → mypy
build_qa_set.py --check → rageval run experiments/ci_baseline.yaml
                            → Recall@5 が 0.50 を下回ったら exit 1
```

これを入れるのは、**評価を手作業に残すと必ず止まる**ため。自動で回るところに置いて初めて、精度の劣化に気づける状態になる。

閾値について。設計時は 0.70 と書いていたが、実装して測った値は **0.562** だった（`hashing` 埋め込み / chunk 512 / top-k 5）。**外れた見込みなので、閾値のほうを実測に合わせて 0.50 に下げた。** スコアを取り繕うために評価セットや条件をいじることはしない。

この 0.50 は到達目標ではなく「壊れていないこと」の下限。正解根拠のオフセットがずれる、チャンク分割が崩れる、といった事故は Recall をほぼ0まで落とすので、この水準でも検知できる。

`build_qa_set.py --check` を前に置いているのは、コーパスを1文字でも動かすと評価セットのオフセットがずれるため。実験を回す前に、生成物と原稿が食い違っていないことを確かめる。

## 9. 想定する拡張

現時点では作らないが、抽象の切り方はこれらを見据えている。

- Vector DB を Chroma から Qdrant へ差し替える（`store.py` の実装追加のみで済む）
- 再ランカーの有無を実験軸に加える（`retrieve.py` は既に受け口を持つ）
- AWS 上での実行 → **一式を用意した**（[../deploy/README.md](../deploy/README.md)）。ECS Fargate のタスクとして都度起動し、終われば止まる。常駐するものは作らない。テンプレートは `cfn-lint` を通しただけで、実際に適用してはいない
