# rag-eval-harness

[![CI](https://github.com/keisuke-develop/rag-eval-harness/actions/workflows/ci.yml/badge.svg)](https://github.com/keisuke-develop/rag-eval-harness/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-informational.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12%2B-informational.svg)](pyproject.toml)

**RAG の設定を変えたとき、検索と生成の精度がどう動くかを、同じ評価セットで比較するための道具です。**

RAG は「とりあえず動くもの」を作るのは簡単ですが、チャンクサイズを 512 から 256 に変えたときに精度が上がったのか下がったのかは、測る仕組みがないと分かりません。このリポジトリは、その「測る仕組み」のほうを主役にしています。

```bash
# 実験条件を宣言して回す（APIキー不要で動く条件）
uv run rageval run experiments/ci_baseline.yaml

# 条件 × スコアの表を出す（標準出力。--out で書き出しもできる）
uv run rageval report
```

---

## 何ができるか

| できること | 内容 |
|---|---|
| 条件を宣言して回す | チャンクサイズ・重複幅・埋め込みモデル・top-k・再ランクの有無を YAML で指定し、コマンド一発で実行する |
| 同じ物差しで比較する | 固定の評価セットに対して、検索側は Recall@k / MRR、生成側は根拠一致率を算出する |
| 結果を残す | 条件 × スコアの表を Markdown で出力し、過去の実験と並べて比較できる |
| 壊れたら気づく | 評価スコアが閾値を下回ったら CI が落ちる。プロンプトや設定の変更が精度を劣化させたことを検知する |

## 設計の方針

**評価セットを最初に作る。** パイプラインより先に、質問と正解根拠の対を 53 件用意しました（うち5件は、文脈に答えが無く棄権すべき質問）。何を良しとするかを決めないまま改善に入ると、後から「良くなった気がする」としか言えなくなるためです。正解根拠はチャンクIDではなく**文字オフセット**で持っています。IDで持つと、チャンクサイズを変えた瞬間に評価セットが使えなくなり、比較実験そのものが成立しないためです。評価セットの作成基準は [docs/02_evaluation.md](docs/02_evaluation.md) に書いています。

**差し替えられる場所を限定する。** チャンク戦略・埋め込みプロバイダ・Vector DB・再ランカーの4つだけを差し替え可能にし、それ以外は固定しました。比較したい軸だけを可変にしないと、何が効いたのか分からなくなります。

**実験は宣言で書く。** 条件をコードに埋め込まず YAML に出しました。実験の再現と、条件の差分の追跡ができるようにするためです。

判断の詳細は [docs/adr/](docs/adr/) に Architecture Decision Record として残しています。

## ドキュメント

| ファイル | 内容 |
|---|---|
| [docs/08_poc_report.md](docs/08_poc_report.md) | **PoC 実施レポート。全体を先に把握したい場合はここから** |
| [docs/00_requirements.md](docs/00_requirements.md) | 何を作るか、何を作らないか |
| [docs/01_architecture.md](docs/01_architecture.md) | モジュール構成とデータフロー |
| [docs/02_evaluation.md](docs/02_evaluation.md) | 評価指標の定義と、なぜその指標か |
| [docs/03_experiment_plan.md](docs/03_experiment_plan.md) | 実験条件のマトリクス |
| [docs/04_results.md](docs/04_results.md) | 実APIで回した実験結果（**未実施**） |
| [docs/05_offline_results.md](docs/05_offline_results.md) | ダミー埋め込みで回した検索側だけの測定。04 とは混ぜない |
| [docs/06_security_review.md](docs/06_security_review.md) | セキュリティと品質の評価。受け入れた残リスクも書いてある |
| [docs/07_review_log.md](docs/07_review_log.md) | レビューの記録。見つからなかった回も残している |
| [docs/adr/](docs/adr/) | 設計判断の記録（5件） |
| [deploy/README.md](deploy/README.md) | AWS で回す場合の構成と手順 |

## 実験結果

**計画どおりの実験（実API）は未実施です。** 代わりに、外部APIに出ない
ダミー埋め込みで**検索側だけ**を10本回しました。詳細と考察は
[docs/05_offline_results.md](docs/05_offline_results.md)。

| 実験 | 動かした軸 | 観測した差 | 判定 |
|---|---|---|---|
| 001 / 009 | チャンクサイズ | Recall 0.034 | 検出できない（233問必要） |
| 002 | 重複幅 | Recall 0.083 | 境界上（33問必要） |
| 003 | top-k | Recall 0.146 | **効果あり**（0.458 → 0.604） |
| 004 / 008 | 再ランクの有無 | MRR +0.145 | **効果あり**（4位相すべてで改善） |
| 006 | 組み合わせ | Recall 0.042 / MRR 0.189 | Recall は検出できない、MRR は効果あり |
| 007 | 境界の位相 | — | **ノイズ幅の実測**（0.083＝4問） |
| **010** | **再ランクの候補数** | **Recall 0.250** | **最大の効果**（0.562 → 0.812） |

この表にあるのは検索側の2指標だけです。**根拠一致率と棄権率は載せていません。**
ダミー生成では前者は「検索の1位が当たった割合」でしかなく、後者は常に 0 になるためです。
実験005（埋め込みモデル比較）は、ダミーの埋め込みが1種類しかないため回していません。

いちばん大きい発見は、**埋め込みが足を引っ張っていたこと**でした。
全チャンクを文字 n-gram の一致で並べ替えるだけで Recall が 0.562 → 0.812 になります。
つまり他の実験で測っていたのは**弱い検索器のうえでの効果**で、
これは「埋め込みモデルの選択が一番効く」という実験計画の見立てを裏づけます。

もう1つは**測定系の分解能**です。条件を何も変えずに Recall が 0.083（4問）振れ、
**48問で検出できるのは 0.066 以上の差**でした。これで、どの実験が読めて
どれが読めなかったかがすべて説明できます。当初「チャンクサイズは中間に山がある」と
書いた結論は、後からこれを測って取り下げました。

**装置の目盛りと、何なら検出できるのかを確かめる前に条件を比べ始めたのが、今回いちばんの手順の誤りでした。**

## 構成

```
rag-eval-harness/
├── src/rageval/
│   ├── config.py        # 実験条件の型定義（Pydantic）
│   ├── ingest.py        # 文書投入 → チャンク分割
│   ├── embed.py         # 埋め込み。プロバイダ抽象 + リトライ / レート制限
│   ├── store.py         # Vector DB 抽象
│   ├── retrieve.py      # 検索。top-k・再ランクの切り替え
│   ├── generate.py      # 生成。構造化出力を Pydantic で検証
│   ├── evaluate.py      # 評価セットの読み込みと指標の算出
│   ├── external.py      # 外部呼び出しの共通処理（タイムアウト / リトライ / レート制限 / 使用量記録）
│   ├── runner.py        # 実験1回ぶんの流れの組み立て
│   └── cli.py           # run / report
├── datasets/
│   ├── corpus/          # 評価用コーパス 12文書（出典・ライセンスは corpus/LICENSE-NOTICE.md）
│   ├── fetch_corpus.py  # コーパスの取得と加工
│   ├── qa_draft.yaml    # 評価セットの原稿（人が編集するのはここだけ）
│   ├── build_qa_set.py  # 原稿を検査して qa_set.jsonl を生成
│   └── qa_set.jsonl     # 評価セット 53件（生成物）
├── experiments/         # 実験条件の YAML
├── tests/               # pytest（unit / integration）
├── deploy/              # AWS で回すための CloudFormation と手順
├── Dockerfile           # 実験1本を回して結果を S3 に置くイメージ
└── .github/workflows/   # テスト・Lint・型チェック・評価の回帰検証
```

## セットアップ

```bash
uv sync
uv run pytest
uv run rageval run experiments/ci_baseline.yaml   # APIキー不要で動く
```

CI では上に加えて、静的解析（ruff の flake8-bandit ルール）、依存ライブラリの
脆弱性検査（pip-audit）、CloudFormation の検査（cfn-lint）、評価スコアの回帰検証を回しています。
何を対策し、何を PoC として外したかは [docs/06_security_review.md](docs/06_security_review.md) に
観点ごとに一覧化しています。

### APIキーが無くても動く

埋め込みと生成には、**外部APIに出ない決定的な実装**をそれぞれ用意しています。
鍵が無くてもパイプライン全体を通して回せ、CI の回帰検証もこちらで動かしています。

| 段 | `provider` | 中身 |
|---|---|---|
| 埋め込み | `hashing` | 文字 n-gram をハッシュして符号つきで足し込み、L2 正規化したベクトル |
| 生成 | `quote` | 最上位ヒットのチャンクをそのまま引用し、そのIDを根拠として返す |

乱数ではなく実際に文字の重なりを拾うので、検索として弱いなりに機能します
（評価セット53件で Recall@5 = 0.562）。ただし **`quote` が測っているのは
検索の1位が当たったかどうかで、文章生成の良し悪しではありません。**
棄権率もこの条件では常に 0 になります（理由は [docs/02_evaluation.md](docs/02_evaluation.md)）。

### 実APIで実験する場合

**実装は入っているので、足すのは鍵と YAML の2箇所だけです。**
`.env` に `OPENAI_API_KEY` を置き（`cp .env.example .env`）、実験ファイルの
`fixed.embedder.provider` を `openai`（モデルは `text-embedding-3-small` など）、
`fixed.generate.provider` を `openai`（`gpt-4o-mini` など）に変えれば、そのまま回ります。
`experiments/001_chunk_size.yaml` は最初からその条件で書いてあります。
呼び出しはタイムアウト・指数バックオフのリトライ・レート制限・トークン数記録を通る
`external.py` を必ず経由し、使ったトークン数と推定所要時間は `runs/<run_id>/result.json`
に残ります。テストでは実際のAPIを叩かず、`respx` で経路ごとモックしています。

## 限界

- 評価セットは小規模です。スコア差は傾向を見るためのもので、統計的な有意差を主張するものではありません。
- 生成側の評価は根拠一致を見るもので、回答の自然さや網羅性は測っていません。
- 外部APIに出ない既定の条件では、生成は「最上位ヒットの引用」であり、文章生成の質は一切測っていません。棄権率もこの条件では常に 0 です。
- コーパスがウィキペディアのため、生成モデルが記事を学習済みである可能性があります。検索側の指標はこの影響を受けませんが、生成側は交絡要因として残ります。
- AWS 向けの一式（[deploy/](deploy/)）は実機で一通り検証済みですが、実APIを使う経路だけは未検証です。
- **PoC のため、インフラ層のセキュリティ機構（WAF・冗長化・監査証跡など）は意図的に外しています。**扱うのが公開文書だけで、受信ポートも持たないためです。理由は [docs/00_requirements.md](docs/00_requirements.md) に書いています。

## ライセンス

MIT License. 詳細は [LICENSE](LICENSE) を参照してください。
評価用コーパスに含まれる文書のライセンスは、それぞれの出典に従います。詳細は [docs/02_evaluation.md](docs/02_evaluation.md) に記載しています。
