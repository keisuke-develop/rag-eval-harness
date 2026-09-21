# rag-eval-harness

**RAG の設定を変えたとき、検索と生成の精度がどう動くかを、同じ評価セットで比較するための道具です。**

RAG は「とりあえず動くもの」を作るのは簡単ですが、チャンクサイズを 512 から 256 に変えたときに精度が上がったのか下がったのかは、測る仕組みがないと分かりません。このリポジトリは、その「測る仕組み」のほうを主役にしています。

```bash
# 実験条件を宣言して回す
uv run rageval run experiments/001_chunk_size.yaml

# 条件 × スコアの表を出力する
uv run rageval report --out docs/04_results.md
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

**評価セットを最初に作る。** パイプラインより先に、質問と正解根拠の対を用意しました。何を良しとするかを決めないまま改善に入ると、後から「良くなった気がする」としか言えなくなるためです。評価セットの作成基準は [docs/02_evaluation.md](docs/02_evaluation.md) に書いています。

**差し替えられる場所を限定する。** チャンク戦略・埋め込みプロバイダ・Vector DB・再ランカーの4つだけを差し替え可能にし、それ以外は固定しました。比較したい軸だけを可変にしないと、何が効いたのか分からなくなります。

**実験は宣言で書く。** 条件をコードに埋め込まず YAML に出しました。実験の再現と、条件の差分の追跡ができるようにするためです。

判断の詳細は [docs/adr/](docs/adr/) に Architecture Decision Record として残しています。

## ドキュメント

| ファイル | 内容 |
|---|---|
| [docs/00_requirements.md](docs/00_requirements.md) | 何を作るか、何を作らないか |
| [docs/01_architecture.md](docs/01_architecture.md) | モジュール構成とデータフロー |
| [docs/02_evaluation.md](docs/02_evaluation.md) | 評価指標の定義と、なぜその指標か |
| [docs/03_experiment_plan.md](docs/03_experiment_plan.md) | 実験条件のマトリクス |
| [docs/04_results.md](docs/04_results.md) | 実験結果と考察 |
| [docs/adr/](docs/adr/) | 設計判断の記録 |

## 実験結果

<!-- docs/04_results.md の要約表をここに転記する。実験実施後に更新。 -->

> 実験はこれから実施します。結果が出次第、条件 × スコアの表をここに掲載します。

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
│   ├── evaluate.py      # 指標の算出
│   └── cli.py           # run / report
├── datasets/
│   ├── corpus/          # 評価用コーパス（ライセンスは docs/02 に記載）
│   └── qa_set.jsonl     # 評価セット
├── experiments/         # 実験条件の YAML
├── tests/               # pytest（unit / integration）
└── .github/workflows/   # テスト・Lint・型チェック・評価の回帰検証
```

## セットアップ

```bash
uv sync
cp .env.example .env   # APIキーを設定する
uv run pytest
```

## 限界

- 評価セットは小規模です。スコア差は傾向を見るためのもので、統計的な有意差を主張するものではありません。
- 生成側の評価は根拠一致を見るもので、回答の自然さや網羅性は測っていません。
- 埋め込みと生成は外部 API に依存します。CI ではモックで動かしています。

## ライセンス

MIT License. 詳細は [LICENSE](LICENSE) を参照してください。
評価用コーパスに含まれる文書のライセンスは、それぞれの出典に従います。詳細は [docs/02_evaluation.md](docs/02_evaluation.md) に記載しています。
