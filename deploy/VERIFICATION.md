# AWS 実機検証の記録

実施日: 2026-09-21 / リージョン: ap-northeast-1
（アカウントIDは伏せる。このリポジトリは public で、IDは狙いを定めるための手がかりになる）

**結果：通った。** スタック作成 → イメージのビルドと push → タスク実行 → 結果の回収 → 片付け、
まで一通り動いた。ただし**6箇所でつまずいた**ので、その内容と直し方をここに残す。

---

## 1. 検証したこと

| 手順 | 結果 |
|---|---|
| CloudFormation スタックの作成（`rageval`） | 成功。8リソース |
| CodeBuild スタックの作成（`rageval-build`） | 成功。3リソース |
| Docker イメージのビルドと ECR への push | 成功（3回目。1回目と2回目は下記のとおり失敗） |
| ECS Fargate タスクの実行 | 成功（2回目。1回目は下記のとおり失敗） |
| S3 への結果の書き出し | 成功。3ファイル、SSE-AES256、バージョンID付き |
| 手元の実行結果との一致 | **完全一致**（下記） |
| 片付け | 完了。作成したものはすべて削除 |

### 手元と AWS で同じ結果が出るか

これが検証のいちばんの目的だった。同じ実験ファイルを手元と AWS で回し、突き合わせた。

| 項目 | 結果 |
|---|---|
| スコア（Recall@5 0.5625 / MRR 0.3365 ほか） | 一致 |
| 解決後の条件（`config`） | 一致 |
| チャンク数（154） | 一致 |
| 質問ごとの検索結果と回答（`predictions.jsonl`） | 一致（所要時間の欄を除き1文字も違わない） |

**実行環境が変わっても同じ結果が出る**ことを確認できた。
`docs/00_requirements.md` の再現性の要件を、手元だけでなくコンテナ上でも満たしている。

## 2. つまずいた点と直した内容

### (1) AWS CLI が TLS 検証で接続できない

```
SSL validation failed for https://sts.ap-northeast-1.amazonaws.com/
[SSL: CERTIFICATE_VERIFY_FAILED]
```

TLS を傍受する環境だと、AWS CLI 同梱の CA では検証が通らない。
`--no-verify-ssl` で黙らせると、傍受と攻撃を区別できなくなるので使わない。

**直し方**：OS の証明書ストアを PEM に書き出して `AWS_CA_BUNDLE` に渡す。
`deploy/make_ca_bundle.py` を追加した。

### (2) Git Bash がスラッシュ始まりの引数を Windows パスに変換する

`--log-group-name-prefix /ecs/rageval` が
`C:/Program Files/Git/ecs/rageval` に化けて、API がパターン違反で拒否した。

**直し方**：`export MSYS_NO_PATHCONV=1`。
あわせて `AWS_CA_BUNDLE` は Windows 形式のパス（`C:/...`）で渡す。
Unix 形式（`/c/...`）だと Windows ネイティブの `aws.exe` が開けない。

### (3) AWS CLI がテンプレートを cp932 で読んで落ちる

```
'cp932' codec can't decode byte 0x92 in position 76: illegal multibyte sequence
```

テンプレートに日本語コメントを書いているため。AWS CLI v2 はファイルを
OS の既定コードページで読む。**`PYTHONUTF8=1` は v2 では効かない**（単一実行ファイルのため）。

**直し方**：`export AWS_CLI_FILE_ENCODING=UTF-8`。

### (4) Dockerfile の `uv pip install` が落ちる

`RUN uv pip install --no-cache "boto3>=1.35"` が exit 2。
`uv pip install` は対象の環境を自分で決めないため、`--system` か `--python` が要る。

**直し方**：それ以前に**ロックされていない依存をイメージに入れていた**のが問題だった。
`boto3` を `pyproject.toml` の `deploy` 依存グループに移し、
`uv sync --locked --no-dev --group deploy` で入れるようにした。
`--locked` を付けてあるので、ロックとずれていればビルドが落ちる。
イメージに入るものを追跡できる状態にするため。

### (5) ECR のタグを不変にしたので、2段階のデプロイが要る

タグを `IMMUTABLE` にしたため `latest` を押し直せない。
ビルドごとに `build-<番号>` を付ける方式にした結果、
「スタックを作る → ビルドする → タグを指定してスタックを更新する」の順になる。

**直し方**：`ImageTag` の既定値を `bootstrap`（存在しない置き値）にして、
スタックだけ先に作れるようにした。ビルド後に
`--parameter-overrides ImageTag=build-3` で更新する。手順書に反映済み。

### (6) 読み取り専用ルートFS + 非root + Fargate ボリュームは両立しない

```
PermissionError: [Errno 13] Permission denied: '/tmp/runs'
```

`ReadonlyRootFilesystem: true` にしたうえで `/tmp` にタスクボリュームを載せ、
非rootユーザー（uid 10001）で動かしていた。**Fargate のタスクボリュームは
root 所有 0755 で作られ、所有者を指定できない**ため、非rootユーザーが書けない。

**直し方**：`ReadonlyRootFilesystem: false` にしてボリュームを外し、非rootは維持した。
権限昇格の余地を狭める非rootのほうを優先している。判断の理由は
[../docs/06_security_review.md](../docs/06_security_review.md) に書いた。

## 3. かかった費用

| 項目 | 実績 | 概算 |
|---|---|---|
| CodeBuild（general1.small） | 3回 / 計 約5.3分 | $0.027 |
| Fargate タスク（0.5 vCPU / 1GB） | 2回 / 各約75秒 | $0.001 |
| ECR ストレージ | 326MB × 3イメージ × 1日 | $0.003 |
| S3・CloudWatch Logs | 数百KB | ほぼ 0 |
| **合計** | | **約 $0.03** |

見積もりは $0.05 未満だったので、その範囲に収まった。

## 4. 片付け

作成したものはすべて削除した。既存のリソース（他プロジェクトの S3 バケット33個、
ECR リポジトリ2件など）には一切触れていない。削除の記録は §6 を参照。

## 5. 残っている宿題

| 項目 | 内容 |
|---|---|
| ベースイメージの脆弱性 | ECR のスキャンで CRITICAL 4件 / HIGH 14件。すべて `python:3.12-slim-bookworm` の OS パッケージ由来。`apt-get upgrade` を当てても**1件も減らなかった**（Debian に修正版がまだ無い）。[../docs/06_security_review.md](../docs/06_security_review.md) を参照 |
| 読み取り専用ルートFS | いまは無効。両立させるなら EFS のアクセスポイント（`PosixUser` で所有者を指定できる）が要る |
| 実APIでの検証 | 未実施。SSM から鍵を渡す経路はテンプレートに入っているが、実際に通していない |
| CI からの自動ビルド | 未実施。いまはソースを手で zip にして S3 に置いている |

## 6. 削除したリソース

削除前に一覧を出して確認したうえで、以下を削除した。

| 種別 | 名前 |
|---|---|
| CloudFormation スタック | `rageval-build` |
| CloudFormation スタック | `rageval` |
| S3 バケット | `rageval-results-<アカウントID>-ap-northeast-1`（中身ごと） |
| ECR リポジトリ | `rageval`（イメージ2件ごと。build-1 は push 前に失敗したため存在しない） |
| IAM ロール | `rageval-execution` / `rageval-task` / `rageval-build`（スタック削除に伴う） |
| CloudWatch ロググループ | `/ecs/rageval` / `/aws/codebuild/rageval-build`（スタック削除に伴う） |
| ECS クラスタ・タスク定義 | `rageval`（スタック削除に伴う） |

削除の実行結果は本ファイル末尾の「削除の実行記録」に追記する。

### 削除の実行記録

削除前に対象の一覧を出し、既存リソースが含まれていないことを確認したうえで実行した。

```
[1/5] スタック rageval-build を削除   完了
[2/5] スタック rageval を削除         完了
[3/5] S3 バケットの中身を削除         6 バージョン（バージョニング有効のため全バージョン）
[4/5] S3 バケットを削除               完了
[5/5] ECR リポジトリを削除            完了
```

削除後の確認。

| 対象 | 結果 |
|---|---|
| スタック `rageval` / `rageval-build` | 削除済み |
| S3 バケット `rageval-results-...` | 削除済み |
| ECR リポジトリ `rageval` | 削除済み |
| IAM ロール `rageval-execution` / `rageval-task` / `rageval-build` | 削除済み |
| ECS クラスタ | 0件（検証前の状態） |
| CodeBuild プロジェクト | 削除済み |

既存リソースの無事も確認した。

| 対象 | 検証前 | 検証後 |
|---|---|---|
| S3 バケット数 | 33 | **33** |
| ECR リポジトリ | `senju-md-mdconvert-convert-image`, `senju-ai-demo-convert` | **同じ2件** |

**課金が残るものは無い。**

> バージョニングを有効にしたバケットは、`aws s3 rm --recursive` だけでは消えない。
> 旧バージョンと削除マーカーが残り、`rb` が失敗する。
> `list-object-versions` で全バージョンを列挙して消す必要がある。
