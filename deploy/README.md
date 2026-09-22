# AWS で実験を回す

**この道具は CLI で完結する。** AWS はあくまで「手元の代わりに回す場所」であり、
無くても困らない。ここに置いてあるのは、必要になったときに薄く載せるための一式。

> **状態：検証済み（2026-09-21）。** 東京リージョンに実際に作り、イメージをビルドし、
> タスクを回し、結果を回収し、片付けるところまで通した。手元での実行と結果が
> 1文字も違わないことを確認している。つまずいた6点と直し方は
> [VERIFICATION.md](VERIFICATION.md) に、セキュリティの評価は
> [../docs/06_security_review.md](../docs/06_security_review.md) にある。

---

## 何を作るか

常駐するものを作らない。実験は都度起動して、終われば止まる。

```
  ローカル / CI
      │  docker push
      ▼
  ┌─────────┐      RunTask     ┌──────────────────┐
  │   ECR   │ ───────────────▶ │ ECS Fargate タスク │
  └─────────┘                  │  rageval run ...  │
                               └────────┬──────────┘
        ┌───────────────┐               │ 結果を置く
        │ SSM Parameter │──── 鍵 ───────┤
        │  (SecureString)│              ▼
        └───────────────┘        ┌────────────┐
                                 │     S3     │
          CloudWatch Logs ◀──────┴────────────┘
```

| リソース | 役割 | 止めている間の課金 |
|---|---|---|
| ECR リポジトリ | イメージの置き場 | ストレージのみ |
| ECS クラスタ | タスクの置き場 | なし |
| ECS タスク定義 | 実験1本の起動条件 | なし |
| S3 バケット | 実験結果の保管 | 置いた量のみ |
| CloudWatch Logs | 実行ログ | 保持期間ぶんのみ |
| IAM ロール2つ | 取得・書き込みの権限 | なし |

**VPC・NAT ゲートウェイ・ロードバランサは作らない。** タスクはパブリックサブネットに
`assignPublicIp=ENABLED` で置けば外へ出られる。NAT ゲートウェイは置くだけで
月 $35 前後かかるので、この用途では割に合わない。

## 前提

- リージョンは **東京 (ap-northeast-1) のみ**。テンプレートに `Rules` で縛ってある
- `aws` CLI が使えること
- CloudFormation・ECR・ECS・S3・IAM・Logs・CodeBuild を作れる権限
- **`docker` は要らない。** イメージは CodeBuild で作る（手元に Docker が無い前提）

### Windows の Git Bash から叩く場合の設定

3つとも実際につまずいた箇所（[VERIFICATION.md](VERIFICATION.md)）。

```bash
export MSYS_NO_PATHCONV=1          # /ecs/rageval が Windows パスに化けるのを止める
export AWS_CLI_FILE_ENCODING=UTF-8 # 日本語コメント入りテンプレートを cp932 で読ませない
export AWS_DEFAULT_REGION=ap-northeast-1
```

`PYTHONUTF8=1` は AWS CLI v2 では効かない（単一実行ファイルのため）。

### TLS 傍受のある環境では CA バンドルが要る

社内プロキシなどで TLS が傍受される環境だと、AWS CLI が
`SSL: CERTIFICATE_VERIFY_FAILED` で接続できない。OS の証明書ストアを
PEM に書き出して `AWS_CA_BUNDLE` に渡す。

```bash
python deploy/make_ca_bundle.py ca-bundle.pem
export AWS_CA_BUNDLE=$PWD/ca-bundle.pem
```

`--no-verify-ssl` で黙らせないこと。証明書の検証を切ると、
傍受されていることと攻撃されていることを区別できなくなる。

## 手順

### 1. スタックを作る

```bash
VPC=$(aws ec2 describe-vpcs --region ap-northeast-1 \
  --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)

aws cloudformation deploy \
  --region ap-northeast-1 \
  --stack-name rageval \
  --template-file deploy/cloudformation/rageval.yaml \
  --parameter-overrides VpcId=$VPC \
  --capabilities CAPABILITY_NAMED_IAM
```

外部APIを使わない条件（`experiments/ci_baseline.yaml`）だけを回すなら、これで足りる。

`VpcId` を渡すと、送信だけを許すセキュリティグループが一緒に作られる。省略しても
スタックは通り、その場合は手順3で既定のSGを使うことになる。

出力を控える。

```bash
aws cloudformation describe-stacks --region ap-northeast-1 \
  --stack-name rageval --query 'Stacks[0].Outputs' --output table
```

### 2. イメージを作って push する

```bash
REGION=ap-northeast-1
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
REPO=$ACCOUNT.dkr.ecr.$REGION.amazonaws.com/rageval

aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com

docker build -t $REPO:latest .
docker push $REPO:latest
```

コーパスと評価セットはイメージに焼き込んでいる。実行時に外部から取りに行かないので、
ネットワークが無くても検索側の実験は回る。

### 3. 実験を1本回す

サブネットは既定VPCのもので足りる。セキュリティグループは、スタックに `VpcId` を
渡してあれば送信専用のものが作られているのでそちらを使う。既定のSGは同じSG内からの
受信を許すので、送信しかしないタスクには広すぎる。

```bash
SUBNET=$(aws ec2 describe-subnets --region ap-northeast-1 \
  --filters Name=default-for-az,Values=true \
  --query 'Subnets[0].SubnetId' --output text)
SG=$(aws cloudformation describe-stacks --region ap-northeast-1 \
  --stack-name rageval \
  --query 'Stacks[0].Outputs[?OutputKey==`TaskSecurityGroupId`].OutputValue' \
  --output text)
# VpcId を渡さずに作った場合はここが空になる。その場合は既定のSGを使う
if [ -z "$SG" ] || [ "$SG" = "None" ]; then
  SG=$(aws ec2 describe-security-groups --region ap-northeast-1 \
    --filters Name=group-name,Values=default \
    --query 'SecurityGroups[0].GroupId' --output text)
fi

aws ecs run-task \
  --region ap-northeast-1 \
  --cluster rageval \
  --task-definition rageval \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[$SUBNET],securityGroups=[$SG],assignPublicIp=ENABLED}" \
  --overrides '{"containerOverrides":[{"name":"rageval","command":["experiments/ci_baseline.yaml"]}]}'
```

`command` を差し替えれば、別の実験ファイルを回せる。

### 4. 結果を取り出す

```bash
BUCKET=$(aws cloudformation describe-stacks --region ap-northeast-1 \
  --stack-name rageval --query 'Stacks[0].Outputs[?OutputKey==`ResultsBucketName`].OutputValue' \
  --output text)

aws s3 ls s3://$BUCKET/runs/
aws s3 cp s3://$BUCKET/runs/<タイムスタンプ>/ ./runs/ --recursive
uv run rageval report
```

結果は実行ごとに `runs/<タイムスタンプ>/` へ分けて置く。同じ `run_id` を
何度回しても上書きしない。過去の結果が消えると条件間の比較ができなくなるため。

## 実APIを使う場合

鍵は SSM パラメータストアの SecureString に置く。Secrets Manager でもできるが、
1シークレットあたり月 $0.40 かかるのに対し、SSM の標準パラメータは無料。

```bash
aws ssm put-parameter --region ap-northeast-1 \
  --name /rageval/openai-api-key --type SecureString --value "sk-..."

aws cloudformation deploy \
  --region ap-northeast-1 \
  --stack-name rageval \
  --template-file deploy/cloudformation/rageval.yaml \
  --capabilities CAPABILITY_NAMED_IAM \
  --parameter-overrides OpenAIApiKeyParameterName=/rageval/openai-api-key
```

鍵はタスク起動時に環境変数として渡される。イメージには入らない。

## コストの目安

**2026年9月時点の東京リージョンの公表価格から概算したもの。実額は
[Fargate の料金表](https://aws.amazon.com/jp/fargate/pricing/)で確認すること。**

| 項目 | 目安 |
|---|---|
| 止めている間 | 月 $0.1 前後（ECR に 1GB 弱のイメージ 1〜5 個 + S3 数MB + ログ） |
| 実験を1本回す | $0.01 未満（0.5 vCPU / 1GB を10分） |
| 実験計画6本を全部回す | $0.1 未満（AWS 側のみ。外部APIの料金は別） |

**AWS 側の費用はほぼ無視できる。効いてくるのは外部APIの料金のほうなので、
条件を増やす前に `result.json` の `usage` でトークン数を確認すること。**

抑えるための既定：

- Container Insights は無効（有効にすると CloudWatch のメトリクス料金がかかる）
- ログの保持は14日、S3 の結果は90日で自動削除
- ECR は直近5イメージだけ残す
- 常駐サービス（ALB・NAT・RDS など）を一切作らない

## 片付け

```bash
aws cloudformation delete-stack --region ap-northeast-1 --stack-name rageval
```

S3 バケットと ECR リポジトリは `DeletionPolicy: Retain` にしてあるので、
スタックを消しても残る。実験結果を巻き添えで消さないため。中身ごと消す場合：

```bash
aws s3 rm s3://$BUCKET --recursive && aws s3 rb s3://$BUCKET
aws ecr delete-repository --region ap-northeast-1 --repository-name rageval --force
```

## 検証の状況

| 対象 | 状態 |
|---|---|
| `cfn-lint`（両テンプレート） | 通過（エラー・警告なし） |
| スタックの作成 | **検証済み**（`rageval` 8リソース / `rageval-build` 3リソース） |
| イメージのビルドと push | **検証済み**（CodeBuild、3回目で成功） |
| ECS でのタスク実行 | **検証済み**（2回目で成功、終了コード 0） |
| S3 への結果の書き出しと回収 | **検証済み**（SSE-AES256、バージョンID付き） |
| 手元との結果一致 | **検証済み**（所要時間を除き1文字も違わない） |
| 片付け | **検証済み**（作成したものはすべて削除） |
| 実APIを使う経路（SSM 経由の鍵） | **未検証** |

つまずいた6点とその直し方は [VERIFICATION.md](VERIFICATION.md) に残してある。
セキュリティの評価と、受け入れた残リスクは
[../docs/06_security_review.md](../docs/06_security_review.md) を参照。
