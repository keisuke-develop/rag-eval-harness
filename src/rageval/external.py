"""外部API呼び出しの共通処理。

埋め込みと生成はどちらも外部APIに出るため、以下を1箇所に集める
（docs/01_architecture.md「外部呼び出しの扱い」）。

| 対策 | ここでの実装 |
|---|---|
| タイムアウト | 接続・読み取りを分けて明示する |
| リトライ | 指数バックオフ。429 と 5xx のみ対象、他の 4xx は即座に失敗させる |
| レート制限 | トークンバケットで秒あたりのリクエスト数を抑える |
| 使用量の記録 | リクエスト数・リトライ回数・トークン数・所要時間を積む |

時刻と sleep を差し込めるようにしてあるのは、リトライとレート制限の
発火条件をテストから実時間を待たずに確認するため。
"""

from __future__ import annotations

import json
import os
import ssl
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential_jitter

#: リトライの対象にするHTTPステータス。これ以外の 4xx は呼び出し側の誤りなので即座に失敗させる。
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504})


def _has_control_chars(text: str) -> bool:
    """制御文字を含むか。正規表現を使わずに書いているのは、エスケープの取り違えを避けるため。"""
    return any(ord(c) < 0x20 or ord(c) == 0x7F for c in text)


def read_api_key(env_name: str, *, fallback_provider: str) -> str:
    """環境変数からAPIキーを読み、ヘッダに載せて安全な形か確かめる。

    鍵は `Authorization` ヘッダに入る。改行が混ざっていると、そこから
    任意のヘッダを継ぎ足せてしまう（HTTPヘッダ・インジェクション）。
    実際には httpx 側でも弾かれるが、**弾かれる場所が遠いほど原因が分かりにくい**ので、
    読んだ直後に確かめる。

    エラーメッセージに鍵そのものを載せないこと。ログに残る。
    """
    key = os.environ.get(env_name, "").strip()
    if not key:
        raise ValueError(
            f"{env_name} が設定されていない。"
            f"リポジトリ直下の .env に「{env_name}=...」の1行を置くか、"
            f"環境変数で渡すか、provider: {fallback_provider} に切り替えること"
        )
    if _has_control_chars(key):
        raise ValueError(
            f"{env_name} に制御文字が含まれている。"
            "改行などが混ざると HTTP ヘッダの改ざんに使われうるので受け付けない"
        )
    return key


def _describe_failure(response: httpx.Response, url: str) -> str:
    """失敗した応答を1行にする。**識別子を本文の切り捨てで失わないこと。**

    OpenAI の応答は `{"error": {"message": ..., "type": ..., "code": ...}}` の形で、
    長い `message` が先に来る。本文をそのまま切ると `type` と `code` が落ちる。

    この2つは捨ててはいけない。たとえば 429 は「残高が無い」と「呼び出しが速すぎる」の
    両方で返るが、**やることは正反対**（前者は待っても直らない）。
    区別できるのは `type` / `code` だけなので、切る前に取り出して先頭に置く。
    """
    head = f"{response.status_code} {url}"
    try:
        error = response.json().get("error", {})
    except (ValueError, AttributeError):
        return f"{head}: {response.text[:200]}"
    if not isinstance(error, dict):
        return f"{head}: {response.text[:200]}"

    marks = [str(error[k]) for k in ("type", "code") if error.get(k)]
    message = str(error.get("message", ""))[:200]
    if marks:
        return f"{head} [{' / '.join(marks)}]: {message}"
    return f"{head}: {message or response.text[:200]}"


class ExternalCallError(RuntimeError):
    """外部API呼び出しの失敗。リトライしても意味がないもの。"""


class RetryableCallError(ExternalCallError):
    """時間を置けば直る可能性のある失敗（429 / 5xx / 接続エラー）。"""


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 5
    initial_delay: float = 0.5
    max_delay: float = 8.0


@dataclass(frozen=True)
class RateLimit:
    """秒あたりのリクエスト数の上限。0 以下なら制限しない。"""

    requests_per_second: float = 0.0
    burst: int = 1


@dataclass
class Usage:
    """1回の実験で外部APIをどれだけ使ったか。result.json に載せる。"""

    requests: int = 0
    retries: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    def merge(self, other: Usage) -> None:
        self.requests += other.requests
        self.retries += other.retries
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.seconds += other.seconds

    def as_dict(self) -> dict[str, float | int]:
        return {
            "requests": self.requests,
            "retries": self.retries,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "seconds": round(self.seconds, 3),
        }


class TokenBucket:
    """秒あたりのリクエスト数を抑えるトークンバケット。"""

    def __init__(
        self,
        limit: RateLimit,
        *,
        now: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._limit = limit
        self._now = now
        self._sleep = sleep
        self._tokens = float(limit.burst)
        self._updated = now()

    def acquire(self) -> None:
        if self._limit.requests_per_second <= 0:
            return
        while True:
            current = self._now()
            self._tokens = min(
                float(self._limit.burst),
                self._tokens + (current - self._updated) * self._limit.requests_per_second,
            )
            self._updated = current
            if self._tokens >= 1.0:
                self._tokens -= 1.0
                return
            self._sleep((1.0 - self._tokens) / self._limit.requests_per_second)


def build_http_client(
    *,
    connect_timeout: float = 10.0,
    read_timeout: float = 60.0,
    headers: dict[str, str] | None = None,
) -> httpx.Client:
    """タイムアウトを明示したHTTPクライアントを作る。

    TLS 傍受のある環境では certifi 同梱の CA では検証に失敗するため、
    truststore が入っていれば OS 側の証明書ストアを使う。
    """
    verify: Any = True
    try:
        import truststore

        verify = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        pass
    return httpx.Client(
        timeout=httpx.Timeout(read_timeout, connect=connect_timeout),
        headers=headers or {},
        verify=verify,
    )


@dataclass
class _RetryingCaller:
    """リトライ・レート制限・使用量記録の土台。

    提供元ごとに呼び方は違っても（HTTP を直に叩く / SDK を通す）、
    **守るべきことは同じ**（docs/01_architecture.md「外部呼び出しの扱い」）。
    その同じ部分だけをここに置き、呼び方の違いは派生側で持つ。
    """

    policy: RetryPolicy = field(default_factory=RetryPolicy)
    rate_limit: RateLimit = field(default_factory=RateLimit)
    usage: Usage = field(default_factory=Usage)
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic
    _bucket: TokenBucket = field(init=False)

    def __post_init__(self) -> None:
        self._bucket = TokenBucket(self.rate_limit, now=self.now, sleep=self.sleep)

    def _with_retry(self, operation: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        """1回ぶんの呼び出しを、リトライと時間の記録でくるむ。"""

        def _record_retry(state: RetryCallState) -> None:
            self.usage.retries += 1

        retrying = Retrying(
            stop=stop_after_attempt(self.policy.max_attempts),
            wait=wait_exponential_jitter(
                initial=self.policy.initial_delay, max=self.policy.max_delay
            ),
            retry=retry_if_exception_type(RetryableCallError),
            sleep=self.sleep,
            before_sleep=_record_retry,
            reraise=True,
        )
        started = self.now()
        try:
            for attempt in retrying:
                with attempt:
                    return operation()
        finally:
            self.usage.seconds += self.now() - started
        raise ExternalCallError("リトライが尽きた")  # pragma: no cover - Retrying が必ず送出する


@dataclass
class ApiCaller(_RetryingCaller):
    """外部APIへのPOSTを、リトライ・レート制限・使用量記録ごと引き受ける。"""

    client: httpx.Client = field(default_factory=httpx.Client)

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """JSONをPOSTしてJSONを受け取る。失敗は ExternalCallError 系に変換する。"""
        return self._with_retry(lambda: self._attempt(url, payload))

    def _attempt(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._bucket.acquire()
        self.usage.requests += 1
        try:
            response = self.client.post(url, json=payload)
        except httpx.TimeoutException as exc:
            raise RetryableCallError(f"タイムアウト: {url}") from exc
        except httpx.TransportError as exc:
            raise RetryableCallError(f"接続エラー: {url} ({exc})") from exc

        if response.status_code in RETRYABLE_STATUS:
            raise RetryableCallError(_describe_failure(response, url))
        if response.status_code >= 400:
            # 4xx は呼び出し側の誤り。待っても直らないので即座に失敗させる。
            raise ExternalCallError(_describe_failure(response, url))

        body: dict[str, Any] = response.json()
        self._record_tokens(body)
        return body

    def _record_tokens(self, body: dict[str, Any]) -> None:
        raw = body.get("usage")
        if not isinstance(raw, dict):
            return
        self.usage.input_tokens += int(raw.get("prompt_tokens", 0))
        self.usage.output_tokens += int(raw.get("completion_tokens", 0))


# ---- Bedrock ------------------------------------------------------------
#
# 鍵を持たずに呼べるのが Bedrock を選んだ理由の1つ（ADR 0002 の追記を参照）。
# 認証は手元なら AWS の認証情報、AWS 上ならタスクロールで、
# `.env` に秘密を置く必要がない。

#: 待てば直る見込みのあるもの。これ以外は即座に失敗させる。
RETRYABLE_BEDROCK_ERRORS = frozenset(
    {
        "ThrottlingException",
        "TooManyRequestsException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelTimeoutException",
        "ModelNotReadyException",
    }
)


def build_bedrock_client(
    *,
    region: str,
    connect_timeout: float = 10.0,
    read_timeout: float = 120.0,
) -> Any:  # botocore のクライアントは型を持たないため Any
    """Bedrock の呼び出し口を作る。

    SDK 側のリトライは切ってある（`total_max_attempts: 1`）。
    切らないと SDK が黙って再試行し、**こちらの記録に残らない**。
    botocore の `max_attempts` は「リトライの回数」で、1 を渡すと合計2回になる。
    切りたいときに指定するのは `total_max_attempts` のほう。
    リトライは `_RetryingCaller` に一本化して、回数も待ち時間も
    result.json に出せるようにする。

    生成は応答が遅いので、読み取りの待ちを埋め込みより長めに取ってある。
    """
    import boto3
    from botocore.config import Config

    return boto3.client(
        "bedrock-runtime",
        region_name=region,
        config=Config(
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            retries={"total_max_attempts": 1, "mode": "standard"},
        ),
    )


@dataclass
class BedrockCaller(_RetryingCaller):
    """Bedrock の呼び出しを、リトライ・レート制限・使用量記録ごと引き受ける。

    `ApiCaller` と守るものは同じ。違うのは呼び方だけ。
    """

    client: Any = None
    region: str = "ap-northeast-1"

    def invoke_model(self, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """埋め込み用。`InvokeModel` は本文を JSON のバイト列でやりとりする。"""
        return self._with_retry(lambda: self._invoke(model_id, payload))

    def converse(
        self,
        model_id: str,
        *,
        system: str,
        user: str,
        temperature: float,
        max_tokens: int,
    ) -> dict[str, Any]:
        """生成用。`Converse` は提供元ごとの本文の形の違いを吸収してくれる。"""
        return self._with_retry(
            lambda: self._converse(model_id, system, user, temperature, max_tokens)
        )

    def _invoke(self, model_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        self._bucket.acquire()
        self.usage.requests += 1
        response = self._guard(
            model_id,
            lambda: self.client.invoke_model(modelId=model_id, body=json.dumps(payload)),
        )
        body: dict[str, Any] = json.loads(response["body"].read())
        # Titan は入力のトークン数を返す。Cohere は返さない。返ったものだけ数える。
        self.usage.input_tokens += int(body.get("inputTextTokenCount", 0))
        return body

    def _converse(
        self, model_id: str, system: str, user: str, temperature: float, max_tokens: int
    ) -> dict[str, Any]:
        self._bucket.acquire()
        self.usage.requests += 1
        body: dict[str, Any] = self._guard(
            model_id,
            lambda: self.client.converse(
                modelId=model_id,
                system=[{"text": system}],
                messages=[{"role": "user", "content": [{"text": user}]}],
                inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
            ),
        )
        raw = body.get("usage")
        if isinstance(raw, dict):
            self.usage.input_tokens += int(raw.get("inputTokens", 0))
            self.usage.output_tokens += int(raw.get("outputTokens", 0))
        return body

    def _guard(self, model_id: str, call: Callable[[], Any]) -> Any:
        """SDK の例外を、こちらの2種類に振り分ける。"""
        from botocore.exceptions import BotoCoreError, ClientError

        try:
            return call()
        except ClientError as exc:
            code = str(exc.response.get("Error", {}).get("Code", ""))
            message = str(exc.response.get("Error", {}).get("Message", ""))[:200]
            described = f"{code} {model_id}: {message}"
            if code in RETRYABLE_BEDROCK_ERRORS:
                raise RetryableCallError(described) from exc
            raise ExternalCallError(described) from exc
        except BotoCoreError as exc:
            # 接続できない・読み取りが間に合わない、など。待てば直る見込みがある。
            raise RetryableCallError(f"接続エラー {model_id}: {exc}") from exc
