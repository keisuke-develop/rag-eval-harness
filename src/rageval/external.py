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
class ApiCaller:
    """外部APIへのPOSTを、リトライ・レート制限・使用量記録ごと引き受ける。"""

    client: httpx.Client
    policy: RetryPolicy = field(default_factory=RetryPolicy)
    rate_limit: RateLimit = field(default_factory=RateLimit)
    usage: Usage = field(default_factory=Usage)
    sleep: Callable[[float], None] = time.sleep
    now: Callable[[], float] = time.monotonic
    _bucket: TokenBucket = field(init=False)

    def __post_init__(self) -> None:
        self._bucket = TokenBucket(self.rate_limit, now=self.now, sleep=self.sleep)

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """JSONをPOSTしてJSONを受け取る。失敗は ExternalCallError 系に変換する。"""

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
                    return self._attempt(url, payload)
        finally:
            self.usage.seconds += self.now() - started
        raise ExternalCallError("リトライが尽きた")  # pragma: no cover - Retrying が必ず送出する

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
            raise RetryableCallError(f"{response.status_code} {url}: {response.text[:200]}")
        if response.status_code >= 400:
            # 4xx は呼び出し側の誤り。待っても直らないので即座に失敗させる。
            raise ExternalCallError(f"{response.status_code} {url}: {response.text[:200]}")

        body: dict[str, Any] = response.json()
        self._record_tokens(body)
        return body

    def _record_tokens(self, body: dict[str, Any]) -> None:
        raw = body.get("usage")
        if not isinstance(raw, dict):
            return
        self.usage.input_tokens += int(raw.get("prompt_tokens", 0))
        self.usage.output_tokens += int(raw.get("completion_tokens", 0))
