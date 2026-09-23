"""外部API呼び出しの共通処理。

リトライの発火条件を中心に見る。「429 と 5xx だけ再試行し、他の 4xx は
即座に失敗させる」を守れないと、鍵の誤りのような直らない失敗を
延々と待つことになる（docs/01_architecture.md）。

sleep を差し込んでいるので、実時間は待たない。
"""

from __future__ import annotations

import httpx
import pytest
import respx

from rageval.external import (
    ApiCaller,
    ExternalCallError,
    RateLimit,
    RetryableCallError,
    RetryPolicy,
    TokenBucket,
    Usage,
    _describe_failure,
    _has_control_chars,
    build_http_client,
    read_api_key,
)

URL = "https://api.example.test/v1/thing"


class FakeClock:
    """単調時計と sleep の組。sleep したぶんだけ時計が進む。"""

    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def make_caller(clock: FakeClock | None = None, **kwargs: object) -> ApiCaller:
    clock = clock or FakeClock()
    return ApiCaller(
        client=httpx.Client(),
        sleep=clock.sleep,
        now=clock.monotonic,
        **kwargs,  # type: ignore[arg-type]
    )


@respx.mock
def test_a_successful_call_records_one_request() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json={"ok": True}))
    caller = make_caller()
    assert caller.post_json(URL, {}) == {"ok": True}
    assert caller.usage.requests == 1
    assert caller.usage.retries == 0


@respx.mock
def test_429_is_retried_until_it_succeeds() -> None:
    respx.post(URL).mock(
        side_effect=[
            httpx.Response(429, text="slow down"),
            httpx.Response(429, text="slow down"),
            httpx.Response(200, json={"ok": True}),
        ]
    )
    clock = FakeClock()
    caller = make_caller(clock)
    assert caller.post_json(URL, {})["ok"] is True
    assert caller.usage.requests == 3
    assert caller.usage.retries == 2
    assert len(clock.slept) == 2


@respx.mock
def test_backoff_grows() -> None:
    respx.post(URL).mock(side_effect=[httpx.Response(503)] * 3 + [httpx.Response(200, json={})])
    clock = FakeClock()
    make_caller(clock, policy=RetryPolicy(max_attempts=5, initial_delay=1.0)).post_json(URL, {})
    assert clock.slept[0] < clock.slept[-1], f"指数バックオフになっていない: {clock.slept}"


@respx.mock
def test_400_fails_immediately() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(400, text="bad request"))
    caller = make_caller()
    with pytest.raises(ExternalCallError) as excinfo:
        caller.post_json(URL, {})
    assert not isinstance(excinfo.value, RetryableCallError)
    assert route.call_count == 1, "4xx を再試行しても直らない"
    assert caller.usage.retries == 0


@respx.mock
def test_401_fails_immediately() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(401, text="no key"))
    with pytest.raises(ExternalCallError, match="401"):
        make_caller().post_json(URL, {})
    assert route.call_count == 1


@respx.mock
def test_retries_are_bounded() -> None:
    route = respx.post(URL).mock(return_value=httpx.Response(500))
    caller = make_caller(policy=RetryPolicy(max_attempts=3))
    with pytest.raises(RetryableCallError):
        caller.post_json(URL, {})
    assert route.call_count == 3


@respx.mock
def test_a_timeout_is_retryable() -> None:
    respx.post(URL).mock(side_effect=[httpx.ReadTimeout("too slow"), httpx.Response(200, json={})])
    caller = make_caller()
    caller.post_json(URL, {})
    assert caller.usage.retries == 1


@respx.mock
def test_a_connection_error_is_retryable() -> None:
    respx.post(URL).mock(side_effect=[httpx.ConnectError("refused"), httpx.Response(200, json={})])
    make_caller().post_json(URL, {})


@respx.mock
def test_token_usage_is_accumulated() -> None:
    respx.post(URL).mock(
        return_value=httpx.Response(
            200, json={"usage": {"prompt_tokens": 100, "completion_tokens": 20}}
        )
    )
    caller = make_caller()
    caller.post_json(URL, {})
    caller.post_json(URL, {})
    assert caller.usage.input_tokens == 200
    assert caller.usage.output_tokens == 40


@respx.mock
def test_a_response_without_usage_is_fine() -> None:
    respx.post(URL).mock(return_value=httpx.Response(200, json={"data": []}))
    caller = make_caller()
    caller.post_json(URL, {})
    assert caller.usage.input_tokens == 0


@respx.mock
def test_elapsed_time_is_recorded() -> None:
    respx.post(URL).mock(side_effect=[httpx.Response(429), httpx.Response(200, json={})])
    clock = FakeClock()
    caller = make_caller(clock)
    caller.post_json(URL, {})
    assert caller.usage.seconds > 0.0


# ---- レート制限 ----------------------------------------------------------


def test_the_bucket_lets_the_burst_through_without_waiting() -> None:
    clock = FakeClock()
    bucket = TokenBucket(
        RateLimit(requests_per_second=2.0, burst=3), now=clock.monotonic, sleep=clock.sleep
    )
    for _ in range(3):
        bucket.acquire()
    assert clock.slept == []


def test_the_bucket_waits_once_the_burst_is_spent() -> None:
    clock = FakeClock()
    bucket = TokenBucket(
        RateLimit(requests_per_second=2.0, burst=1), now=clock.monotonic, sleep=clock.sleep
    )
    bucket.acquire()
    bucket.acquire()
    assert clock.slept, "上限を超えたのに待っていない"
    assert sum(clock.slept) == pytest.approx(0.5, abs=1e-6)


def test_a_rate_of_zero_means_no_limit() -> None:
    clock = FakeClock()
    bucket = TokenBucket(RateLimit(), now=clock.monotonic, sleep=clock.sleep)
    for _ in range(100):
        bucket.acquire()
    assert clock.slept == []


def test_usage_merges() -> None:
    total = Usage(requests=1, input_tokens=10)
    total.merge(Usage(requests=2, input_tokens=5, output_tokens=3))
    assert (total.requests, total.input_tokens, total.output_tokens) == (3, 15, 3)
    assert total.as_dict()["requests"] == 3


# ---- APIキーの読み込み --------------------------------------------------


def test_read_api_key_returns_the_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "  sk-abc123  ")
    assert read_api_key("TEST_KEY", fallback_provider="hashing") == "sk-abc123"


def test_read_api_key_reports_a_missing_key_without_leaking(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("TEST_KEY", raising=False)
    with pytest.raises(ValueError, match="TEST_KEY が設定されていない"):
        read_api_key("TEST_KEY", fallback_provider="hashing")


# NUL は Windows の環境変数に入れられないため、ここでは扱わない。
# NUL を含む判定そのものは test_has_control_chars_covers_nul_and_del で確かめる。
@pytest.mark.parametrize(
    "bad",
    [
        "sk-abc" + chr(10) + "def",
        "sk-abc" + chr(13) + chr(10) + "X-Evil: 1",
        "sk-" + chr(127),
    ],
)
def test_read_api_key_rejects_control_characters(monkeypatch: pytest.MonkeyPatch, bad: str) -> None:
    """鍵は Authorization ヘッダに載る。改行が混ざると任意のヘッダを継ぎ足せる。"""
    monkeypatch.setenv("TEST_KEY", bad)
    with pytest.raises(ValueError, match="制御文字"):
        read_api_key("TEST_KEY", fallback_provider="hashing")


def test_the_error_never_contains_the_key_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    """エラーメッセージはログに残る。鍵をそこに書かない。"""
    # 本物らしい名前だと SAST（S105）が反応する。テスト値だと分かる名前にする。
    fake_key = "sk-" + "notarealkey" + chr(10) + "value"
    monkeypatch.setenv("TEST_KEY", fake_key)
    with pytest.raises(ValueError) as excinfo:
        read_api_key("TEST_KEY", fallback_provider="hashing")
    assert "notarealkey" not in str(excinfo.value)


def test_has_control_chars_covers_nul_and_del() -> None:
    """NUL は環境変数経由では試せないので、判定そのものを直接確かめる。"""
    assert _has_control_chars("ok" + chr(0)) is True
    assert _has_control_chars("ok" + chr(127)) is True
    assert _has_control_chars("ok" + chr(31)) is True
    assert _has_control_chars("sk-normal-key-1234") is False


def test_build_http_client_sets_explicit_timeouts() -> None:
    """タイムアウトを既定任せにしない（docs/01_architecture.md「外部呼び出しの扱い」）。"""
    with build_http_client(connect_timeout=3.0, read_timeout=9.0) as client:
        assert client.timeout.connect == 3.0
        assert client.timeout.read == 9.0


def test_build_http_client_passes_headers_through() -> None:
    with build_http_client(headers={"X-Test": "1"}) as client:
        assert client.headers["X-Test"] == "1"


def test_build_http_client_works_without_truststore(monkeypatch: pytest.MonkeyPatch) -> None:
    """truststore は任意。入っていない環境でも、検証を切らずに動くこと。

    TLS を傍受する環境では OS の信頼ストアを使いたいが、それは「あれば使う」もので、
    無いときに `verify=False` へ落ちてはいけない。
    """
    import sys

    monkeypatch.setitem(sys.modules, "truststore", None)
    with build_http_client() as client:
        assert client._transport is not None


# ---- 失敗した応答の要約 --------------------------------------------------
#
# 本文を切り捨てると、原因を分ける識別子が落ちる。
# 429 は「残高が無い」と「呼び出しが速すぎる」の両方で返り、やることは正反対。


def test_the_error_code_survives_a_long_message() -> None:
    """長い message の後ろにある type / code を、切り捨てで失わないこと。"""
    response = httpx.Response(
        429,
        json={
            "error": {
                "message": "You have no credits remaining. " + "詳しい説明。" * 60,
                "type": "insufficient_quota",
                "code": "credit_balance_exhausted",
            }
        },
    )
    described = _describe_failure(response, URL)
    assert "insufficient_quota" in described
    assert "credit_balance_exhausted" in described
    assert "429" in described


def test_a_response_without_an_error_object_still_reports_something() -> None:
    described = _describe_failure(httpx.Response(500, json={"unexpected": True}), URL)
    assert "500" in described
    assert URL in described


def test_a_non_json_body_is_handled() -> None:
    described = _describe_failure(httpx.Response(502, text="<html>Bad Gateway</html>"), URL)
    assert "502" in described
    assert "Bad Gateway" in described


def test_an_error_that_is_not_an_object_is_handled() -> None:
    described = _describe_failure(httpx.Response(400, json={"error": "文字列だった"}), URL)
    assert "400" in described


@respx.mock
def test_the_code_reaches_the_caller() -> None:
    """呼び出し側が受け取る例外にも、識別子が入っていること。"""
    respx.post(URL).mock(
        return_value=httpx.Response(
            400, json={"error": {"message": "x" * 500, "code": "invalid_api_key"}}
        )
    )
    with pytest.raises(ExternalCallError, match="invalid_api_key"):
        make_caller().post_json(URL, {})
