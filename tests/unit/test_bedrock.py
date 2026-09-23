"""Amazon Bedrock 経路。

`boto3` は必ずモックする。CI で実際の API を叩かない（CLAUDE.md）。

ここで確かめたいのは3つ。

1. **Titan と Cohere で本文の形が違う**のを、こちらが正しく吸収しているか
2. 失敗を「待てば直る」と「直らない」に正しく振り分けているか
3. 使用量（リクエスト数・リトライ回数・トークン数）を落とさず記録しているか

3つめを外すと、実験の費用が後から追えなくなる。
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from rageval.config import EmbedderConfig, GenerateConfig
from rageval.embed import BedrockEmbedder, build_embedder
from rageval.external import (
    BedrockCaller,
    ExternalCallError,
    RetryableCallError,
    RetryPolicy,
)
from rageval.generate import _BedrockGenerator, build_generator, extract_json
from rageval.ingest import Chunk
from rageval.store import SearchHit

TITAN = "amazon.titan-embed-text-v2:0"
COHERE = "cohere.embed-multilingual-v3"
CLAUDE = "jp.anthropic.claude-haiku-4-5-20251001-v1:0"


class _Body:
    """`invoke_model` の戻り値は、読み取り可能なストリームを持つ。"""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._raw


class FakeBedrock:
    """boto3 の bedrock-runtime クライアントの代わり。"""

    def __init__(self) -> None:
        self.invocations: list[tuple[str, dict[str, Any]]] = []
        self.converses: list[dict[str, Any]] = []
        self.invoke_responses: list[Any] = []
        self.converse_responses: list[Any] = []

    # 引数名は boto3 の API に合わせる（camelCase）。
    def invoke_model(self, *, modelId: str, body: str) -> dict[str, Any]:
        self.invocations.append((modelId, json.loads(body)))
        nxt = self.invoke_responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return {"body": _Body(nxt)}

    def converse(self, **kwargs: Any) -> dict[str, Any]:
        self.converses.append(kwargs)
        nxt = self.converse_responses.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return dict(nxt)


def caller(client: FakeBedrock, **kwargs: Any) -> BedrockCaller:
    return BedrockCaller(client=client, sleep=lambda _: None, **kwargs)


def client_error(code: str) -> Exception:
    from botocore.exceptions import ClientError

    error: Exception = ClientError(
        {"Error": {"Code": code, "Message": "だめでした"}}, "InvokeModel"
    )
    return error


# ---- 埋め込み：Titan と Cohere で本文の形が違う ---------------------------


def test_titan_sends_one_text_at_a_time_and_reads_embedding() -> None:
    client = FakeBedrock()
    client.invoke_responses = [
        {"embedding": [1.0, 0.0], "inputTextTokenCount": 7},
        {"embedding": [0.0, 1.0], "inputTextTokenCount": 5},
    ]
    embedder = BedrockEmbedder(model=TITAN, caller=caller(client))
    assert embedder.embed(["ひとつめ", "ふたつめ"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert [payload for _, payload in client.invocations] == [
        {"inputText": "ひとつめ"},
        {"inputText": "ふたつめ"},
    ], "Titan はまとめて送れないので1件ずつ"


def test_cohere_sends_a_batch_and_reads_embeddings() -> None:
    client = FakeBedrock()
    client.invoke_responses = [{"embeddings": [[1.0, 0.0], [0.0, 1.0]]}]
    embedder = BedrockEmbedder(model=COHERE, caller=caller(client))
    assert embedder.embed(["ひとつめ", "ふたつめ"]) == [[1.0, 0.0], [0.0, 1.0]]
    _, payload = client.invocations[0]
    assert payload["texts"] == ["ひとつめ", "ふたつめ"], "Cohere はまとめて送れる"
    assert payload["input_type"] == "search_document", "検索用途の指定が要る"


def test_cohere_respects_the_batch_size() -> None:
    client = FakeBedrock()
    client.invoke_responses = [{"embeddings": [[1.0]]}, {"embeddings": [[2.0]]}]
    embedder = BedrockEmbedder(model=COHERE, caller=caller(client), batch_size=1)
    assert embedder.embed(["a", "b"]) == [[1.0], [2.0]]
    assert len(client.invocations) == 2


def test_a_mismatched_batch_is_rejected() -> None:
    """送った数と返った数が違うのを黙って通すと、ベクトルと本文の対応が崩れる。"""
    client = FakeBedrock()
    client.invoke_responses = [{"embeddings": [[1.0]]}]
    embedder = BedrockEmbedder(model=COHERE, caller=caller(client))
    with pytest.raises(ExternalCallError, match="対応していない"):
        embedder.embed(["a", "b"])


def test_titan_without_an_embedding_is_rejected() -> None:
    client = FakeBedrock()
    client.invoke_responses = [{"message": "なにかおかしい"}]
    embedder = BedrockEmbedder(model=TITAN, caller=caller(client))
    with pytest.raises(ExternalCallError, match="返ってこなかった"):
        embedder.embed(["a"])


def test_the_input_token_count_is_recorded() -> None:
    client = FakeBedrock()
    client.invoke_responses = [
        {"embedding": [1.0], "inputTextTokenCount": 7},
        {"embedding": [1.0], "inputTextTokenCount": 5},
    ]
    call = caller(client)
    BedrockEmbedder(model=TITAN, caller=call).embed(["a", "b"])
    assert call.usage.requests == 2
    assert call.usage.input_tokens == 12


def test_cohere_without_a_token_count_is_fine() -> None:
    """Cohere はトークン数を返さない。返ったものだけ数える。"""
    client = FakeBedrock()
    client.invoke_responses = [{"embeddings": [[1.0]]}]
    call = caller(client)
    BedrockEmbedder(model=COHERE, caller=call).embed(["a"])
    assert call.usage.requests == 1
    assert call.usage.input_tokens == 0


# ---- 失敗の振り分け -------------------------------------------------------


@pytest.mark.parametrize("code", ["ThrottlingException", "ServiceUnavailableException"])
def test_a_transient_failure_is_retried(code: str) -> None:
    client = FakeBedrock()
    client.invoke_responses = [client_error(code), {"embedding": [1.0]}]
    call = caller(client)
    BedrockEmbedder(model=TITAN, caller=call).embed(["a"])
    assert call.usage.retries == 1
    assert call.usage.requests == 2


@pytest.mark.parametrize("code", ["AccessDeniedException", "ValidationException"])
def test_a_permanent_failure_stops_at_once(code: str) -> None:
    """権限が無い・引数が違うのは、待っても直らない。"""
    client = FakeBedrock()
    client.invoke_responses = [client_error(code)]
    call = caller(client)
    with pytest.raises(ExternalCallError) as excinfo:
        BedrockEmbedder(model=TITAN, caller=call).embed(["a"])
    assert not isinstance(excinfo.value, RetryableCallError)
    assert code in str(excinfo.value)
    assert call.usage.retries == 0


def test_retries_are_bounded() -> None:
    client = FakeBedrock()
    client.invoke_responses = [client_error("ThrottlingException")] * 3
    call = caller(client, policy=RetryPolicy(max_attempts=3))
    with pytest.raises(RetryableCallError):
        BedrockEmbedder(model=TITAN, caller=call).embed(["a"])
    assert call.usage.requests == 3


def test_a_connection_problem_is_retryable() -> None:
    from botocore.exceptions import ConnectTimeoutError

    client = FakeBedrock()
    client.invoke_responses = [
        ConnectTimeoutError(endpoint_url="https://bedrock-runtime.example"),
        {"embedding": [1.0]},
    ]
    call = caller(client)
    BedrockEmbedder(model=TITAN, caller=call).embed(["a"])
    assert call.usage.retries == 1


def test_the_model_id_is_in_the_message() -> None:
    """どのモデルで落ちたのかが分からないと、条件を絞り込めない。"""
    client = FakeBedrock()
    client.invoke_responses = [client_error("AccessDeniedException")]
    with pytest.raises(ExternalCallError, match=TITAN):
        BedrockEmbedder(model=TITAN, caller=caller(client)).embed(["a"])


# ---- 生成 -----------------------------------------------------------------


def hit(chunk_id: str, text: str) -> SearchHit:
    return SearchHit(
        chunk=Chunk(chunk_id=chunk_id, doc_id="d", index=0, start=0, end=len(text), text=text),
        score=0.9,
        rank=1,
    )


def converse_response(text: str, *, tokens: tuple[int, int] = (120, 30)) -> dict[str, Any]:
    return {
        "output": {"message": {"content": [{"text": text}]}},
        "usage": {"inputTokens": tokens[0], "outputTokens": tokens[1]},
    }


def generate_config(**kwargs: Any) -> GenerateConfig:
    return GenerateConfig.model_validate({"provider": "bedrock", "model": CLAUDE, **kwargs})


def test_the_structured_output_is_parsed() -> None:
    client = FakeBedrock()
    client.converse_responses = [
        converse_response('{"answer": "3776m", "cited_chunk_ids": ["a#c001"]}')
    ]
    answer = _BedrockGenerator(config=generate_config(), caller=caller(client))(
        "標高は？", [hit("a#c001", "本文")]
    )
    assert answer.cited_chunk_ids == ["a#c001"]


def test_the_fixed_settings_are_sent() -> None:
    client = FakeBedrock()
    client.converse_responses = [converse_response('{"answer": "x"}')]
    config = generate_config(temperature=0.0, max_tokens=256)
    _BedrockGenerator(config=config, caller=caller(client))("質問", [hit("a", "本文")])
    sent = client.converses[0]
    assert sent["modelId"] == CLAUDE
    assert sent["inferenceConfig"] == {"temperature": 0.0, "maxTokens": 256}
    assert sent["system"][0]["text"], "system プロンプトを渡していない"


def test_the_token_usage_is_recorded() -> None:
    client = FakeBedrock()
    client.converse_responses = [converse_response('{"answer": "x"}', tokens=(100, 20))]
    call = caller(client)
    _BedrockGenerator(config=generate_config(), caller=call)("質問", [hit("a", "本文")])
    assert call.usage.input_tokens == 100
    assert call.usage.output_tokens == 20


def test_an_unexpected_response_shape_is_reported() -> None:
    client = FakeBedrock()
    client.converse_responses = [{"何か違うもの": True}]
    with pytest.raises(ExternalCallError, match="応答の形が想定と違う"):
        _BedrockGenerator(config=generate_config(), caller=caller(client))(
            "質問", [hit("a", "本文")]
        )


# ---- JSON の取り出し ------------------------------------------------------
#
# Converse には OpenAI の response_format に当たるものが無い。
# 前置きや ``` で囲った形が混ざるので、そこだけを吸収する。


@pytest.mark.parametrize(
    "content",
    [
        '{"answer": "x"}',
        'はい、以下が回答です。\n{"answer": "x"}',
        '```json\n{"answer": "x"}\n```',
        '```\n{"answer": "x"}\n```',
        '{"answer": "x"}\n以上です。',
    ],
)
def test_the_json_is_extracted(content: str) -> None:
    assert json.loads(extract_json(content)) == {"answer": "x"}


def test_a_response_without_json_is_left_alone() -> None:
    """直しはしない。壊れたものを繕うと、生成の失敗が記録に残らなくなる。"""
    assert extract_json("すみません、わかりません") == "すみません、わかりません"


def test_a_broken_output_is_raised_not_swallowed() -> None:
    client = FakeBedrock()
    client.converse_responses = [converse_response("すみません、わかりません")]
    from rageval.generate import GenerationError

    with pytest.raises(GenerationError):
        _BedrockGenerator(config=generate_config(), caller=caller(client))(
            "質問", [hit("a", "本文")]
        )


# ---- 組み立て -------------------------------------------------------------


def test_build_embedder_needs_a_bedrock_caller() -> None:
    config = EmbedderConfig(provider="bedrock", model=TITAN)
    with pytest.raises(ValueError, match="BedrockCaller"):
        build_embedder(config)


def test_build_embedder_returns_the_bedrock_one() -> None:
    config = EmbedderConfig(provider="bedrock", model=TITAN)
    built = build_embedder(config, bedrock=caller(FakeBedrock()))
    assert isinstance(built, BedrockEmbedder)


def test_build_generator_needs_a_bedrock_caller() -> None:
    with pytest.raises(ValueError, match="BedrockCaller"):
        build_generator(generate_config())


def test_build_generator_returns_the_bedrock_one() -> None:
    built = build_generator(generate_config(), bedrock=caller(FakeBedrock()))
    assert isinstance(built, _BedrockGenerator)


def test_bedrock_needs_no_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """鍵を持たないのが、この経路を選んだ理由の1つ。鍵が無くても組み上がること。"""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    build_embedder(EmbedderConfig(provider="bedrock", model=TITAN), bedrock=caller(FakeBedrock()))
    build_generator(generate_config(), bedrock=caller(FakeBedrock()))


# ---- 呼び出し口の組み立て -------------------------------------------------


def test_the_client_is_built_without_sdk_retries() -> None:
    """SDK が黙って再試行すると、回数がこちらの記録に残らない。

    実験の費用と所要時間を後から追えるようにするため、リトライは
    `_RetryingCaller` に一本化する。
    """
    from rageval.external import build_bedrock_client

    client = build_bedrock_client(region="ap-northeast-1")
    config = client.meta.config
    # botocore の max_attempts は「リトライの回数」。1 を渡すと合計2回になる。
    # 実際にそれで1回ぶん黙って再試行されていた。指定するのは total_max_attempts。
    assert config.retries["total_max_attempts"] == 1, "SDK 側のリトライが残っている"
    assert config.connect_timeout > 0, "接続のタイムアウトが無い"
    assert config.read_timeout > 0, "読み取りのタイムアウトが無い"
    assert client.meta.region_name == "ap-northeast-1"


def test_the_generation_timeout_is_longer_than_the_default() -> None:
    """生成は応答が遅い。埋め込みと同じ待ちだと途中で切れる。"""
    from rageval.external import build_bedrock_client

    client = build_bedrock_client(region="ap-northeast-1")
    assert client.meta.config.read_timeout >= 120
