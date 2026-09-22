"""埋め込み。決定性と、外部APIを叩く側の振る舞い。

外部APIは respx でモックする。CI で実際のAPIを叩かない（CLAUDE.md）。
"""

from __future__ import annotations

import math
import os
import subprocess
import sys
from typing import Any

import httpx
import pytest
import respx

from rageval.config import EmbedderConfig
from rageval.embed import (
    HashingEmbedder,
    OpenAIEmbedder,
    build_embedder,
    character_ngrams,
    normalize_text,
)
from rageval.external import ApiCaller, ExternalCallError

EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"


def caller() -> ApiCaller:
    return ApiCaller(client=httpx.Client(), sleep=lambda _: None)


# ---- HashingEmbedder ----------------------------------------------------


def test_same_text_gives_the_same_vector() -> None:
    embedder = HashingEmbedder(dim=64)
    assert embedder.embed_one("富士山の標高") == embedder.embed_one("富士山の標高")


def test_vectors_are_l2_normalised() -> None:
    vector = HashingEmbedder(dim=64).embed_one("光合成とは何か")
    assert math.isclose(math.sqrt(sum(x * x for x in vector)), 1.0, rel_tol=1e-6)


def test_empty_text_gives_a_zero_vector() -> None:
    assert HashingEmbedder(dim=16).embed_one("") == [0.0] * 16


def test_overlapping_text_scores_higher_than_unrelated_text() -> None:
    """乱数ではなく、文字の重なりを実際に拾っていること。"""
    embedder = HashingEmbedder(dim=2048)

    def cosine(a: str, b: str) -> float:
        return sum(x * y for x, y in zip(embedder.embed_one(a), embedder.embed_one(b), strict=True))

    related = cosine("富士山の標高は3776メートル", "富士山の最高地点の標高")
    unrelated = cosine("富士山の標高は3776メートル", "TCPのヘッダ長は可変である")
    assert related > unrelated
    assert related > 0.1, "重なりをまったく拾えていないと検索として機能しない"


def test_vectors_are_stable_across_processes() -> None:
    """Python の hash() は起動ごとに salt が変わる。それを使っていないことの確認。

    ここが壊れると、実行のたびに検索結果が変わって再現性が失われる。
    """
    code = (
        "from rageval.embed import HashingEmbedder;"
        "print(sum(HashingEmbedder(dim=64).embed_one('再現性の確認')))"
    )
    # 環境変数は丸ごと捨てずに、PYTHONHASHSEED だけ差し替える。
    # 捨ててしまうと、venv の解決や共有ライブラリの探索に必要な変数まで消えて、
    # 「ハッシュの種が結果に影響するか」ではなく「子プロセスが起動できるか」を
    # 測ることになる（OS によって結果が変わる）。
    outputs = {
        subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        ).stdout.strip()
        for seed in ("0", "1", "12345")
    }
    assert len(outputs) == 1, f"PYTHONHASHSEED で結果が変わった: {outputs}"


def test_normalise_folds_width_and_case() -> None:
    assert normalize_text("ＴＣＰ　１２３") == normalize_text("tcp 123")


def test_normalise_collapses_whitespace() -> None:
    assert normalize_text("a \n\t b ") == "a b"


def test_character_ngrams() -> None:
    assert list(character_ngrams("あいう", (2,))) == ["あい", "いう"]
    assert list(character_ngrams("あい", (3,))) == []


def test_build_embedder_returns_the_hashing_implementation() -> None:
    config = EmbedderConfig(provider="hashing", model="char-ngram", dim=32)
    embedder = build_embedder(config)
    assert isinstance(embedder, HashingEmbedder)
    assert embedder.dim == 32


# ---- OpenAIEmbedder -----------------------------------------------------


def _embedding_response(request: httpx.Request) -> httpx.Response:
    import json

    payload: dict[str, Any] = json.loads(request.content)
    inputs: list[str] = payload["input"]
    # index を故意に降順で返す。呼び出し側が並べ替えていないと入力と対応がずれる。
    data = [
        {"index": i, "embedding": [float(len(text)), 0.0]}
        for i, text in reversed(list(enumerate(inputs)))
    ]
    return httpx.Response(200, json={"data": data, "usage": {"prompt_tokens": 7}})


@respx.mock
def test_openai_embedder_restores_the_input_order() -> None:
    respx.post(EMBEDDINGS_URL).mock(side_effect=_embedding_response)
    embedder = OpenAIEmbedder(model="text-embedding-3-small", caller=caller())
    vectors = embedder.embed(["a", "bb", "ccc"])
    assert [v[0] for v in vectors] == [1.0, 2.0, 3.0]


@respx.mock
def test_openai_embedder_splits_into_batches() -> None:
    route = respx.post(EMBEDDINGS_URL).mock(side_effect=_embedding_response)
    embedder = OpenAIEmbedder(model="m", caller=caller(), batch_size=2)
    embedder.embed(["a", "b", "c", "d", "e"])
    assert route.call_count == 3


@respx.mock
def test_openai_embedder_records_token_usage() -> None:
    respx.post(EMBEDDINGS_URL).mock(side_effect=_embedding_response)
    api = caller()
    OpenAIEmbedder(model="m", caller=api, batch_size=2).embed(["a", "b", "c"])
    assert api.usage.requests == 2
    assert api.usage.input_tokens == 14


@respx.mock
def test_openai_embedder_rejects_a_mismatched_response() -> None:
    respx.post(EMBEDDINGS_URL).mock(
        return_value=httpx.Response(200, json={"data": [{"index": 0, "embedding": [1.0]}]})
    )
    with pytest.raises(ExternalCallError, match="対応していない"):
        OpenAIEmbedder(model="m", caller=caller()).embed(["a", "b"])


def test_build_embedder_needs_a_caller_for_openai() -> None:
    config = EmbedderConfig(provider="openai", model="text-embedding-3-small")
    with pytest.raises(ValueError, match="ApiCaller"):
        build_embedder(config)


def test_build_embedder_explains_a_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = EmbedderConfig(provider="openai", model="text-embedding-3-small")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_embedder(config, caller=caller())


def test_build_embedder_returns_the_openai_implementation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = EmbedderConfig(provider="openai", model="text-embedding-3-small")
    assert isinstance(build_embedder(config, caller=caller()), OpenAIEmbedder)
