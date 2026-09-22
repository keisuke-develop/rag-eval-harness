"""生成。構造化出力の検証と、壊れた出力の扱い。

LLM に採点させない代わりに、生成が返す根拠のIDを機械的に照合する（ADR 0003）。
その前提が成り立つのは、出力が必ず `Answer` の形をしているときだけなので、
形が崩れたときに黙って通さないことを確かめる。
"""

from __future__ import annotations

import httpx
import pytest
import respx
from pydantic import ValidationError

from rageval.config import GenerateConfig
from rageval.external import ApiCaller, ExternalCallError
from rageval.generate import (
    Answer,
    GenerationError,
    _OpenAIGenerator,
    _QuoteGenerator,
    build_generator,
    format_context,
    parse_answer,
)
from rageval.ingest import Chunk
from rageval.store import SearchHit

CHAT_URL = "https://api.openai.com/v1/chat/completions"


def hit(chunk_id: str, text: str, score: float = 0.9, rank: int = 1) -> SearchHit:
    return SearchHit(
        chunk=Chunk(chunk_id=chunk_id, doc_id="d", index=0, start=0, end=len(text), text=text),
        score=score,
        rank=rank,
    )


def quote_config(**kwargs: object) -> GenerateConfig:
    return GenerateConfig.model_validate({"provider": "quote", "model": "q", **kwargs})


# ---- Answer -------------------------------------------------------------


def test_abstaining_with_a_citation_is_rejected() -> None:
    with pytest.raises(ValidationError, match="棄権したのに根拠"):
        Answer(answer="", cited_chunk_ids=["a#c000"], abstained=True)


def test_an_answer_may_cite_nothing_without_abstaining() -> None:
    """根拠を挙げずに答えるのは壊れた出力ではない。評価側で不一致として数える。"""
    assert Answer(answer="なにか", cited_chunk_ids=[]).abstained is False


def test_parse_answer_reads_the_documented_shape() -> None:
    answer = parse_answer(
        '{"answer": "3776メートル", "cited_chunk_ids": ["doc01#c004"], "abstained": false}'
    )
    assert answer.cited_chunk_ids == ["doc01#c004"]


def test_parse_answer_rejects_non_json() -> None:
    with pytest.raises(GenerationError, match="JSON として読めない"):
        parse_answer("はい、答えは3776メートルです。")


def test_parse_answer_rejects_a_contradictory_answer() -> None:
    with pytest.raises(GenerationError, match="Answer の形"):
        parse_answer('{"answer": "x", "cited_chunk_ids": ["a"], "abstained": true}')


def test_parse_answer_ignores_extra_fields() -> None:
    assert parse_answer('{"answer": "x", "confidence": 0.9}').answer == "x"


# ---- provider: quote ----------------------------------------------------


def test_quote_generator_cites_the_top_hit() -> None:
    answer = _QuoteGenerator(quote_config())("質問", [hit("a#c001", "富士山の標高は3776メートル")])
    assert answer.cited_chunk_ids == ["a#c001"]
    assert answer.abstained is False
    assert "3776" in answer.answer


def test_quote_generator_truncates_the_answer() -> None:
    answer = _QuoteGenerator(quote_config(max_answer_chars=5))("質問", [hit("a", "あ" * 100)])
    assert len(answer.answer) == 5


def test_quote_generator_collapses_newlines() -> None:
    answer = _QuoteGenerator(quote_config())("質問", [hit("a", "一行目\n\n二行目")])
    assert "\n" not in answer.answer


def test_quote_generator_abstains_only_when_there_is_no_context() -> None:
    """スコアの閾値では棄権を判定しないと決めた。文脈が空のときだけ棄権する。"""
    assert _QuoteGenerator(quote_config())("質問", []).abstained is True
    low = _QuoteGenerator(quote_config())("質問", [hit("a", "本文", score=0.001)])
    assert low.abstained is False, "スコアが低くても棄権しない"


# ---- provider: openai ---------------------------------------------------


def caller() -> ApiCaller:
    return ApiCaller(client=httpx.Client(), sleep=lambda _: None)


def chat_response(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30},
        },
    )


@respx.mock
def test_openai_generator_parses_the_structured_output() -> None:
    respx.post(CHAT_URL).mock(
        return_value=chat_response('{"answer": "3776m", "cited_chunk_ids": ["a#c001"]}')
    )
    config = GenerateConfig(provider="openai", model="gpt-4o-mini")
    answer = _OpenAIGenerator(config=config, caller=caller())("標高は？", [hit("a#c001", "本文")])
    assert answer.cited_chunk_ids == ["a#c001"]


@respx.mock
def test_openai_generator_sends_the_fixed_settings() -> None:
    route = respx.post(CHAT_URL).mock(return_value=chat_response('{"answer": "x"}'))
    config = GenerateConfig(provider="openai", model="gpt-4o-mini", temperature=0.0, seed=42)
    _OpenAIGenerator(config=config, caller=caller())("質問", [hit("a", "本文")])
    import json

    payload = json.loads(route.calls[0].request.content)
    assert payload["temperature"] == 0.0
    assert payload["seed"] == 42
    assert payload["response_format"] == {"type": "json_object"}


@respx.mock
def test_a_broken_structured_output_is_raised_not_swallowed() -> None:
    respx.post(CHAT_URL).mock(return_value=chat_response("すみません、わかりません"))
    config = GenerateConfig(provider="openai", model="gpt-4o-mini")
    with pytest.raises(GenerationError):
        _OpenAIGenerator(config=config, caller=caller())("質問", [hit("a", "本文")])


def test_format_context_labels_every_chunk_with_its_id() -> None:
    context = format_context([hit("a#c000", "一つ目"), hit("b#c001", "二つ目")])
    assert "[a#c000]" in context
    assert "[b#c001]" in context


def test_build_generator_switches_on_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(build_generator(quote_config()), _QuoteGenerator)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = GenerateConfig(provider="openai", model="gpt-4o-mini")
    assert isinstance(build_generator(config, caller=caller()), _OpenAIGenerator)


def test_build_generator_explains_a_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = GenerateConfig(provider="openai", model="gpt-4o-mini")
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        build_generator(config, caller=caller())


@respx.mock
def test_an_unexpected_response_shape_is_reported() -> None:
    """応答の形が想定と違うのは、生成の失敗ではなく呼び出しの失敗として扱う。"""
    respx.post(CHAT_URL).mock(return_value=httpx.Response(200, json={"unexpected": True}))
    config = GenerateConfig(provider="openai", model="gpt-4o-mini")
    with pytest.raises(ExternalCallError, match="応答の形が想定と違う"):
        _OpenAIGenerator(config=config, caller=caller())("質問", [hit("a", "本文")])


def test_build_generator_needs_a_caller_for_openai() -> None:
    config = GenerateConfig(provider="openai", model="gpt-4o-mini")
    with pytest.raises(ValueError, match="ApiCaller"):
        build_generator(config)
