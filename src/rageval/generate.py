"""文脈つきプロンプトの組み立てと生成。出力の構造化と検証。

生成の評価は LLM に採点させず、構造化出力で返させた「根拠として使ったチャンクID」を
機械的に照合する（ADR 0003）。そのためここでの責務は、`Answer` の形で
必ず返させ、壊れていたら弾くことにある。

**生成プロバイダには Protocol を切らない。** ADR 0002 で「あえて固定する」と
決めた箇所なので、差し替え点の見た目を与えず、`provider` による明示的な分岐にしてある。
差し替えたくなったら、ここに分岐を1つ足すという判断が要る。

実装は2つ。

`provider: quote`
    外部APIに出ない決定的な生成。最上位ヒットのチャンクをそのまま引用し、
    そのチャンクIDを根拠として返す。**これが測っているのは検索の1位が
    当たっているかどうかであって、文章生成の良し悪しではない。**
    結果を読むときはこの区別を崩さないこと。

    **この実装は棄権を判定しない。** 当初はスコアの閾値で棄権させる設計にしたが、
    評価セットで測ったところ、答えられない質問の最上位スコア（中央値 0.197）が
    答えられる質問（同 0.156）より高く、閾値では分離できなかった。
    語彙の一致率に替えても重なりは解消しなかった。判定できないものを
    それらしい閾値で判定した風にするより、判定しないことを明示する。
    したがって `provider: quote` では棄権率は常に 0 になる。
    この指標が生きるのは実際の生成モデルを使うときだけ。

`provider: openai`
    実APIを叩く。JSON で返させ、`Answer` で検証する。

`provider: bedrock`
    Amazon Bedrock の Converse API を叩く。**APIキーを持たない**のが選んだ理由の1つ。
    Converse は提供元ごとの本文の形の違いを吸収してくれるので、
    Claude でも Nova でも同じ呼び方になる。
    ただし OpenAI の `response_format` に当たるものが無いため、
    **JSON だけを返させる強制力が弱い。** 前後に説明文が付くことがあるので、
    `parse_answer` に入れる前に JSON の部分を取り出す。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from rageval.config import GenerateConfig
from rageval.external import ApiCaller, BedrockCaller, ExternalCallError, read_api_key
from rageval.store import SearchHit

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"

#: プロンプトは固定する。暗黙に変わると条件の切り分けができなくなる（docs/02_evaluation.md）。
SYSTEM_PROMPT = """あなたは与えられた文脈だけを根拠に日本語で答える調査アシスタントです。
次の規則を必ず守ってください。

1. 文脈に書かれていることだけを使う。推測や一般知識で補わない。
2. 答えの根拠にしたチャンクのIDを cited_chunk_ids にすべて挙げる。使っていないIDは挙げない。
3. 文脈に答えが無い場合は abstained を true にし、cited_chunk_ids は空にする。
4. 出力は次の形の JSON のみとする。前置きも説明も付けない。

{"answer": "回答の本文", "cited_chunk_ids": ["doc01_fujisan#c003"], "abstained": false}"""

USER_PROMPT_TEMPLATE = """# 文脈

{context}

# 質問

{question}"""


class GenerationError(RuntimeError):
    """生成の出力が壊れていて `Answer` にできなかった。失敗として記録する。"""


class Answer(BaseModel):
    """生成の構造化出力（ADR 0003）。"""

    model_config = ConfigDict(extra="ignore")

    answer: str = ""
    cited_chunk_ids: list[str] = Field(default_factory=list)
    abstained: bool = False

    @model_validator(mode="after")
    def _abstention_has_no_citation(self) -> Answer:
        if self.abstained and self.cited_chunk_ids:
            raise ValueError("棄権したのに根拠が挙がっている。どちらかが誤り")
        return self


def format_context(hits: list[SearchHit]) -> str:
    """検索結果をプロンプトに載る形にする。IDを明示しないと根拠を挙げさせられない。"""
    blocks = []
    for hit in hits:
        body = " ".join(hit.chunk.text.split())
        blocks.append(f"[{hit.chunk.chunk_id}]\n{body}")
    return "\n\n".join(blocks)


@dataclass(frozen=True)
class _QuoteGenerator:
    """最上位ヒットを引用して返すだけの決定的な生成。"""

    config: GenerateConfig

    def __call__(self, question: str, hits: list[SearchHit]) -> Answer:
        # 文脈がまったく無いときだけ棄権する。スコアの閾値では答えられない質問を
        # 見分けられないことを測って確かめてあるので、閾値は持たない（モジュールの説明を参照）。
        if not hits:
            return Answer(
                answer="文脈に該当する記述が見つかりませんでした。",
                cited_chunk_ids=[],
                abstained=True,
            )
        top = hits[0]
        body = " ".join(top.chunk.text.split())
        return Answer(
            answer=body[: self.config.max_answer_chars],
            cited_chunk_ids=[top.chunk.chunk_id],
            abstained=False,
        )


@dataclass(frozen=True)
class _OpenAIGenerator:
    config: GenerateConfig
    caller: ApiCaller
    url: str = OPENAI_CHAT_URL

    def __call__(self, question: str, hits: list[SearchHit]) -> Answer:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "seed": self.config.seed,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": USER_PROMPT_TEMPLATE.format(
                        context=format_context(hits), question=question
                    ),
                },
            ],
        }
        body = self.caller.post_json(self.url, payload)
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ExternalCallError(f"生成の応答の形が想定と違う: {body}") from exc
        return parse_answer(content)


def parse_answer(content: str) -> Answer:
    """生成が返した文字列を `Answer` にする。壊れていれば GenerationError。"""
    try:
        raw = json.loads(content)
    except json.JSONDecodeError as exc:
        raise GenerationError(f"JSON として読めない: {content[:200]}") from exc
    try:
        return Answer.model_validate(raw)
    except Exception as exc:  # pydantic.ValidationError
        raise GenerationError(f"Answer の形になっていない: {content[:200]}") from exc


def extract_json(content: str) -> str:
    """応答から JSON の部分を取り出す。

    OpenAI には `response_format` があり JSON だけが返るが、**Converse には無い。**
    「はい、以下が回答です」のような前置きや、```json で囲った形が混ざる。

    ここでやるのは**取り出しだけで、直しはしない。** 壊れた JSON を推測で
    繕うと、生成が失敗したことが記録に残らなくなる（ADR 0003 の前提が崩れる）。
    最初の `{` から最後の `}` までを切り出し、あとは `parse_answer` に任せる。
    """
    fenced = content.strip()
    if fenced.startswith("```"):
        # ```json ... ``` の中身だけにする
        fenced = fenced.split("```")[1] if fenced.count("```") >= 2 else fenced
        fenced = fenced.removeprefix("json").strip()
    start = fenced.find("{")
    end = fenced.rfind("}")
    if start == -1 or end == -1 or end < start:
        return content
    return fenced[start : end + 1]


@dataclass
class _BedrockGenerator:
    """Amazon Bedrock の Converse API。呼び出しは必ず `BedrockCaller` を通す。"""

    config: GenerateConfig
    caller: BedrockCaller

    def __call__(self, question: str, hits: list[SearchHit]) -> Answer:
        body = self.caller.converse(
            self.config.model,
            system=SYSTEM_PROMPT,
            user=USER_PROMPT_TEMPLATE.format(context=format_context(hits), question=question),
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        try:
            content = body["output"]["message"]["content"][0]["text"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ExternalCallError(f"生成の応答の形が想定と違う: {body}") from exc
        return parse_answer(extract_json(content))


def build_generator(
    config: GenerateConfig,
    *,
    caller: ApiCaller | None = None,
    bedrock: BedrockCaller | None = None,
) -> _QuoteGenerator | _OpenAIGenerator | _BedrockGenerator:
    """条件から生成器を組み立てる。

    戻り値に共通の Protocol を与えていないのは意図的で、ADR 0002 の
    「生成プロバイダは差し替え点にしない」に従っている。
    **分岐を足すのは意識的な判断として行うこと**、という決まりなので、
    増やした経緯は ADR 0002 に追記してある。
    """
    if config.provider == "quote":
        return _QuoteGenerator(config=config)

    if config.provider == "bedrock":
        if bedrock is None:
            raise ValueError("provider: bedrock には BedrockCaller が要る")
        return _BedrockGenerator(config=config, caller=bedrock)

    if caller is None:
        raise ValueError("provider: openai には ApiCaller が要る")
    read_api_key("OPENAI_API_KEY", fallback_provider="quote")
    return _OpenAIGenerator(config=config, caller=caller)
