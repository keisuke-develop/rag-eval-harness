"""チャンクとクエリのベクトル化。

差し替え可能な4点のうち「埋め込みプロバイダ」がここにある（ADR 0002）。
実験005でモデル差を測るための軸でもある。

実装は2つ。

`HashingEmbedder`（provider: hashing）
    外部APIに出ない決定的な埋め込み。文字 n-gram をハッシュして符号つきで
    足し込み、L2 正規化する。**乱数ではない**——乱数だと Recall が0に張り付き、
    CI の回帰検証が「落ちないだけ」の飾りになるため。実際に文字の重なりを
    拾う弱い検索器として動く。鍵が無くてもパイプライン全体を回せる。

`OpenAIEmbedder`（provider: openai）
    実APIを叩く。タイムアウト・リトライ・レート制限・トークン数記録は
    `external.ApiCaller` を必ず通す（docs/01_architecture.md）。

`BedrockEmbedder`（provider: bedrock）
    Amazon Bedrock を叩く。**APIキーを持たない**のが選んだ理由の1つで、
    認証は手元なら AWS の認証情報、AWS 上ならタスクロールで行う。
    Titan と Cohere で本文の形が違うので、そこだけを吸収する。
"""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from rageval.config import EmbedderConfig
from rageval.external import (
    ApiCaller,
    BedrockCaller,
    ExternalCallError,
    read_api_key,
)

Vector = list[float]

OPENAI_EMBEDDINGS_URL = "https://api.openai.com/v1/embeddings"

_WHITESPACE = re.compile(r"\s+")


class Embedder(Protocol):
    """差し替え点：埋め込みプロバイダ。"""

    def embed(self, texts: list[str]) -> list[Vector]: ...


def normalize_text(text: str) -> str:
    """表記ゆれを均す。全角半角（ＴＣＰ→tcp）と大小文字、連続する空白をまとめる。"""
    folded = unicodedata.normalize("NFKC", text).casefold()
    return _WHITESPACE.sub(" ", folded).strip()


def character_ngrams(text: str, sizes: tuple[int, ...]) -> Iterator[str]:
    """文字 n-gram を順に返す。日本語は語の切れ目が無いので、単語ではなく文字で見る。"""
    for size in sizes:
        for i in range(len(text) - size + 1):
            yield text[i : i + size]


@dataclass(frozen=True)
class HashingEmbedder:
    """文字 n-gram のハッシュ化ベクトル。外部に出ず、プロセスをまたいで同じ結果になる。

    Python 組み込みの `hash()` は起動ごとに salt が変わるため使えない。
    blake2b に固定の `person` を与えて、実行のたびに同じ値が出るようにしている。
    """

    dim: int = 512
    sizes: tuple[int, ...] = (2, 3)
    seed: str = "rageval"

    @classmethod
    def from_config(cls, config: EmbedderConfig) -> HashingEmbedder:
        return cls(dim=config.dim)

    def _bucket(self, ngram: str) -> tuple[int, float]:
        digest = hashlib.blake2b(
            ngram.encode("utf-8"), digest_size=8, person=self.seed.encode("utf-8")
        ).digest()
        value = int.from_bytes(digest, "big")
        # 最下位ビットで符号を決める。異なる n-gram が同じバケットに落ちたとき、
        # 符号が揃っていると無関係な語どうしが似て見えるため。
        sign = 1.0 if value & 1 else -1.0
        return (value >> 1) % self.dim, sign

    def embed_one(self, text: str) -> Vector:
        vector = np.zeros(self.dim, dtype=np.float64)
        counts: dict[str, int] = {}
        for ngram in character_ngrams(normalize_text(text), self.sizes):
            counts[ngram] = counts.get(ngram, 0) + 1
        for ngram, count in counts.items():
            index, sign = self._bucket(ngram)
            # 出現回数をそのまま足すと、長いチャンクの頻出語だけが支配的になる。
            # 対数で潰して、語の「種類」が効くようにする。
            vector[index] += sign * (1.0 + math.log(count))
        norm = float(np.linalg.norm(vector))
        if norm > 0.0:
            vector /= norm
        return [float(x) for x in vector]

    def embed(self, texts: list[str]) -> list[Vector]:
        return [self.embed_one(text) for text in texts]


@dataclass
class OpenAIEmbedder:
    """OpenAI の埋め込みAPI。呼び出しは必ず `ApiCaller` を通す。"""

    model: str
    caller: ApiCaller
    batch_size: int = 64
    url: str = OPENAI_EMBEDDINGS_URL

    def embed(self, texts: list[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            body = self.caller.post_json(self.url, {"model": self.model, "input": batch})
            data = body.get("data")
            if not isinstance(data, list) or len(data) != len(batch):
                raise ExternalCallError(
                    f"埋め込みの応答が入力と対応していない: {len(batch)} 件送って "
                    f"{len(data) if isinstance(data, list) else '不明'} 件返ってきた"
                )
            # data は index 順に並ぶ保証がないので、明示的に並べ替える。
            ordered = sorted(data, key=lambda item: int(item["index"]))
            vectors.extend([float(x) for x in item["embedding"]] for item in ordered)
        return vectors


@dataclass
class BedrockEmbedder:
    """Amazon Bedrock の埋め込み。呼び出しは必ず `BedrockCaller` を通す。

    Titan と Cohere で本文の形が違う。**違うのはそこだけ**なので、
    送る形と受け取る形の対応だけを持ち、他は共通にしてある。

    | | 送る | 受け取る | まとめて送れるか |
    |---|---|---|---|
    | Titan | `inputText`（1件） | `embedding` | いいえ |
    | Cohere | `texts`（複数） | `embeddings` | はい |

    Cohere は検索用途で `input_type` を要求する。文書とクエリで別の値を使うのが
    本来だが、**この道具はチャンクもクエリも同じ経路で埋め込む**ので、
    片方に決め打つと比較の条件が揃わない。`search_document` に固定してある。
    """

    model: str
    caller: BedrockCaller
    batch_size: int = 64

    def embed(self, texts: list[str]) -> list[Vector]:
        if self.model.startswith("cohere."):
            return self._embed_cohere(texts)
        return self._embed_titan(texts)

    def _embed_titan(self, texts: list[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for text in texts:
            body = self.caller.invoke_model(self.model, {"inputText": text})
            raw = body.get("embedding")
            if not isinstance(raw, list):
                raise ExternalCallError(f"埋め込みが返ってこなかった: {self.model}")
            vectors.append([float(x) for x in raw])
        return vectors

    def _embed_cohere(self, texts: list[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            body = self.caller.invoke_model(
                self.model, {"texts": batch, "input_type": "search_document"}
            )
            raw = body.get("embeddings")
            if not isinstance(raw, list) or len(raw) != len(batch):
                raise ExternalCallError(
                    f"埋め込みの応答が入力と対応していない: {len(batch)} 件送って "
                    f"{len(raw) if isinstance(raw, list) else '不明'} 件返ってきた"
                )
            vectors.extend([float(x) for x in item] for item in raw)
        return vectors


def build_embedder(
    config: EmbedderConfig,
    *,
    caller: ApiCaller | None = None,
    bedrock: BedrockCaller | None = None,
) -> Embedder:
    """条件から埋め込み器を組み立てる。"""
    if config.provider == "hashing":
        return HashingEmbedder.from_config(config)

    if config.provider == "bedrock":
        if bedrock is None:
            raise ValueError("provider: bedrock には BedrockCaller が要る")
        return BedrockEmbedder(model=config.model, caller=bedrock, batch_size=config.batch_size)

    if caller is None:
        raise ValueError("provider: openai には ApiCaller が要る")
    read_api_key("OPENAI_API_KEY", fallback_provider="hashing")
    return OpenAIEmbedder(model=config.model, caller=caller, batch_size=config.batch_size)
