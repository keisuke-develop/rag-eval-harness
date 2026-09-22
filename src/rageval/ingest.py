"""コーパスの読み込みとチャンク分割。

差し替え可能な4点のうち「チャンク戦略」がここにある（ADR 0002）。

チャンクは**元の文書の文字オフセット**を保持する。評価セットの正解根拠も
文字オフセットで持っているため、両者を突き合わせれば、チャンクサイズを
変えても同じ評価セットで比較できる（docs/02_evaluation.md）。
オフセットを落とすと、その瞬間にチャンクサイズの比較実験が成立しなくなる。
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from rageval.config import ChunkConfig


@dataclass(frozen=True)
class Document:
    """コーパスの1文書。"""

    doc_id: str
    text: str


@dataclass(frozen=True)
class Chunk:
    """検索単位。`start`/`end` は元文書に対する文字オフセット（`end` は含まない）。"""

    chunk_id: str
    doc_id: str
    index: int
    start: int
    end: int
    text: str

    def covers(self, doc_id: str, start: int, end: int) -> bool:
        """指定の範囲と少しでも重なるか。正解根拠を含むチャンクの判定に使う。"""
        if self.doc_id != doc_id:
            return False
        return self.start < end and start < self.end


class Chunker(Protocol):
    """差し替え点：チャンク戦略。"""

    def split(self, doc: Document) -> list[Chunk]: ...


@dataclass(frozen=True)
class FixedSizeChunker:
    """固定長で切り、隣接チャンクを `overlap` 文字だけ重ねる。

    文の境界を見ずに文字数だけで切る。境界で意味が切れる問題を重複幅で
    どこまで救えるかは、実験002で測る対象そのものなので、ここでは賢くしない。
    """

    size: int
    overlap: int
    #: 境界をずらす文字数。最初のチャンクだけ短くして、以降の境界を全体にずらす。
    #: 既定の 0 では、これを入れる前とまったく同じ切り方になる。
    phase: int = 0

    @classmethod
    def from_config(cls, config: ChunkConfig) -> FixedSizeChunker:
        return cls(size=config.size, overlap=config.overlap, phase=config.phase)

    def split(self, doc: Document) -> list[Chunk]:
        if self.size - self.phase <= self.overlap:  # pragma: no cover - ChunkConfig が先に弾く
            raise ValueError("size - phase は overlap より大きいこと")

        chunks: list[Chunk] = []
        text_length = len(doc.text)
        start = 0
        # 最初のチャンクだけ phase のぶん短くする。これで以降の境界がすべてずれる。
        end = min(self.size - self.phase, text_length)
        while start < text_length:
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}#c{len(chunks):03d}",
                    doc_id=doc.doc_id,
                    index=len(chunks),
                    start=start,
                    end=end,
                    text=doc.text[start:end],
                )
            )
            if end == text_length:
                break
            start = end - self.overlap
            end = min(start + self.size, text_length)
        return chunks


def load_corpus(directory: Path) -> list[Document]:
    """コーパスのディレクトリから `.txt` を読む。

    doc_id はファイル名の拡張子を除いた部分。並び順はファイル名順に固定する。
    順序が環境で変わると、同じ条件でもチャンクIDがずれて再現性が壊れるため。
    """
    paths = sorted(directory.glob("*.txt"))
    if not paths:
        raise ValueError(f"コーパスに .txt がない: {directory}")

    documents = []
    for path in paths:
        text = path.read_text(encoding="utf-8")
        if text.startswith("﻿"):
            # BOM は文字として残り、その文書のオフセットが全部1文字ずれる。
            # 評価セットを作った時点と読む時点で BOM の有無が変われば、
            # 正解根拠が静かに1文字ずれた場所を指すことになる。黙って剥がさず止める。
            raise ValueError(
                f"{path.name} に BOM が付いている。"
                "BOM は1文字として数えられ、この文書の正解根拠のオフセットが全部ずれる。"
                "BOM 無しの UTF-8 で保存し直すこと"
            )
        if not text.strip():
            # 空の文書はチャンクを1つも生まないので、黙って無視すると
            # 「入れたはずの文書が検索に出ない」ことに気づけない。
            raise ValueError(f"{path.name} が空。コーパスに入れる文書には本文が要る")
        documents.append(Document(doc_id=path.stem, text=text))
    return documents


def chunk_documents(documents: Iterable[Document], chunker: Chunker) -> list[Chunk]:
    """全文書をチャンクに割る。"""
    return [chunk for document in documents for chunk in chunker.split(document)]
