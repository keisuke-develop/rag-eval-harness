"""評価用コーパスを Wikipedia 日本語版から取得して datasets/corpus/ に保存する。

コーパスの選定条件は docs/02_evaluation.md の「コーパス」を参照。
再配布可能なライセンス（CC BY-SA 4.0）の公開文書のみを対象にしている。

取得したテキストには以下の加工を加える。加工内容を明示するのは、
評価セットの文字オフセットがこの加工後のテキストを基準にしているため。

1. 1行目に文書タイトルを置き、空行を1行はさむ
2. 出典・脚注・関連項目・外部リンクなど、本文でない節を落とす
3. 断片化した数式ブロックを落とす
4. 先頭から TARGET_CHARS 字を上限に、段落境界で切り出す
5. 改行を LF に統一し、3行以上の空行を2行に詰める

使い方:
    uv run python datasets/fetch_corpus.py          # 取得して保存
    uv run python datasets/fetch_corpus.py --check  # 保存済みと取得結果の差分だけ報告
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import ssl
import sys
import time
from pathlib import Path
from typing import Any

import httpx

API = "https://ja.wikipedia.org/w/api.php"
USER_AGENT = "rag-eval-harness/0.1 (https://github.com/keisuke-develop/rag-eval-harness)"

CORPUS_DIR = Path(__file__).parent / "corpus"
MANIFEST_PATH = CORPUS_DIR / "MANIFEST.json"
NOTICE_PATH = CORPUS_DIR / "LICENSE-NOTICE.md"

#: 1文書あたりの上限文字数。チャンク分割の挙動が観察できる長さを確保しつつ、
#: 評価セット作成時に人が全文を読み通せる範囲に収める。
TARGET_CHARS = 6000

#: 本文ではない節。見出しがこれらに一致する節は、下位の節ごと落とす。
DROP_SECTIONS = frozenset(
    {
        "脚注",
        "注釈",
        "出典",
        "参考文献",
        "関連項目",
        "外部リンク",
        "画像",
        "ギャラリー",
        "参考書籍",
        "参照",
        "関連文献",
    }
)

#: 取得する文書。分野が偏らないように選んでいる。
SOURCES: list[tuple[str, str, str]] = [
    ("doc01_fujisan", "富士山", "地理・自然"),
    ("doc02_photosynthesis", "光合成", "生物・化学"),
    ("doc03_hummingbird", "ハチドリ", "生物"),
    ("doc04_iss", "国際宇宙ステーション", "宇宙開発"),
    ("doc05_tcp", "Transmission Control Protocol", "情報技術"),
    ("doc06_solar_power", "太陽光発電", "エネルギー"),
    ("doc07_constitution", "日本国憲法", "法"),
    ("doc08_world_heritage", "世界遺産", "国際制度"),
    ("doc09_edo_period", "江戸時代", "歴史"),
    ("doc10_tokaido_shinkansen", "東海道新幹線", "交通"),
    ("doc11_earthquake", "地震", "自然科学"),
    ("doc12_washi", "和紙", "文化・工芸"),
]

_H2 = re.compile(r"^== ([^=].*?) ==$", re.MULTILINE)

#: 数式ブロックの残骸とみなす行の巻き込み上限。この長さ以下の行は数式の
#: ラベル（「一般式」「（炭水化物）」など）とみなして一緒に落とす。
MAX_LABEL_CHARS = 24


def _client() -> httpx.Client:
    """OS の証明書ストアを使う HTTP クライアント。

    TLS 傍受のある環境では certifi 同梱の CA では検証に失敗するため、
    truststore が入っていれば OS 側の証明書を使う。
    """
    verify: Any = True
    try:
        import truststore

        verify = truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    except ImportError:
        pass
    return httpx.Client(
        timeout=httpx.Timeout(30.0, connect=10.0),
        headers={"User-Agent": USER_AGENT},
        verify=verify,
    )


def fetch(client: httpx.Client, title: str) -> tuple[str, int, str]:
    """記事のプレーンテキストと版IDを取得する。戻り値は (正式タイトル, revid, 本文)。"""
    res = client.get(
        API,
        params={
            "action": "query",
            "prop": "extracts|revisions",
            "rvprop": "ids",
            "titles": title,
            "explaintext": "1",
            "format": "json",
            "formatversion": "2",
        },
    )
    res.raise_for_status()
    page = res.json()["query"]["pages"][0]
    if page.get("missing"):
        raise RuntimeError(f"記事が見つからない: {title}")
    return str(page["title"]), int(page["revisions"][0]["revid"]), str(page["extract"])


def drop_non_body_sections(text: str) -> str:
    """出典・関連項目などの節を、下位の節ごと落とす。"""
    marks = [(m.start(), m.group(1)) for m in _H2.finditer(text)]
    if not marks:
        return text
    kept = [text[: marks[0][0]]]
    for i, (start, heading) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        if heading.strip() not in DROP_SECTIONS:
            kept.append(text[start:stop])
    return "\n\n".join(part.strip() for part in kept if part.strip())


def _is_math_fragment(line: str) -> bool:
    """プレーンテキスト抽出で断片化した数式の行か。

    TextExtracts は数式を LaTeX の断片と字下げされた記号の並びとして吐く。
    本文の行は字下げされないため、行頭の空白を手がかりにできる。
    """
    return bool(line[:1].isspace()) or "{\\displaystyle" in line


def drop_math_artifacts(text: str) -> str:
    """断片化した数式ブロックを、その見出しラベルごと落とす。

    断片行を核として、前後にある空行と短い行（数式のラベル）を巻き込んだ範囲を
    まとめて削除する。節見出しは境界として扱い、巻き込まない。
    """
    lines = text.split("\n")
    fragment = [_is_math_fragment(line) for line in lines]
    drop = [False] * len(lines)

    def absorbable(line: str) -> bool:
        stripped = line.strip()
        return not stripped.startswith("=") and len(stripped) <= MAX_LABEL_CHARS

    i = 0
    while i < len(lines):
        if not fragment[i]:
            i += 1
            continue
        start, end = i, i
        while end < len(lines) and (fragment[end] or absorbable(lines[end])):
            end += 1
        while start > 0 and absorbable(lines[start - 1]):
            start -= 1
        for k in range(start, end):
            drop[k] = True
        i = end

    return "\n".join(line for line, dropped in zip(lines, drop, strict=True) if not dropped)


def truncate_at_paragraph(text: str, limit: int) -> str:
    """limit 字を超えない範囲で、直前の段落境界まで切り出す。"""
    if len(text) <= limit:
        return text
    cut = text.rfind("\n\n", 0, limit)
    if cut <= 0:
        cut = limit
    return text[:cut]


def clean(title: str, raw: str) -> str:
    text = raw.replace("\r\n", "\n").replace("\r", "\n")
    text = drop_non_body_sections(text)
    text = drop_math_artifacts(text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = f"{title}\n\n{text.strip()}"
    return truncate_at_paragraph(text, TARGET_CHARS).strip() + "\n"


def render_notice(entries: list[dict[str, Any]]) -> str:
    rows = "\n".join(
        "| `{doc_id}` | [{title}]({url}) | {revid} | {chars:,} | {domain} |".format(**e)
        for e in entries
    )
    return f"""<!-- このファイルは datasets/fetch_corpus.py が生成する。直接編集しない。 -->

# コーパスの出典とライセンス

このディレクトリの `.txt` は、**ウィキペディア日本語版**の記事から取得したものです。

- ライセンス: [クリエイティブ・コモンズ 表示-継承 4.0 国際 (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/deed.ja)
- 著作者表示: 各記事の執筆者（各記事の履歴ページを参照）
- 改変の有無: **あり**（下記「加工の内容」を参照）
- 継承: このディレクトリ配下の `.txt` は CC BY-SA 4.0 で再配布されます。
  リポジトリ本体の MIT ライセンス（[../../LICENSE](../../LICENSE)）とは別の扱いです。

## 収録文書

取得日: {entries[0]["fetched_at"]}

| doc_id | 記事 | 版 (oldid) | 文字数 | 分野 |
|---|---|---|---|---|
{rows}

各記事のその版は `https://ja.wikipedia.org/w/index.php?oldid=<版ID>` で参照できます。

## 加工の内容

原文そのままではありません。`datasets/fetch_corpus.py` が以下の加工を行っています。

1. 1行目に記事タイトルを置き、空行を1行はさむ
2. 脚注・出典・参考文献・関連項目・外部リンク・画像などの節を、下位の節ごと削除
3. 数式がプレーンテキスト化で断片化した領域を、ラベル行ごと削除
4. 先頭から {TARGET_CHARS:,} 字を上限に、段落境界で切り出し
5. 改行を LF に統一し、3行以上連続する空行を2行に圧縮
6. 表・画像は取得時点で脱落している（プレーンテキスト抽出のため）

**評価セット `datasets/qa_set.jsonl` の文字オフセットは、この加工後のテキストを基準にしています。**
加工の条件を変えるとオフセットがずれるため、変更する場合は評価セットのバージョンを上げてください。

## 再取得

```bash
uv run python datasets/fetch_corpus.py --check   # 保存済みとの差分を確認
uv run python datasets/fetch_corpus.py           # 取得して上書き
```

ウィキペディアの記事は更新されるため、再取得すると版IDと本文が変わります。
評価セットとの整合を保つため、通常は再取得せず、コミット済みの `.txt` をそのまま使ってください。
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="評価用コーパスを取得する")
    parser.add_argument("--check", action="store_true", help="保存せず差分だけ報告する")
    args = parser.parse_args()

    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    fetched_at = dt.date.today().isoformat()
    entries: list[dict[str, Any]] = []
    changed = False

    with _client() as client:
        for i, (doc_id, title, domain) in enumerate(SOURCES):
            if i:
                time.sleep(1.5)  # Wikipedia API のレート制限に触れないように間隔を空ける
            official, revid, raw = fetch(client, title)
            body = clean(official, raw)
            path = CORPUS_DIR / f"{doc_id}.txt"
            previous = path.read_text(encoding="utf-8") if path.exists() else None

            if previous is None:
                changed = True
                print(f"  NEW    {doc_id}  ({len(body):,} 字)")
            elif previous != body:
                changed = True
                print(f"  DIFF   {doc_id}  ({len(previous):,} -> {len(body):,} 字)")
            else:
                print(f"  SAME   {doc_id}  ({len(body):,} 字)")

            if not args.check:
                path.write_text(body, encoding="utf-8", newline="\n")

            entries.append(
                {
                    "doc_id": doc_id,
                    "title": official,
                    "domain": domain,
                    "url": f"https://ja.wikipedia.org/wiki/{official.replace(' ', '_')}",
                    "revid": revid,
                    "chars": len(body),
                    "license": "CC BY-SA 4.0",
                    "fetched_at": fetched_at,
                }
            )

    if args.check:
        print("\n差分あり。再取得が必要。" if changed else "\n差分なし。")
        return 1 if changed else 0

    MANIFEST_PATH.write_text(
        json.dumps(
            {"source": "ja.wikipedia.org", "target_chars": TARGET_CHARS, "documents": entries},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    NOTICE_PATH.write_text(render_notice(entries), encoding="utf-8", newline="\n")
    total = sum(int(e["chars"]) for e in entries)
    print(f"\n{len(entries)} 文書 / 総 {total:,} 字を {CORPUS_DIR} に保存した。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
