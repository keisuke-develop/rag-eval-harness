# datasets

評価セットとコーパス。**パイプラインより先にここを作った**（[ADR 0001](../docs/adr/0001-evaluation-set-first.md)）。

## ファイルの役割

| ファイル | 人が編集するか | 内容 |
|---|---|---|
| `qa_draft.yaml` | **する** | 評価セットの原稿。質問と、根拠として引用する本文 |
| `build_qa_set.py` | する | 原稿を検査して `qa_set.jsonl` を生成する |
| `qa_set.jsonl` | しない（生成物） | 評価セット本体。1行1質問 |
| `qa_set.meta.json` | しない（生成物） | 版と件数の内訳 |
| `fetch_corpus.py` | する | コーパスを取得して加工する |
| `corpus/*.txt` | しない（取得物） | 評価用コーパス。12文書 / 66,496字 |
| `corpus/MANIFEST.json` | しない（生成物） | 出典・版ID・文字数 |
| `corpus/LICENSE-NOTICE.md` | しない（生成物） | 出典とライセンスの表示 |

## qa_set.jsonl

```json
{"qid": "q001", "question": "...", "gold_spans": [{"doc": "doc01_fujisan", "start": 2481, "end": 2529}], "answerable": true, "factoid": true}
{"qid": "q049", "question": "...", "gold_spans": [], "answerable": false, "factoid": true}
```

正解根拠を**チャンクIDではなく文字オフセットで持つ**のが要点。
チャンク分割の条件を変えるとチャンクIDが変わるため、IDで持つとチャンクサイズの比較実験が成立しなくなる。
実行時は、この範囲を含むチャンクを正解とみなす。

`start` / `end` は加工後の `corpus/<doc>.txt` に対する Python の文字列インデックス（コードポイント単位、
改行は LF、`end` は含まない）。

## 編集の手順

質問を足したり直したりするときは `qa_draft.yaml` だけを触る。
オフセットは手で書かない——必ずずれるため、引用した本文から機械に計算させる。

```bash
uv run python datasets/build_qa_set.py --check   # 検査だけ
uv run python datasets/build_qa_set.py           # qa_set.jsonl / qa_set.meta.json を生成
uv run pytest tests/unit/test_qa_set.py          # 生成物とコーパスの突き合わせ
```

検査に落ちる代表例は、引用がコーパスに見つからない（タイプミス）、
引用が2回以上現れる（根拠が一意に定まらない）、質問が本文を写している、の3つ。

評価セットを変更したら `version` を上げ、過去の実験結果と混ぜないこと。

## corpus/

**再配布可能なライセンスの公開文書のみ**を置く。出典・版・ライセンスは
[corpus/LICENSE-NOTICE.md](corpus/LICENSE-NOTICE.md) が `fetch_corpus.py` によって自動で更新される。

```bash
uv run python datasets/fetch_corpus.py --check   # コミット済みと最新版の差分を確認
uv run python datasets/fetch_corpus.py           # 取得して上書き
```

ウィキペディアの記事は更新されるため、再取得すると本文が変わり、**評価セットのオフセットがずれる**。
通常は再取得せず、コミット済みの `.txt` をそのまま使う。再取得した場合は評価セットを作り直し、
version を上げること。

選定条件と選定結果は [../docs/02_evaluation.md](../docs/02_evaluation.md) の「コーパス」を参照。
