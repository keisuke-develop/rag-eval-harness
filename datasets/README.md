# datasets

## corpus/

評価に使う文書。**再配布可能なライセンスの公開文書のみ**を置く。
文書を追加したら `LICENSE-NOTICE.md` に出典とライセンスを追記する。

選定条件は [../docs/02_evaluation.md](../docs/02_evaluation.md) の「コーパス」を参照。

## qa_set.jsonl

評価セット。1行1質問。

```json
{"qid": "q001", "question": "...", "gold_spans": [{"doc": "doc03", "start": 1420, "end": 1655}], "answerable": true}
{"qid": "q042", "question": "...", "gold_spans": [], "answerable": false}
```

正解根拠を**チャンクIDではなく文字オフセットで持つ**のが要点。
チャンク分割の条件を変えるとチャンクIDが変わるため、IDで持つとチャンクサイズの比較実験が成立しなくなる。

作成手順と質問作成の基準は [../docs/02_evaluation.md](../docs/02_evaluation.md) に記載。
