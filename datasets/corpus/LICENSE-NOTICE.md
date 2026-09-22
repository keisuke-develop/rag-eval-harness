<!-- このファイルは datasets/fetch_corpus.py が生成する。直接編集しない。 -->

# コーパスの出典とライセンス

このディレクトリの `.txt` は、**ウィキペディア日本語版**の記事から取得したものです。

- ライセンス: [クリエイティブ・コモンズ 表示-継承 4.0 国際 (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/deed.ja)
- 著作者表示: 各記事の執筆者（各記事の履歴ページを参照）
- 改変の有無: **あり**（下記「加工の内容」を参照）
- 継承: このディレクトリ配下の `.txt` は CC BY-SA 4.0 で再配布されます。
  リポジトリ本体の MIT ライセンス（[../../LICENSE](../../LICENSE)）とは別の扱いです。

## 収録文書

取得日: 2026-09-21

| doc_id | 記事 | 版 (oldid) | 文字数 | 分野 |
|---|---|---|---|---|
| `doc01_fujisan` | [富士山](https://ja.wikipedia.org/wiki/富士山) | 111087163 | 5,807 | 地理・自然 |
| `doc02_photosynthesis` | [光合成](https://ja.wikipedia.org/wiki/光合成) | 110797416 | 5,777 | 生物・化学 |
| `doc03_hummingbird` | [ハチドリ](https://ja.wikipedia.org/wiki/ハチドリ) | 110299642 | 4,663 | 生物 |
| `doc04_iss` | [国際宇宙ステーション](https://ja.wikipedia.org/wiki/国際宇宙ステーション) | 110740409 | 5,910 | 宇宙開発 |
| `doc05_tcp` | [Transmission Control Protocol](https://ja.wikipedia.org/wiki/Transmission_Control_Protocol) | 110486055 | 5,701 | 情報技術 |
| `doc06_solar_power` | [太陽光発電](https://ja.wikipedia.org/wiki/太陽光発電) | 110898902 | 5,689 | エネルギー |
| `doc07_constitution` | [日本国憲法](https://ja.wikipedia.org/wiki/日本国憲法) | 111066263 | 5,967 | 法 |
| `doc08_world_heritage` | [世界遺産](https://ja.wikipedia.org/wiki/世界遺産) | 110544168 | 5,873 | 国際制度 |
| `doc09_edo_period` | [江戸時代](https://ja.wikipedia.org/wiki/江戸時代) | 110958346 | 5,956 | 歴史 |
| `doc10_tokaido_shinkansen` | [東海道新幹線](https://ja.wikipedia.org/wiki/東海道新幹線) | 110714341 | 4,171 | 交通 |
| `doc11_earthquake` | [地震](https://ja.wikipedia.org/wiki/地震) | 111012052 | 5,897 | 自然科学 |
| `doc12_washi` | [和紙](https://ja.wikipedia.org/wiki/和紙) | 110432896 | 5,085 | 文化・工芸 |

各記事のその版は `https://ja.wikipedia.org/w/index.php?oldid=<版ID>` で参照できます。

## 加工の内容

原文そのままではありません。`datasets/fetch_corpus.py` が以下の加工を行っています。

1. 1行目に記事タイトルを置き、空行を1行はさむ
2. 脚注・出典・参考文献・関連項目・外部リンク・画像などの節を、下位の節ごと削除
3. 数式がプレーンテキスト化で断片化した領域を、ラベル行ごと削除
4. 先頭から 6,000 字を上限に、段落境界で切り出し
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
