# Requirements: `empirical_probability_*` を `emp_probability_*` にリネーム

## 背景

本パッケージの確率系属性は、`optimize(kind=...)` および経験値の `kind` 名と一致するプレフィックスを用いた命名規則を持つ。

例:

| 属性 | 対応する `kind` |
|------|----------------|
| `er_probability_` | `er` |
| `ef_probability_` | `ef` |
| `mono_probability_` | `mono` |
| `mr_probability_` / `mf_probability_` | `mr` / `mf` |
| `mrc_probability_` / `mfc_probability_` / `mcc_probability_` | `mrc` / `mfc` / `mcc` |

ただし、**`empirical_probability_` だけが `kind="emp"` に対応しない長形（`empirical`）を採用**しており、命名規則の唯一の例外となっている。同様の問題が以下にも波及している：

- `empirical_probability_table_` / `empirical_probability_dict_`
- `export_probability_csv(kind="all")` の CSV カラム名 `empirical_probability`

この例外は、ユーザーが属性名を予測する際の認知負荷を増やし、ドキュメント・テスト・コードの一貫性を損ねている。

## 目的

`empirical_probability_*` 属性および関連する CSV カラム名を `emp_probability_*` / `emp_probability` にリネームし、命名規則の例外をゼロにする。

## ユーザーストーリー

### US-1: 一貫した属性アクセス
- **As a** データサイエンティスト
- **I want** `scorer.emp_probability_` で経験的商品選択確率にアクセスしたい
- **So that** `er_probability_` / `mr_probability_` などと同じ規則で予測でき、API を覚えやすい

### US-2: 一貫した CSV カラム名
- **As a** 分析者
- **I want** `export_probability_csv(kind="all")` のカラム名で `emp_probability` を期待したい
- **So that** 他の `*_probability` カラムと並んだとき不自然な長さの違いがない

### US-3: ドキュメントの規則の単純化
- **As a** ドキュメント読者
- **I want** 属性名の命名規則が「`{kind}_probability_*`」という1つのルールで説明されることを望む
- **So that** 例外を覚える必要がない

## 受け入れ条件

### AC-1: 属性のリネーム

以下3属性を完全にリネームする：

| Before | After |
|--------|-------|
| `empirical_probability_` | `emp_probability_` |
| `empirical_probability_table_` | `emp_probability_table_` |
| `empirical_probability_dict_` | `emp_probability_dict_` |

互換シム（プロパティ alias 等）は設けない。

### AC-2: CSV カラム名のリネーム

`export_probability_csv(kind="all")` で出力される DataFrame のカラム名を変更：

| Before | After |
|--------|-------|
| `empirical_probability` | `emp_probability` |

`kind="emp"` 単独出力時のデフォルトファイル名 `emp_probability.csv` は元から短形のため変更なし（既存仕様維持）。

### AC-3: メソッド内部参照の更新

`scorer.py` 内で `empirical_probability_*` を参照している全箇所を `emp_probability_*` に更新する。具体的には：

- `__init__()` の属性初期化
- `_fit_impl()` の代入
- `predict()` の参照（`empirical_probability_dict_`）
- `transform()` の事前チェック（`empirical_probability_dict_`）
- `plot_probability_surface()` の事前チェック（`empirical_probability_table_`）
- `_probability_dict()` のデフォルト返却（`empirical_probability_dict_`）
- `show()` の参照（`empirical_probability_table_`）
- `export_probability_csv()` の `kind="all"` マージ起点と `kind="emp"` 分岐

### AC-4: テストの更新

`tests/test_scorer.py` 内で以下を全て更新：

- `scorer_fitted.empirical_probability_*` 参照 → `emp_probability_*`
- CSV 出力テストの期待カラム集合に `empirical_probability` がある箇所 → `emp_probability`
- テストメソッド名で `empirical` が使われている箇所（あれば）→ `emp`

### AC-5: ドキュメントの更新

以下3ファイルで `empirical_probability_*` の言及を全て `emp_probability_*` に更新：

- `docs/product-requirements.md`
- `docs/functional-design.md`
- `docs/glossary.md`

加えて以下のファイルで該当があれば更新：

- `docs/development-guidelines.md`
- `README.md`（英語）

### AC-6: 例の更新

- `examples/basic_usage.ipynb` で `empirical_probability_*` を使用している箇所を `emp_probability_*` に置換

### AC-7: kind エイリアス機能は影響なし

`emp` ↔ `empirical` の kind alias は維持する（ユーザーが `predict(kind="empirical")` 等を使えること）。これは属性名の話とは独立のレイヤー（kind 文字列）であり、リネームの影響を受けない。

### AC-8: 既存テストの全パス

リネーム以外の挙動変更を含まないため、テスト数の増減はなく、既存テストが全パスすることが完了条件。

## 制約事項

### 破壊的変更であること

- 公開属性のリネームは API の破壊的変更
- 互換シムは設けない（CLAUDE.md および過去判断の方針に準拠）
- 現在 `feature/time-col-unit-support` ブランチで複数の破壊的変更を集約しており、次期メジャーリリースに同梱する

### `recency_probability_` / `frequency_probability_` は対象外

これらは `kind` 名と直接対応しない内部派生属性であり、本リネームの範囲外。`kind="emp"` 系の整理のみを行う。

### 内部 dict（`R2N` / `R2Prob` 等）は不変

経験値の集約に使う内部 dict は本リネームと無関係。

### ステアリング・履歴ファイルは不変

`.steering/` 配下の過去ステアリングドキュメントおよび `CHANGELOG.md` の過去エントリは履歴として保持。

## 完了条件

1. `uv run ruff check .` がパス
2. `uv run ruff format --check .` がパス
3. `uv run pytest` が全テスト green（テスト件数は変動しない想定）
4. `docs/` 配下に `empirical_probability` の文字列が残らない（履歴的なファイルを除く）
5. `src/rfscorer/scorer.py` 内に `empirical_probability` の文字列が残らない
6. `tests/test_scorer.py` 内に `empirical_probability` の文字列が残らない
7. `examples/basic_usage.ipynb` で `empirical_probability` の文字列が残らない
