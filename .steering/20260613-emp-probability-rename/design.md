# Design: `empirical_probability_*` を `emp_probability_*` にリネーム

## 設計方針

純粋な機械的リネームである一方、**「`empirical` という文字列を全置換してはいけない」**点に最大の注意が必要。本パッケージでは `empirical` という文字列が以下の**4つの異なる意味**で使われており、それぞれ扱いが異なる。

| # | 用途 | 例 | 扱い |
|---|------|-----|------|
| 1 | **属性名識別子** | `self.empirical_probability_` | **リネーム対象** |
| 2 | **CSV カラム名（文字列）** | `rename(columns={"probability": "empirical_probability"})` | **リネーム対象** |
| 3 | **kind エイリアスのキー** | `_KIND_ALIASES = {"empirical": "emp", ...}` | **絶対に変更しない** |
| 4 | **英語 docstring の自然言語** | "empirical recency marginal" 等 | **基本的に維持**（必要に応じ整理） |

`replace_all` 等の機械的全置換は使えない。**個別の文字列パターンを意識的に分けて置換**する。

## 置換ルール

### 置換するパターン（attribute / column name）

| Before | After | スコープ |
|--------|-------|--------|
| `empirical_probability_` | `emp_probability_` | 属性識別子（末尾アンダースコア付き） |
| `empirical_probability_table_` | `emp_probability_table_` | 属性識別子 |
| `empirical_probability_dict_` | `emp_probability_dict_` | 属性識別子 |
| `"empirical_probability"` | `"emp_probability"` | CSV カラム名（文字列リテラル） |
| `"empirical_probability_table_"` | `"emp_probability_table_"` | show() の print ラベル |

実装上の置換キーは「`empirical_probability_`」と「`empirical_probability`」の2系統で十分。長い方から置換することで誤マッチを防ぐ。

### 維持するパターン（絶対変更禁止）

| 維持するパターン | 理由 |
|-----------------|------|
| `_KIND_ALIASES = {"empirical": "emp", ...}` の **キー文字列** | kind alias 機能を維持するため。`predict(kind="empirical")` 等の旧 API を壊さない |
| `"empirical_recency": "er"` のキー | 同上 |
| `"empirical_frequency": "ef"` のキー | 同上 |

### 維持または整理（英語 docstring）

scorer.py の英語 docstring 内で "empirical" が形容詞として使われている箇所：

```
"empirical recency marginal"
"empirical frequency marginal"
"empirical marginal (R2Prob or F2Prob)"
"empirical product-choice probabilities"
"empirical revisit"  (もう存在しないはず)
```

これらは英語として「経験的な〜」という意味で正しく使われており、属性名の話ではないため**そのまま維持**する。

## ファイル別の変更点

### `src/rfscorer/scorer.py`

#### __init__()
```python
# Before
self.empirical_probability_ = None
self.empirical_probability_table_ = None
self.empirical_probability_dict_ = None

# After
self.emp_probability_ = None
self.emp_probability_table_ = None
self.emp_probability_dict_ = None
```

コメント `# empirical` セクションヘッダもそのまま（英語のラベルとして）または `# emp` に更新可。

#### _fit_impl()
```python
# Before
self.empirical_probability_dict_ = {(r, f): prob for r, f, _, _, prob in RowsRF}
self.empirical_probability_ = pd.DataFrame(...)
self.empirical_probability_table_ = self.empirical_probability_.pivot_table(...)
df_r = self.empirical_probability_.groupby("recency")[["N", "cv"]].sum().reset_index()
df_f = self.empirical_probability_.groupby("frequency")[["N", "cv"]].sum().reset_index()

# After
self.emp_probability_dict_ = {(r, f): prob for r, f, _, _, prob in RowsRF}
self.emp_probability_ = pd.DataFrame(...)
self.emp_probability_table_ = self.emp_probability_.pivot_table(...)
df_r = self.emp_probability_.groupby("recency")[["N", "cv"]].sum().reset_index()
df_f = self.emp_probability_.groupby("frequency")[["N", "cv"]].sum().reset_index()
```

#### predict() / transform() / _probability_dict()
```python
# Before
if kind in ("emp", "er", "ef") and self.empirical_probability_dict_ is None:
prob = self.empirical_probability_dict_.get((r, f), 0.0)
return self.empirical_probability_dict_
if self.empirical_probability_dict_ is None:

# After
if kind in ("emp", "er", "ef") and self.emp_probability_dict_ is None:
prob = self.emp_probability_dict_.get((r, f), 0.0)
return self.emp_probability_dict_
if self.emp_probability_dict_ is None:
```

#### plot_probability_surface()
```python
# Before
if kind == "emp" and self.empirical_probability_table_ is None:
table = self.empirical_probability_table_

# After
if kind == "emp" and self.emp_probability_table_ is None:
table = self.emp_probability_table_
```

#### show()
```python
# Before
if self.empirical_probability_table_ is not None:
    print("empirical_probability_table_:")
    print(self.empirical_probability_table_.round(3).to_string())

# After
if self.emp_probability_table_ is not None:
    print("emp_probability_table_:")
    print(self.emp_probability_table_.round(3).to_string())
```

#### export_probability_csv()
```python
# Before
if kind == "all":
    df = (
        self.empirical_probability_.rename(columns={"probability": "empirical_probability"})
        ...
    )
elif kind == "emp":
    df = self.empirical_probability_

# After
if kind == "all":
    df = (
        self.emp_probability_.rename(columns={"probability": "emp_probability"})
        ...
    )
elif kind == "emp":
    df = self.emp_probability_
```

#### docstring (Notes section など)
```python
# Before
``empirical_probability_``, ``empirical_probability_table_``,
``empirical_probability_dict_``, ``recency_probability_``,

# After
``emp_probability_``, ``emp_probability_table_``,
``emp_probability_dict_``, ``recency_probability_``,
```

#### __main__ ブロック
`scorer.plot_probability_surface("empirical")` のような呼び出しがある場合、これは kind alias を渡しているので変更不要。

#### _KIND_ALIASES（**変更しない**）
```python
_KIND_ALIASES = {
    "empirical": "emp",          # キー: 変更しない
    "empirical_recency": "er",   # キー: 変更しない
    "empirical_frequency": "ef", # キー: 変更しない
    ...
}
```

### `src/rfscorer/optimizer.py`

docstring 内で "empirical" は形容詞として使われている可能性が高い。属性名としての `empirical_probability_*` の参照はないと予想されるが、念のため確認して変更があれば更新。

### `tests/test_scorer.py`

#### 属性参照の更新
```python
# Before
scorer_fitted.empirical_probability_
scorer_fitted.empirical_probability_table_
scorer_fitted.empirical_probability_dict_

# After
scorer_fitted.emp_probability_
scorer_fitted.emp_probability_table_
scorer_fitted.emp_probability_dict_
```

#### CSV テストの期待カラム集合
```python
# Before
assert set(df.columns) == {..., "empirical_probability", ...}

# After
assert set(df.columns) == {..., "emp_probability", ...}
```

#### テストメソッド名
`test_empirical_*` のような名前はあれば検出して更新（`test_emp_*` か、より説明的な名前へ）。grep して個別確認。

### `docs/product-requirements.md`

- 出力表（L90 周辺）: `empirical_probability_` → `emp_probability_`
- コード例（L129 周辺）: `df_emp = scorer.empirical_probability_` → `df_emp = scorer.emp_probability_`

### `docs/functional-design.md`

- 属性表内の `empirical_probability_` / `_table_` / `_dict_` 行
- データフロー図内の `empirical_probability_ / _table_ / _dict_` 表記
- 各メソッドの Notes セクション内の属性名列挙

### `docs/glossary.md`

- `empirical_probability_` / `empirical_probability_table_` / `empirical_probability_dict_` の3エントリ

### `docs/development-guidelines.md`

- `empirical` で grep して該当があれば更新（おそらく単発の言及）

### `examples/basic_usage.ipynb`

- 該当セルがあれば `scorer.empirical_probability_` → `scorer.emp_probability_` に更新
- 出力セルの再実行は別タスク（コードセルの修正だけで OK、出力リフレッシュは Optional）

### `README.md`

- 英語の説明内に `empirical_probability_` の参照があれば更新

## 実装手順（推奨順）

1. **scorer.py をリネーム**（中核）
   - __init__, _fit_impl, predict, transform, _probability_dict, plot_probability_surface, show, export_probability_csv, docstring の Notes 部
2. **optimizer.py をチェック**（おそらく変更なし、確認のみ）
3. **tests/test_scorer.py をリネーム**
   - 属性参照、CSV 期待カラム、テストメソッド名
4. **docs を順次更新**
   - product-requirements.md, functional-design.md, glossary.md, development-guidelines.md
5. **examples/basic_usage.ipynb をチェック**
6. **README.md をチェック**
7. **品質チェック**
   - `uv run ruff check .` / `uv run ruff format --check .` / `uv run pytest`

## リスクと対策

| リスク | 対策 |
|--------|------|
| kind alias `"empirical"` を誤って削除 | `_KIND_ALIASES` dict の編集禁止を design.md で明記。grep で `"empirical":` のキー文字列が残っていることを最終確認 |
| docstring 内の英語 "empirical" を不必要に書き換え | "empirical" 単独の英単語は属性名ではないため、`empirical_probability` という連結文字列のみを対象に置換 |
| `empirical_recency_*` / `empirical_frequency_*` 属性が存在すると誤認 | これらは存在しない属性で kind alias のキーのみ。grep `\bempirical_` で残った結果を kind alias 由来かどうか目視確認 |
| `examples/basic_usage.ipynb` の出力セルが古い `empirical` 表示のまま | 出力セルはノートブック再実行で更新される。コードセルだけ修正し、必要なら別途再実行 |
| テスト件数の予期せぬ変動 | 純粋なリネームのため、テストの増減は0が期待値。Phase 5 で件数を確認 |

## 検証方法

完了確認のためのコマンド：

```bash
# kind alias 以外の "empirical_probability" の残存をチェック
grep -rn "empirical_probability" --include="*.py" --include="*.md" --include="*.ipynb" \
  | grep -v "^\.steering" | grep -v "^CHANGELOG"
# → 残存ゼロが期待値

# kind alias の維持を確認
grep -n '"empirical":' src/rfscorer/scorer.py
grep -n '"empirical_recency":' src/rfscorer/scorer.py
grep -n '"empirical_frequency":' src/rfscorer/scorer.py
# → それぞれ1件ずつ残っていることが期待値

# kind alias テストが通ることを確認
uv run pytest tests/test_scorer.py::TestKindAliases -v
```
