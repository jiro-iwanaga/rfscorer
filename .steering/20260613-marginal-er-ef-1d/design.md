# Design: er / ef を1次元周辺確率モデルにリファクタする

## 設計方針

`mr` / `mf` 1次元化リファクタ（過去の `refactor: implement 1D mr/mf models without 2D broadcast`）と**完全に同じパターン**を踏襲する。新規概念は導入せず、既存の1次元処理を er / ef に拡張する形で実装する。

## 構造変更

### 属性の構造変更

#### Before
| 属性 | 型 | 構造 |
|------|-----|------|
| `er_probability_` | `pd.DataFrame` | カラム: `recency`, `frequency`, `probability`（R2Prob を全 f にブロードキャスト） |
| `er_probability_dict_` | `dict[tuple[int, int], float]` | キー: `(r, f)` |
| `er_probability_table_` | `pd.DataFrame` | 2D pivot（index: recency, columns: frequency） |
| `ef_probability_` | `pd.DataFrame` | カラム: `recency`, `frequency`, `probability`（F2Prob を全 r にブロードキャスト） |
| `ef_probability_dict_` | `dict[tuple[int, int], float]` | キー: `(r, f)` |
| `ef_probability_table_` | `pd.DataFrame` | 2D pivot |

#### After
| 属性 | 型 | 構造 |
|------|-----|------|
| `er_probability_` | `pd.DataFrame` | カラム: `recency`, `probability`（mr_probability_ と同じ形式） |
| `er_probability_dict_` | `dict[int, float]` | キー: `r` |
| `er_probability_table_` | — | **削除** |
| `ef_probability_` | `pd.DataFrame` | カラム: `frequency`, `probability`（mf_probability_ と同じ形式） |
| `ef_probability_dict_` | `dict[int, float]` | キー: `f` |
| `ef_probability_table_` | — | **削除** |

### 内部辞書（変更なし）

- `R2N`, `R2CV`, `R2Prob`（既存）: 周辺集約値の保持
- `F2N`, `F2CV`, `F2Prob`（既存）: 周辺集約値の保持
- `recency_probability_`, `frequency_probability_`（既存 DataFrame）: 周辺集約値 + N/cv の DataFrame

er/ef の値は `R2Prob` / `F2Prob` と一致する（実質的に再構成 DataFrame）。

## 各メソッドの設計変更

### `__init__()`

- `self.er_probability_table_ = None` を削除
- `self.ef_probability_table_ = None` を削除

### `_fit_impl()`

#### Before
```python
rows_er = [(r, f, self.R2Prob[r]) for r in self.R for f in self.F]
self.er_probability_dict_ = {(r, f): p for r, f, p in rows_er}
self.er_probability_ = pd.DataFrame(rows_er, columns=["recency", "frequency", "probability"])
self.er_probability_table_ = self.er_probability_.pivot_table(
    index="recency", columns="frequency", values="probability"
)

rows_ef = [(r, f, self.F2Prob[f]) for r in self.R for f in self.F]
# ...同様
```

#### After
```python
self.er_probability_dict_ = dict(self.R2Prob)  # キーは r、値は周辺確率
self.er_probability_ = (
    pd.DataFrame(list(self.R2Prob.items()), columns=["recency", "probability"])
    .sort_values("recency")
    .reset_index(drop=True)
)

self.ef_probability_dict_ = dict(self.F2Prob)
self.ef_probability_ = (
    pd.DataFrame(list(self.F2Prob.items()), columns=["frequency", "probability"])
    .sort_values("frequency")
    .reset_index(drop=True)
)
```

mr/mf の構築コードと完全に同形になる。

### `predict()`

#### 変更点
`er` / `ef` を 1次元クランプ + 1次元辞書参照に変更。`mr` / `mf` の既存ロジックと共通化する。

#### Before（1D 部の抜粋）
```python
if kind == "mr":
    r_clipped = min(r, self.recency_limit)
    return self.mr_probability_dict_.get(r_clipped, 0.0)
if kind == "mf":
    f_clipped = min(f, self.frequency_limit)
    return self.mf_probability_dict_.get(f_clipped, 0.0)

# 2D 部
r = min(r, self.recency_limit)
f = min(f, self.frequency_limit)
if kind == "emp":
    prob = self.empirical_probability_dict_.get((r, f), 0.0)
elif kind == "er":
    prob = self.er_probability_dict_.get((r, f), 0.0)
elif kind == "ef":
    prob = self.ef_probability_dict_.get((r, f), 0.0)
elif kind == "mono":
    ...
```

#### After
```python
if kind in ("mr", "er"):
    r_clipped = min(r, self.recency_limit)
    return self._marginal_dict(kind).get(r_clipped, 0.0)
if kind in ("mf", "ef"):
    f_clipped = min(f, self.frequency_limit)
    return self._marginal_dict(kind).get(f_clipped, 0.0)

# 2D 部
r = min(r, self.recency_limit)
f = min(f, self.frequency_limit)
if kind == "emp":
    prob = self.empirical_probability_dict_.get((r, f), 0.0)
elif kind == "mono":
    prob = self.mono_probability_dict_.get((r, f), 0.0)
# ... 2D models のみ
```

新規ヘルパー `_marginal_dict()` を追加（または直接 if 分岐で書く）：
```python
def _marginal_dict(self, kind):
    if kind == "mr":
        return self.mr_probability_dict_
    if kind == "mf":
        return self.mf_probability_dict_
    if kind == "er":
        return self.er_probability_dict_
    if kind == "ef":
        return self.ef_probability_dict_
    raise ValueError(f"_marginal_dict called with non-marginal kind: {kind!r}")
```

### `_probability_dict()`

#### 変更点
er / ef を削除（marginal 側に移動するため）

#### Before
```python
def _probability_dict(self, kind):
    if kind == "er":
        return self.er_probability_dict_
    if kind == "ef":
        return self.ef_probability_dict_
    if kind == "mono":
        ...
```

#### After
```python
def _probability_dict(self, kind):
    if kind == "mono":
        return self.mono_probability_dict_
    if kind == "mrc":
        return self.mrc_probability_dict_
    if kind == "mfc":
        return self.mfc_probability_dict_
    if kind == "mcc":
        return self.mcc_probability_dict_
    return self.empirical_probability_dict_  # emp はデフォルト
```

### `transform()`

#### 変更点
er / ef を 1次元 merge に統合（mr / mf と共通ロジック）

#### Before
```python
if kind == "mr":
    prob_df = pd.DataFrame(
        list(self.mr_probability_dict_.items()),
        columns=["recency_adj", "probability"],
    )
    df_rf = df_rf.merge(prob_df, on="recency_adj", how="left")
elif kind == "mf":
    ...
else:
    # 2D（emp, er, ef, mono, mrc, mfc, mcc）
    prob_dict = self._probability_dict(kind)
    prob_df = pd.DataFrame(...)
    df_rf = df_rf.merge(prob_df, on=["recency_adj", "frequency_adj"], how="left")
```

#### After
```python
if kind in ("mr", "er"):
    prob_df = pd.DataFrame(
        list(self._marginal_dict(kind).items()),
        columns=["recency_adj", "probability"],
    )
    df_rf = df_rf.merge(prob_df, on="recency_adj", how="left")
elif kind in ("mf", "ef"):
    prob_df = pd.DataFrame(
        list(self._marginal_dict(kind).items()),
        columns=["frequency_adj", "probability"],
    )
    df_rf = df_rf.merge(prob_df, on="frequency_adj", how="left")
else:
    # 2D（emp, mono, mrc, mfc, mcc）のみ
    ...
```

### `plot_probability_surface()`

#### 変更点
- 有効な `kind` から er / ef を削除
- 不正な `kind` への分岐に er / ef を追加（mr / mf と同じ 1D 拒否扱い）

#### Before
```python
if kind in ("mr", "mf"):
    raise ValueError(
        f"kind={kind!r} is a 1D marginal model and cannot be plotted as a surface."
        " Use plot_marginal_probability() instead."
    )
if kind not in ("emp", "er", "ef", "mono", "mrc", "mfc", "mcc"):
    raise ValueError(...)

if kind in ("emp", "er", "ef") and self.empirical_probability_table_ is None:
    ...

if kind == "er":
    table = self.er_probability_table_
elif kind == "ef":
    table = self.ef_probability_table_
...
```

#### After
```python
if kind in ("mr", "mf", "er", "ef"):
    raise ValueError(
        f"kind={kind!r} is a 1D marginal model and cannot be plotted as a surface."
        " Use plot_marginal_probability() instead."
    )
if kind not in ("emp", "mono", "mrc", "mfc", "mcc"):
    raise ValueError(...)

if kind == "emp" and self.empirical_probability_table_ is None:
    raise RuntimeError(...)
# er / ef のブランチ削除

if kind == "emp":
    table = self.empirical_probability_table_
# er / ef ケース削除
elif kind == "mono":
    ...
```

### `plot_marginal_probability()`

#### 変更点
- 有効な `kind` に `er` / `ef` を追加
- 軸×kind の妥当性チェックに er / ef を追加
- データソース選択ロジックを更新

#### Before
```python
valid_kinds = ("emp", "mr", "mf", "all")
if kind not in valid_kinds:
    raise ValueError(...)
if axis == "recency" and kind == "mf":
    raise ValueError("kind='mf' is not valid when axis='recency'. Use kind='mr'.")
if axis == "frequency" and kind == "mr":
    raise ValueError("kind='mr' is not valid when axis='frequency'. Use kind='mf'.")

opt_kind = "mr" if axis == "recency" else "mf"
if kind in (opt_kind, "all"):
    opt_attr = f"{opt_kind}_probability_"
    if getattr(self, opt_attr) is None:
        raise RuntimeError(...)

if axis == "recency":
    df_emp = self.recency_probability_
else:
    df_emp = self.frequency_probability_
```

#### After
```python
valid_kinds = ("emp", "er", "ef", "mr", "mf", "all")
if kind not in valid_kinds:
    raise ValueError(...)

# 軸×kind の妥当性
recency_invalid = ("mf", "ef")
frequency_invalid = ("mr", "er")
if axis == "recency" and kind in recency_invalid:
    raise ValueError(f"kind={kind!r} is not valid when axis='recency'.")
if axis == "frequency" and kind in frequency_invalid:
    raise ValueError(f"kind={kind!r} is not valid when axis='frequency'.")

# opt_kind は引き続き mr / mf を指す（all 表示用）
opt_kind = "mr" if axis == "recency" else "mf"
if kind in (opt_kind, "all"):
    opt_attr = f"{opt_kind}_probability_"
    if getattr(self, opt_attr) is None:
        raise RuntimeError(...)

# データソース選択
if axis == "recency":
    df_emp = self.recency_probability_  # emp（既存の仕様）
else:
    df_emp = self.frequency_probability_

# er / ef は emp と同じ周辺確率を指すため、kind=="er" でも recency_probability_ を使用
# （データ自体は同じだが、kind 指定での明示的な参照経路を提供）
```

#### `kind="er"` / `kind="ef"` 時の挙動
- データは `recency_probability_` / `frequency_probability_` から取得（emp と同じ値）
- 単独表示（折れ線1本、レジェンドなし）
- `kind="er"` は `axis="recency"` のみ有効、それ以外は `ValueError`

#### `kind="all"` 時の挙動（既存）
- emp（実線）+ 最適化 mr/mf（破線）の重ね表示。**変更なし**

### `export_probability_csv()`

#### 変更点
- `kind="er"` / `kind="ef"` 時のカラム構成を 1D 化
- `kind="all"` のマージロジック更新（er は recency マージ、ef は frequency マージ）

#### Before
```python
if kind == "all":
    df = (
        self.empirical_probability_.rename(...)
        .merge(
            self.er_probability_.rename(...),
            on=["recency", "frequency"],
        )
        .merge(
            self.ef_probability_.rename(...),
            on=["recency", "frequency"],
        )
        .merge(self.mono_probability_..., on=["recency", "frequency"])
        .merge(self.mr_probability_..., on="recency")
        .merge(self.mf_probability_..., on="frequency")
        ...
    )
```

#### After
```python
if kind == "all":
    df = (
        self.empirical_probability_.rename(...)
        .merge(self.mono_probability_..., on=["recency", "frequency"])
        .merge(self.mrc_probability_..., on=["recency", "frequency"])
        .merge(self.mfc_probability_..., on=["recency", "frequency"])
        .merge(self.mcc_probability_..., on=["recency", "frequency"])
        .merge(
            self.er_probability_.rename(columns={"probability": "er_probability"}),
            on="recency",  # 変更: 1D マージ
        )
        .merge(
            self.ef_probability_.rename(columns={"probability": "ef_probability"}),
            on="frequency",  # 変更: 1D マージ
        )
        .merge(self.mr_probability_..., on="recency")
        .merge(self.mf_probability_..., on="frequency")
    )
```

emp / 2D 系を先に joint してから、1D 系（er, ef, mr, mf）をマージ。

### `show()`（変更なし）

`empirical_probability_table_` のみ表示しているため変更不要。

## テスト変更方針

### TestFitPeriodResult（要更新）

- 既存の er/ef テストは2D前提（行数 = R × F、ブロードキャスト検証）
- 1次元前提に書き換え

| 現テスト | 修正方針 |
|---------|---------|
| `test_er_set_after_fit` | `er_probability_table_` チェックを削除 |
| `test_ef_set_after_fit` | 同上 |
| `test_er_constant_across_frequency` | **削除**（1次元なので frequency 次元は存在しない） |
| `test_ef_constant_across_recency` | **削除** |
| `test_er_matches_R2Prob` | 1次元 dict の同値性として書き換え |
| `test_ef_matches_F2Prob` | 同上 |
| `test_er_table_shape` | **削除**（属性自体が消えるため） |
| `test_ef_table_shape` | **削除** |

### TestPredict（要更新）

| 現テスト | 修正方針 |
|---------|---------|
| `test_er_kind` | `er_probability_dict_[(r,f)]` → `er_probability_dict_[r]` |
| `test_ef_kind` | `ef_probability_dict_[(r,f)]` → `ef_probability_dict_[f]` |
| 新規: `test_er_ignores_f` | mr 同様、f を変えても結果不変 |
| 新規: `test_ef_ignores_r` | mf 同様 |
| 新規: `test_clamps_r_to_recency_limit_er` | mr と同じ |
| 新規: `test_clamps_f_to_frequency_limit_ef` | mf と同じ |

### TestTransform（要更新）

| 現テスト | 修正方針 |
|---------|---------|
| 既存の er/ef 検証はおそらく無い（emp 中心） | 新規追加 |
| 新規: `test_er_probability_matches_dict` | mr の対応テストと同様、recency クランプ後の dict 引きと一致 |
| 新規: `test_ef_probability_matches_dict` | mf 同様 |

### TestPlotProbabilitySurface（要更新）

| 現テスト | 修正方針 |
|---------|---------|
| `test_returns_figure_er` | **削除**（`ValueError` に変わるため） |
| `test_returns_figure_ef` | **削除** |
| 新規: `test_er_raises_value_error` | mr の対応テストと同様、`ValueError` + 1D marginal メッセージ |
| 新規: `test_ef_raises_value_error` | 同上 |

### TestPlotMarginalProbability（要追加）

| 新規テスト | 内容 |
|-----------|------|
| `test_returns_figure_er_recency` | `axis="recency", kind="er"` で Figure 返却 |
| `test_returns_figure_ef_frequency` | `axis="frequency", kind="ef"` で Figure 返却 |
| `test_ef_on_recency_axis_raises` | `axis="recency", kind="ef"` で `ValueError` |
| `test_er_on_frequency_axis_raises` | `axis="frequency", kind="er"` で `ValueError` |

### TestExportProbabilityCsv（要更新）

| 現テスト | 修正方針 |
|---------|---------|
| `test_er_output_columns` | カラム `{recency, frequency, probability}` → `{recency, probability}` |
| `test_ef_output_columns` | カラム `{recency, frequency, probability}` → `{frequency, probability}` |
| `test_all_output_columns` | `kind="all"` の全カラム検証（変更なしのはず） |
| `test_all_row_count` | 行数は emp が支配的なので R × F のまま（変更なし） |

## ドキュメント変更方針

### docs/product-requirements.md

- L75（機能テーブル）: er/ef 行の説明文を「ブロードキャストした確率面」→「最新度 / 頻度の周辺確率（1次元）」に変更
- L91, L92（出力テーブル）: `er_probability_` / `ef_probability_` のカラム構成を1次元に
- `er_probability_table_` / `ef_probability_table_` の言及があれば削除（現状なし）

### docs/functional-design.md

- 数式定義（er / ef）: `$x_{r,f} = p_r$` → `$p_r$`（1次元として定義）
- 属性表: `er_probability_` / `ef_probability_` の説明を1次元 DataFrame に。`_table_` 行を削除
- データフロー図: er / ef の説明を「R2Prob を全 f にブロードキャスト」→「R2Prob を1次元 dict として保持」

### docs/glossary.md

- `er_probability_` / `ef_probability_` の構造説明を1次元に
- `er_probability_table_` / `ef_probability_table_` のエントリ削除（mr/mf 同様）
- `er_probability_dict_` / `ef_probability_dict_` のキー型を `(r,f)` → `r` / `f` に

### examples/basic_usage.ipynb

- er / ef を明示的に使う箇所があれば1次元前提に修正
- 現状は plot_probability_surface で `er` / `ef` を呼んでいる可能性あり → 確認して必要に応じて削除または `plot_marginal_probability(kind="er")` 等に変更

## 影響範囲まとめ

| ファイル | 変更点 |
|---------|------|
| `src/rfscorer/scorer.py` | `__init__`, `_fit_impl`, `predict`, `_probability_dict`, `transform`, `plot_probability_surface`, `plot_marginal_probability`, `export_probability_csv` + 新規ヘルパー `_marginal_dict` |
| `tests/test_scorer.py` | TestFitPeriodResult / TestPredict / TestTransform / TestPlotProbabilitySurface / TestPlotMarginalProbability / TestExportProbabilityCsv |
| `docs/product-requirements.md` | 機能・出力テーブル |
| `docs/functional-design.md` | 数式・属性表・データフロー |
| `docs/glossary.md` | 属性エントリ |
| `examples/basic_usage.ipynb` | er/ef の使用確認・修正 |

## リスクと対策

| リスク | 対策 |
|--------|------|
| 既存 `er_probability_table_` 利用箇所の見落とし | grep で全コードベース確認、`AttributeError` で発覚するためテスト実行で検出可能 |
| `kind="all"` マージで NaN が発生する場合 | 1D マージは cross join 的に展開されるため NaN なし（全 (r,f) 組に対応する r または f が必ず存在） |
| `plot_marginal_probability(kind="er"|"ef")` の data source 一意性 | データは `recency_probability_` から取得（既存仕様）。`er_probability_` の DataFrame と値は一致する |
| ノートブック出力が古いまま | `examples/basic_usage.ipynb` を再実行して出力更新（既存の方針を踏襲） |

## バージョニング

- 破壊的変更を含むため、コミット履歴のサマリーで明示
- リリース時に `0.4.0`（破壊的変更の集約点）として一括公開
- CHANGELOG には er/ef の構造変更を Breaking Change として記載
