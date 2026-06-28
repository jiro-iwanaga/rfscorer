# Design: view_recency の追加

## 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
|----------|----------|------|
| `src/rfscorer/_time_utils.py` | 追加 | `normalize_view_key()`（時刻を保持する高解像度キー生成） |
| `src/rfscorer/scorer.py` | 修正 | `__init__`, `_to_internal`, `transform`, `_build_ui_rf_df`（実装＋docstring）, `show`, `save_zip`, `fit`/`transform` docstring |
| `tests/test_scorer.py` | 修正 | view recency のテスト追加 |
| `docs/functional-design.md` | 修正 | `recency_mode` パラメータの記述追加 |

## 実装アプローチ

### 1. `__init__()` への `recency_mode` 追加

`recency_mode: str = "day"` を追加し、`__init__` 内でバリデーションする。

```python
_VALID_RECENCY_MODES = frozenset({"day", "view"})

def __init__(self, user_col="user", item_col="item", time_col="datetime", unit=1, recency_mode="day"):
    ...
    if recency_mode not in self._VALID_RECENCY_MODES:
        raise ValueError(
            f"recency_mode must be 'day' or 'view', got {recency_mode!r}."
        )
    self.recency_mode = recency_mode
```

- 型は `str`（`Literal` は使わない）。将来 mode が増えても `_VALID_RECENCY_MODES` の追加だけで対応できる。
- `_VALID_RECENCY_MODES` はクラス変数として定義し、`_build_ui_rf_df` の防御的ガードとも共有する。

### 2. 高解像度時刻キーの併走（view モードの核心）

#### 問題

`_time_utils.normalize_sequence_col()` は datetime を **日単位の序数に切り捨てる**（`.dt.days`）。
`_to_internal()` / `transform()` は全経路でこれを通すため、内部の `_SEQUENCE_COL` では
同一日内の時刻（17:00 と 16:50 等）が区別できない。一方 `fit_rolling()` の窓フィルタや
`observation_start_/end_` はこの日単位序数に依存しているため、`_SEQUENCE_COL` を高解像度に
差し替えることはできない。

#### 方針

view モードでのみ、**日単位 `_SEQUENCE_COL` はそのまま維持しつつ、時刻を保持した別カラム
`_VIEW_KEY_COL`（高解像度キー）を内部フレームに併走**させ、view ランクの順位付けに用いる。
day モードでは一切生成せず、既存挙動を完全保存する。

#### `normalize_view_key()`（`_time_utils.py` に追加）

`normalize_sequence_col()` と同じ dtype 分岐で、順序を保つ整数キーを返す。

```python
def normalize_view_key(series: pd.Series) -> pd.Series:
    """Normalize a time column to an order-preserving integer key (sub-day resolution kept)."""
    if is_datetime64_any_dtype(series):
        return (series - _ORDINAL_ORIGIN) // pd.Timedelta("1ns")
    elif is_string_dtype(series):
        return (pd.to_datetime(series) - _ORDINAL_ORIGIN) // pd.Timedelta("1ns")
    elif is_integer_dtype(series) or is_float_dtype(series):
        return series.astype("int64")  # ユーザー指定の粒度をそのまま使う
    else:
        raise ValueError(f"time_col must be datetime or integer type, got {series.dtype}")
```

- `normalize_sequence_col` と同じ origin / timedelta 減算スタイルに揃える（解像度のみ日→ns に変える）。
  `series.astype("int64")` は pandas 2.x で datetime64 に対し `TypeError` になる版があるため避ける。
- ns 序数の有効範囲は datetime64[ns] と同一（約 1678–2262）で、追加のオーバーフロー懸念はない。
- 絶対値ではなく user 内の相対順序のみ使うため、ns と整数の混在は問題にならない。
- 整数 time_col は元々切り捨てが無いため、この経路でも値をそのまま使う。

#### `_to_internal()` / `transform()` での生成

両者とも「元の time 列をコピー → `normalize_sequence_col` で上書き」している。
**上書き前**に元の値から高解像度キーを生成する（view モードのみ）。

```python
# _to_internal() / transform() 内、normalize_sequence_col による上書きの直前に挿入
if self.recency_mode == "view":
    result[self._VIEW_KEY_COL] = normalize_view_key(result[self._SEQUENCE_COL])
result[self._SEQUENCE_COL] = normalize_sequence_col(result[self._SEQUENCE_COL])
```

- `_VIEW_KEY_COL = "_view_key"` をクラス定数（内部列名）として追加する。
- この列は行フィルタ（`fit_rolling` の窓抽出）でも保持され、`_build_ui_rf_df` まで到達する。
- 下流は `_build_ui_rf_df` が列を明示選択して返すため、`_VIEW_KEY_COL` は最終出力に漏れない。

### 3. `_build_ui_rf_df()` のリファクタリング

現行の実装はイベント行ごとに recency を計算してから groupby min を取っている。
リファクタリング後は **先に (user, item) 単位で集計し、その後 `recency_mode` で分岐**する。
これにより recency 軸・frequency 軸それぞれの計算ロジックが独立し、将来の `frequency_mode` 追加も同じパターンで行える。

**現行:**
```python
def _build_ui_rf_df(self, df, ref_int):
    tmp = df[[self._USER_COL, self._ITEM_COL]].copy()
    tmp["recency"] = (ref_int - df[self._SEQUENCE_COL]) // self.unit + 1
    return (
        tmp.groupby([self._USER_COL, self._ITEM_COL], sort=False)
        .agg(recency=("recency", "min"), frequency=("recency", "count"))
        .reset_index()
    )
```

**リファクタリング後:**
```python
def _build_ui_rf_df(self, df, ref_int):
    df = df.reset_index(drop=True)
    # day モードは日単位序数、view モードは高解像度キーで last_ts を集約する
    ts_col = self._VIEW_KEY_COL if self.recency_mode == "view" else self._SEQUENCE_COL
    ui = (
        df.assign(_row_idx=df.index)
        .groupby([self._USER_COL, self._ITEM_COL], sort=False)
        .agg(
            last_ts=(ts_col, "max"),
            frequency=(self._SEQUENCE_COL, "count"),
            first_idx=("_row_idx", "min"),
        )
        .reset_index()
    )
    if self.recency_mode == "day":
        ui["recency"] = (ref_int - ui["last_ts"]) // self.unit + 1
    elif self.recency_mode == "view":
        # Rank items per user by most-recent view (newest first), breaking ties
        # by first appearance in the input (smaller row index first). cumcount on
        # the sorted frame yields a 1-indexed rank (same pattern as transform()).
        ui = ui.sort_values(
            [self._USER_COL, "last_ts", "first_idx"],
            ascending=[True, False, True],
        )
        ui["recency"] = ui.groupby(self._USER_COL, sort=False).cumcount() + 1
    else:
        raise ValueError(
            f"recency_mode must be 'day' or 'view', got {self.recency_mode!r}."
        )
    # last_ts and first_idx are intermediate columns; excluded by explicit column selection
    return ui[[self._USER_COL, self._ITEM_COL, "recency", "frequency"]]
```

> `frequency` のカウント源は両モードとも `_SEQUENCE_COL`（行数カウントなので列の選択は結果に影響しない）。
> `last_ts` の集約源のみモードで切り替える。

> 戻り値の行順は view モードで user 昇順に変わるが、下流（`_aggregate_empirical` の recency/frequency 集約、`transform()` の再ソート）は行順に依存しないため影響しない。

**等価性の確認（day モード）:**

現行は `min((ref - ts) // unit + 1)` を groupby で計算しており、
リファクタリング後は `(ref - max(ts)) // unit + 1` を直接計算する。
`(ref - ts) // unit` は ts が大きいほど小さくなるため、`min = max(ts)` 対応の値となり、数学的に等価。

**将来 `frequency_mode` を追加するときのパターン:**
```python
# frequency_mode="day" が追加される場合のイメージ（本タスクでは実装しない）
if self.frequency_mode == "view":
    pass  # frequency = count（現行のまま）
elif self.frequency_mode == "day":
    ui["frequency"] = (
        df.groupby([self._USER_COL, self._ITEM_COL])[self._SEQUENCE_COL]
        .nunique()  # ユニーク日付数
        .values
    )
```

### 4. `ref` / `ref_int` の扱い（`fit` / `transform`）

`fit()` と `transform()` はどちらも `ref_int` を計算してから `_build_ui_rf_df()` に渡す。
`recency_mode="view"` 時、`ref_int` は recency 計算には使われないが、**コードの変更はしない**。

- `fit()`: `self.observation_end_ = ref_int` で使われるため計算自体は必要
- `transform()`: `ref_int` は `_build_ui_rf_df()` に渡されるが view mode では無視される

いずれも警告を出さず、silently ignore とする（`fit()` の既存方針に合わせる）。

### 5. `show()` への `recency_mode` 表示追加

Model セクションに `recency_mode` を追記する。

```python
# 変更前
print(f"  recency_limit    : {self.recency_limit}")
print(f"  frequency_limit  : {self.frequency_limit}")

# 変更後
print(f"  recency_mode     : {self.recency_mode}")
print(f"  recency_limit    : {self.recency_limit}")
print(f"  frequency_limit  : {self.frequency_limit}")
```

### 6. `save_zip()` メタデータへの `recency_mode` 追加

`save_zip()` 内の metadata dict に `recency_mode` を追加する。
（`save()` は pickle のみで metadata dict を持たない。pickle で `self` ごと保存するため `load()` / `load_zip()` での復元は自動で行われる。metadata は `save_zip()` が書く人間可読の JSON のみ。）

```python
metadata = {
    ...
    "unit": _to_python(self.unit),
    "recency_mode": self.recency_mode,   # 追加
    "recency_limit": _to_python(self.recency_limit),
    ...
}
```

### 7. docstring 更新

**`__init__()`**: `recency_mode` パラメータの説明を追加。

```
recency_mode : str, default "day"
    How recency is computed for each (user, item) pair.
    "day"  : elapsed-days bin relative to ref: ``(ref - last_view) // unit + 1``.
    "view" : 1-indexed rank within each user ordered by most-recent view
             timestamp (1 = latest). Full timestamp resolution is used
             (sub-day order is preserved). ``ref`` and ``unit`` are ignored.
```

**`_build_ui_rf_df()`**: 現行 docstring は day 固定（`Recency is (ref_int - value) // unit + 1`）。day / view 両モードの recency 定義に書き換える。

**`fit()`**: `ref` パラメータの説明に `recency_mode="view"` 時は無視される旨を追記。

**`transform()`**: `ref` パラメータの説明に同様の注記を追記。

## 影響範囲まとめ

| 対象 | 変更内容 | 変更種別 |
|------|----------|----------|
| `_time_utils.normalize_view_key()` | 高解像度時刻キー生成関数を追加 | 機能追加 |
| `__init__` | `recency_mode` パラメータ追加・バリデーション・`_VIEW_KEY_COL` 定数 | 機能追加 |
| `_to_internal()` | view モード時に `_VIEW_KEY_COL` 列を併走生成 | 機能追加 |
| `transform()` | view モード時に `_VIEW_KEY_COL` 列を併走生成 ＋ docstring | 機能追加 |
| `_build_ui_rf_df` | 先に groupby →`recency_mode` で `last_ts` 集約源を切替・分岐 ＋ docstring | リファクタリング＋機能追加 |
| `fit()` | docstring のみ | ドキュメント |
| `show()` | `recency_mode` 表示追加 | 機能追加 |
| `save_zip()` | metadata.json に `recency_mode` 追加 | 機能追加 |
| `save()` / `load()` / `load_zip()` | 変更なし（pickle 復元で自動対応） | — |
| テスト | view recency のテスト追加 | テスト追加 |
| `docs/functional-design.md` | `recency_mode` の記述追加 | ドキュメント |
