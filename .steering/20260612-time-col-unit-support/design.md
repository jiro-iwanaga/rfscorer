# Design: time_col / unit サポート

## 実装アプローチ

「入力直後に整数へ正規化し、以降はすべて整数で処理する」方針を取る。
型を意識するのは 2 つのプライベートメソッドのみとし、既存のアルゴリズム本体には最小限の変更で済むよう設計する。

## 正規化の変換ルール

| 入力型 | 変換方法 | 例 |
|--------|----------|----|
| `datetime64` | `pd.Timestamp.toordinal()` | `2024-01-01` → `738887` |
| `str` | `pd.to_datetime()` 経由で同上 | `"2024-01-01"` → `738887` |
| `int` / `float` | `int` にキャスト | `100` → `100` |
| その他 | `ValueError` を raise | — |

ordinal（西暦 1 年 1 月 1 日からの通算日数）を採用することで、`target_int - observation_days` のような整数演算がそのまま「日数差」として成立する。

## 変更するコンポーネント

### 1. `__init__` シグネチャ変更

```python
# 変更前
def __init__(self, user_col="user", item_col="item", datetime_col="datetime"):

# 変更後
def __init__(self, user_col="user", item_col="item", time_col="datetime", unit=1):
```

- `datetime_col` → `time_col` にリネーム（削除）
- `unit: int = 1` を追加。`unit <= 0` の場合は `__init__` 内で `ValueError` を raise する

### 2. 内部定数のリネーム

```python
# 変更前
_DATETIME_COL = "datetime"

# 変更後
_SEQUENCE_COL = "datetime"  # 内部カラム名の文字列は変えない（DataFrame の列名のため）
```

クラス変数名のみ変更し、ソースコード内の `self._DATETIME_COL` 参照を `self._SEQUENCE_COL` に一括置換する。

### 3. 新規プライベートメソッド追加

#### `_normalize_ref(value) -> int`

単一の参照値（`ref_date`、`target_date`、期間境界値）を整数に変換する。

```python
def _normalize_ref(self, value) -> int:
    if isinstance(value, (pd.Timestamp, datetime.datetime, datetime.date)):
        return value.toordinal()
    elif isinstance(value, str):
        return pd.to_datetime(value).toordinal()
    elif isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    else:
        raise ValueError(f"time value could not be normalized: {value!r}")
```

#### `_normalize_sequence_col(series) -> pd.Series`

DataFrame のシーケンスカラム全体を整数 Series に変換する。

```python
_ORDINAL_ORIGIN = pd.Timestamp("0001-01-01")  # toordinal() の基準点（ordinal=1）

def _normalize_sequence_col(self, series: pd.Series) -> pd.Series:
    if is_datetime64_any_dtype(series):
        return (series - self._ORDINAL_ORIGIN).dt.days + 1
    elif is_string_dtype(series):
        return (pd.to_datetime(series) - self._ORDINAL_ORIGIN).dt.days + 1
    elif is_integer_dtype(series) or is_float_dtype(series):
        return series.astype(int)
    else:
        raise ValueError(
            f"time_col must be datetime or integer type, got {series.dtype}"
        )
```

`.map(lambda x: x.toordinal())` は Python ループのため大規模データで低速になる。代わりに pandas のベクトル演算 `(series - origin).dt.days + 1` を使う。`pd.Timestamp("0001-01-01").toordinal() == 1` であるため両者は同じ値を返す。

### 4. `_to_internal()` の変更

正規化の呼び出しをここに集約する。これ以降のすべての処理は整数のみを扱う。

```python
def _to_internal(self, df):
    result = df[[self.user_col, self.item_col, self.time_col]].copy()
    result.columns = [self._USER_COL, self._ITEM_COL, self._SEQUENCE_COL]
    if not is_string_dtype(result[self._USER_COL]):
        result[self._USER_COL] = result[self._USER_COL].astype(str)
    if not is_string_dtype(result[self._ITEM_COL]):
        result[self._ITEM_COL] = result[self._ITEM_COL].astype(str)
    # 追加: 日付型・整数型を問わず整数に正規化
    result[self._SEQUENCE_COL] = self._normalize_sequence_col(result[self._SEQUENCE_COL])
    return result
```

### 5. `_build_ui_rf_df()` の変更

`.dt.days` を削除し、整数演算と `unit` による bin 化に変更する。

```python
# 変更前
tmp["recency"] = (ref_date - df[self._DATETIME_COL]).dt.days + 1

# 変更後
tmp["recency"] = (ref_int - df[self._SEQUENCE_COL]) // self.unit + 1
```

引数名も `ref_date` → `ref_int` に変更する。

### 6. `fit()` の変更

- パラメータ: `ref_date=None` → `ref=None`
- `pd.to_datetime()` の呼び出しを `_normalize_ref()` に置き換える

```python
# 変更前
if ref_date is None:
    ref_date = obs_log[self._DATETIME_COL].max()
else:
    ref_date = pd.to_datetime(ref_date)  # ...

# 変更後
if ref is None:
    ref_int = obs_log[self._SEQUENCE_COL].max()
else:
    ref_int = self._normalize_ref(ref)
```

### 7. `fit_date()` の変更

`pd.to_datetime()` と `pd.Timedelta` を削除し、整数演算に置き換える。
日付文字列が渡された場合は `_normalize_ref()` が ordinal に変換するため、`observation_days` の引き算がそのまま成立する。

```python
# 変更前
target_date = pd.to_datetime(target_date)
obs_start = max(df_min, target_date - pd.Timedelta(days=observation_days))
eval_start = target_date + pd.Timedelta(days=1)

# 変更後（_to_internal() 後の interaction_log は整数列）
target_int = self._normalize_ref(target_date)
df_min = interaction_log[self._SEQUENCE_COL].min()
df_max = interaction_log[self._SEQUENCE_COL].max()
obs_start = max(df_min, target_int - observation_days) if observation_days else df_min
obs_end = target_int
eval_start = target_int + 1
eval_end = min(df_max, target_int + evaluation_days) if evaluation_days else df_max
```

整数入力の場合も `target_date` に整数を渡せば同じパスで動作する。

### 8. `fit_period()` の変更

`pd.to_datetime()` を `_normalize_ref()` に置き換える。

```python
# 変更前
obs_start, obs_end = pd.to_datetime(observation_period)

# 変更後
obs_start, obs_end = (self._normalize_ref(v) for v in observation_period)
eval_start, eval_end = (self._normalize_ref(v) for v in evaluation_period)
```

### 9. `transform()` / `transform_date()` の変更

`fit()` と同様に `ref_date` → `ref`、`datetime_col` → `time_col` にリネームし、`_normalize_ref()` を使用する。

### 10. 公開属性のリネーム

現在の `*_date_` 属性は内部整数（ordinal または入力整数）を格納するようになるため、サフィックスから `date` を削除する。

| 変更前 | 変更後 |
|--------|--------|
| `observation_start_date_` | `observation_start_` |
| `observation_end_date_` | `observation_end_` |
| `evaluation_start_date_` | `evaluation_start_` |
| `evaluation_end_date_` | `evaluation_end_` |

## 変更しないコンポーネント

- `_fit_impl()` — `_build_ui_rf_df()` の引数名変更のみ、ロジック変更なし
- `predict()` — RF テーブルの参照のみ、時系列データを扱わない
- `optimizer.py` — 時系列データを扱わない
- `plot_*()` 系メソッド — 同上

## 影響範囲まとめ

| ファイル | 変更箇所 |
|----------|---------|
| `scorer.py` | `__init__`, `_to_internal`, `fit`, `fit_date`, `fit_period`, `transform`, `transform_date`, `_build_ui_rf_df` + 新規 2 メソッド追加 |
| `tests/` | 既存テストの `datetime_col` → `time_col`、`ref_date` → `ref` 置き換え、整数入力テスト追加 |
| `README.md` | 破壊的変更の記載、整数入力の使用例追加 |
| `CHANGELOG.md` | 破壊的変更の明記 |

## インポートの変更

```python
# 変更前
from pandas.api.types import is_datetime64_any_dtype, is_string_dtype

# 変更後
import numpy as np

from pandas.api.types import (
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
)
```
