# Design: `fit()` のみ公開 + `split_by_date()` ユーティリティ提供

## 設計方針

1. **API surface の最小化**: `RecencyFrequencyScorer` の公開メソッドから data preparation を完全分離する
2. **時刻正規化ロジックの共有モジュール化**: scorer.py 内の `_normalize_ref` / `_normalize_sequence_col` を `_time_utils.py` に切り出し、scorer.py と新規 utils.py の両方から再利用する
3. **将来のユーティリティ拡張に備える**: `rfscorer/utils.py` に集約することで、`split_by_value()` 等の兄弟関数を追加する余地を残す

## モジュール構成

### 新規追加

| ファイル | 役割 |
|---------|------|
| `src/rfscorer/_time_utils.py` | 時刻値・時刻列の正規化ヘルパー（内部モジュール） |
| `src/rfscorer/utils.py` | 公開ユーティリティ（`split_by_date` 等） |
| `tests/test_utils.py` | `utils.py` のテスト |

### 変更

| ファイル | 変更 |
|---------|------|
| `src/rfscorer/scorer.py` | `fit_date()` / `fit_period()` / `transform_date()` 削除、時刻正規化を `_time_utils.py` 経由に変更 |
| `src/rfscorer/__init__.py` | `split_by_date` を export |
| `tests/test_scorer.py` | `fit_date` / `fit_period` / `transform_date` 関連テストを削除、`scorer_fitted` 等のフィクスチャを `fit()` ベースに変更、`TestNormalizeRef` / `TestNormalizeSequenceCol` を `_time_utils` の関数テストに置換 |
| 3 ドキュメント | 該当メソッドの記述削除、`split_by_date` の使い方を追記 |

## `_time_utils.py` 設計

### 公開する関数（パッケージ内 internal）

```python
# src/rfscorer/_time_utils.py
import datetime
import numpy as np
import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_float_dtype,
    is_integer_dtype,
    is_string_dtype,
)

_ORDINAL_ORIGIN = pd.Timestamp("0001-01-01")


def normalize_ref(value) -> int:
    """単一の時刻参照値（日付・整数）を ordinal int に正規化する。"""
    if isinstance(value, (pd.Timestamp, datetime.datetime)):
        return value.toordinal()
    elif isinstance(value, str):
        return pd.to_datetime(value).toordinal()
    elif isinstance(value, (int, float, np.integer, np.floating)):
        return int(value)
    else:
        try:
            return int(pd.to_datetime(value).toordinal())
        except Exception:
            raise ValueError(f"time value could not be normalized: {value!r}")


def normalize_sequence_col(series: pd.Series) -> pd.Series:
    """時刻列（datetime / 整数）を ordinal int 列に正規化する。"""
    if is_datetime64_any_dtype(series):
        return (series - _ORDINAL_ORIGIN).dt.days + 1
    elif is_string_dtype(series):
        return (pd.to_datetime(series) - _ORDINAL_ORIGIN).dt.days + 1
    elif is_integer_dtype(series) or is_float_dtype(series):
        return series.astype(int)
    else:
        raise ValueError(f"time_col must be datetime or integer type, got {series.dtype}")
```

### 命名規則

- モジュール名: `_time_utils.py`（先頭 `_` で internal 表明）
- 関数名: アンダースコアなし（モジュール内では public、外部からは internal 扱い）
- 定数: `_ORDINAL_ORIGIN`（モジュール内 private）

### scorer.py からの利用

```python
# src/rfscorer/scorer.py
from ._time_utils import normalize_ref, normalize_sequence_col

class RecencyFrequencyScorer:
    ...

    def _to_internal(self, df):
        ...
        result[self._SEQUENCE_COL] = normalize_sequence_col(result[self._SEQUENCE_COL])
        return result

    def fit(self, df_obs, df_eval, ref=None, ...):
        ...
        if ref is None:
            ref_int = int(obs_log[self._SEQUENCE_COL].max())
        else:
            ref_int = normalize_ref(ref)
        ...
```

- `RecencyFrequencyScorer._normalize_ref()` メソッドは削除
- `RecencyFrequencyScorer._normalize_sequence_col()` メソッドは削除
- `RecencyFrequencyScorer._ORDINAL_ORIGIN` クラス定数は削除（`_time_utils._ORDINAL_ORIGIN` で参照）

## `split_by_date()` 設計

### シグネチャ

```python
# src/rfscorer/utils.py
import pandas as pd
from ._time_utils import normalize_ref, normalize_sequence_col


def split_by_date(
    df: pd.DataFrame,
    target_date,
    observation_days: int | None = 28,
    evaluation_days: int | None = 7,
    time_col: str = "datetime",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """target_date を基準に df を観測ログと評価ログに分割する。

    Parameters
    ----------
    df : pd.DataFrame
        分割対象の DataFrame。time_col 列を含む。
    target_date : str | datetime | int
        観測期間と評価期間の分割点。観測期間は target_date まで（含む）、
        評価期間は target_date の翌時点から始まる。
    observation_days : int or None, default 28
        target_date から遡る観測期間の最大時間単位数（time_col が datetime
        なら日数、整数なら整数ステップ）。None の場合は df の先頭まで使用。
    evaluation_days : int or None, default 7
        target_date から進む評価期間の最大時間単位数。None の場合は
        df の末尾まで使用。
    time_col : str, default "datetime"
        時点カラム名。

    Returns
    -------
    df_obs, df_eval : tuple[pd.DataFrame, pd.DataFrame]
        観測ログと評価ログ。元の df の構造（カラム・型）を保持したサブセット。
        target_date 当日の行は df_obs に含まれる。

    Raises
    ------
    TypeError
        df が pandas DataFrame でない場合。
    ValueError
        time_col が df にない、または target_date が正規化できない場合。

    Examples
    --------
    >>> df_obs, df_eval = split_by_date(df, "2024-01-07", observation_days=28, evaluation_days=7)
    >>> scorer = RecencyFrequencyScorer()
    >>> scorer.fit(df_obs, df_eval)
    """
```

### 実装

```python
def split_by_date(
    df,
    target_date,
    observation_days=28,
    evaluation_days=7,
    time_col="datetime",
):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")
    if time_col not in df.columns:
        raise ValueError(f"Missing required column: {time_col!r}")

    target_int = normalize_ref(target_date)
    normalized = normalize_sequence_col(df[time_col])

    df_min = int(normalized.min())
    df_max = int(normalized.max())

    if observation_days is None:
        obs_start = df_min
    else:
        obs_start = max(df_min, target_int - observation_days)
    obs_end = target_int
    eval_start = target_int + 1
    eval_end = (
        df_max if evaluation_days is None else min(df_max, target_int + evaluation_days)
    )

    obs_mask = (obs_start <= normalized) & (normalized <= obs_end)
    eval_mask = (eval_start <= normalized) & (normalized <= eval_end)

    return df[obs_mask], df[eval_mask]
```

### 設計上の注意

| 観点 | 決定 |
|------|------|
| 戻り値の型 | `tuple[pd.DataFrame, pd.DataFrame]` — unpacking が直感的 |
| 元の df の mutation | 行わない（boolean mask によるサブセット返却） |
| 戻り値の time_col 型 | 元の df の型を保持（normalize した値は内部計算のみに使用） |
| target_date の境界 | 観測期間に含む（`<= target_date`）。評価期間は翌時点（`> target_date`） |
| `unit` パラメータ | 提供しない。observation_days/evaluation_days はあくまで「時間単位の数」であり、recency binning の `unit` とは独立した概念 |

## `__init__.py` 変更

```python
# src/rfscorer/__init__.py
from .scorer import RecencyFrequencyScorer
from .utils import split_by_date

__all__ = ["RecencyFrequencyScorer", "split_by_date"]
```

`RecencyFrequencyOptimizer` の追加は**別タスク**のため本タスクの対象外。

## scorer.py の変更詳細

### メソッド削除

| メソッド | 削除範囲 |
|---------|---------|
| `fit_date()` | メソッド定義全体（docstring 含む） |
| `fit_period()` | メソッド定義全体（docstring 含む） |
| `transform_date()` | メソッド定義全体（docstring 含む） |
| `_normalize_ref()` | メソッド削除（モジュール関数経由に移行） |
| `_normalize_sequence_col()` | メソッド削除（モジュール関数経由に移行） |

### docstring の整合

`fit()` の Notes セクション内で「同じ属性が `fit_date()` / `fit_period()` 後にも利用可能」と書かれている箇所を「fit() 後に利用可能」に修正。
`transform()` の docstring 内で「use transform_date() to apply automatic filtering instead」と案内している箇所を、ユーザーが手動で df をフィルタする例に置き換え。

### import 変更

```python
# scorer.py の冒頭付近
from ._time_utils import normalize_ref, normalize_sequence_col
```

`_ORDINAL_ORIGIN` クラス定数は削除（参照箇所が消えるため不要）。

### 内部参照の置換

| Before | After |
|--------|-------|
| `self._normalize_ref(value)` | `normalize_ref(value)` |
| `self._normalize_sequence_col(series)` | `normalize_sequence_col(series)` |
| `self._ORDINAL_ORIGIN` | （削除、`_time_utils._ORDINAL_ORIGIN` を内部利用） |

### fit() の Notes セクション

Before:
```
Notes
-----
After a successful call, the following attributes become available
for use with predict(), transform(), and plot_*() methods:
``emp_probability_``, ``emp_probability_table_``,
``emp_probability_dict_``, ``recency_probability_``,
``frequency_probability_``, ``recency_limit``, ``frequency_limit``.
```

これは変更不要（`fit_date` / `fit_period` への言及がないため）。

### Estimate empirical product-choice probabilities docstring

`fit_date()` / `fit_period()` の Notes 部分に「same attributes as fit() become available」と書いてある箇所が削除されるが、これらメソッド自体が削除されるため対応不要。

## tests/test_scorer.py の変更詳細

### 削除するテストクラス

| クラス | 削除理由 |
|-------|---------|
| `TestFitDateValidation` | fit_date 削除のため |
| `TestFitDateResult` | 同上 |
| `TestFitPeriodValidation` | fit_period 削除のため |
| `TestFitPeriodResult` | 同上 |
| `TestTransformDate` | transform_date 削除のため |

### フィクスチャの変更

`scorer_fitted`, `scorer_optimized_*`, `scorer_all_optimized`, `df_rec` などが `fit_period` を使っているため、これらを `fit()` ベースに書き換える。

#### Before（例: `scorer_fitted`）
```python
@pytest.fixture(scope="module")
def scorer_fitted():
    s = RecencyFrequencyScorer()
    s.fit_period(
        _make_df(),
        _OBS_PERIOD,
        _EVAL_PERIOD,
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    return s
```

#### After
```python
def _split_by_period(df, obs_period, eval_period):
    obs_mask = (df["datetime"] >= obs_period[0]) & (df["datetime"] <= obs_period[1])
    eval_mask = (df["datetime"] >= eval_period[0]) & (df["datetime"] <= eval_period[1])
    return df[obs_mask], df[eval_mask]


@pytest.fixture(scope="module")
def scorer_fitted():
    s = RecencyFrequencyScorer()
    df_obs, df_eval = _split_by_period(_make_df(), _OBS_PERIOD, _EVAL_PERIOD)
    s.fit(
        df_obs,
        df_eval,
        recency_limit=_RECENCY_LIMIT,
        frequency_limit=_FREQUENCY_LIMIT,
    )
    return s
```

`_make_df()` の `datetime` 列は ISO 形式の文字列なので、文字列比較で正しく動作する（`"2024-01-01" <= "2024-01-07"` は字典順で正しい）。

#### `df_rec` フィクスチャ

Before:
```python
@pytest.fixture(scope="module")
def df_rec(scorer_fitted):
    return scorer_fitted.transform_date(_make_df(), _FIT_TARGET_DATE)
```

After:
```python
@pytest.fixture(scope="module")
def df_rec(scorer_fitted):
    df = _make_df()
    df_obs = df[df["datetime"] <= _FIT_TARGET_DATE]
    return scorer_fitted.transform(df_obs, ref=_FIT_TARGET_DATE)
```

### `TestNormalizeRef` / `TestNormalizeSequenceCol` の変更

これらは `RecencyFrequencyScorer._normalize_ref()` / `._normalize_sequence_col()` を呼んでいるため、モジュール関数に向けて書き換える。

#### Before
```python
class TestNormalizeRef:
    @pytest.fixture
    def scorer(self):
        return RecencyFrequencyScorer()

    def test_string_date(self, scorer):
        expected = pd.Timestamp("2024-01-01").toordinal()
        assert scorer._normalize_ref("2024-01-01") == expected
```

#### After（test_utils.py または test_scorer.py に残す）
```python
from rfscorer._time_utils import normalize_ref

class TestNormalizeRef:
    def test_string_date(self):
        expected = pd.Timestamp("2024-01-01").toordinal()
        assert normalize_ref("2024-01-01") == expected
```

判断: これらは時刻正規化のテストなので、新規 `tests/test_utils.py` に移動するのが筋。`tests/test_scorer.py` からは削除し、`tests/test_utils.py` に移す。

### `TestFitResult` 等の `scorer.fit()` 直接テストは変更不要

新 API そのものをテストしているため。

### `TestIntegerTimeCol` / `TestUnit`

これらは fit_date / fit_period も使っている可能性があるため、`fit()` + 手動分割ベースに修正。

## tests/test_utils.py 新規作成

### 配置構成

```python
# tests/test_utils.py
import pandas as pd
import pytest

from rfscorer.utils import split_by_date
from rfscorer._time_utils import normalize_ref, normalize_sequence_col


class TestNormalizeRef:
    # (test_scorer.py から移動)


class TestNormalizeSequenceCol:
    # (test_scorer.py から移動)


class TestSplitByDate:
    def test_basic_split_with_string_date(self):
        ...

    def test_basic_split_with_int_target(self):
        ...

    def test_observation_days_none_uses_df_start(self):
        ...

    def test_evaluation_days_none_uses_df_end(self):
        ...

    def test_observation_days_caps_at_df_start(self):
        ...

    def test_evaluation_days_caps_at_df_end(self):
        ...

    def test_target_date_inclusive_in_obs(self):
        ...

    def test_target_date_exclusive_in_eval(self):
        ...

    def test_returns_tuple_of_dataframes(self):
        ...

    def test_does_not_mutate_input_df(self):
        ...

    def test_preserves_original_columns(self):
        ...

    def test_preserves_original_time_col_type(self):
        ...

    def test_not_dataframe_raises(self):
        ...

    def test_missing_time_col_raises(self):
        ...

    def test_custom_time_col(self):
        ...

    def test_integer_time_col(self):
        ...

    def test_invalid_target_date_raises(self):
        ...

    def test_chained_with_fit(self):
        # split_by_date の戻り値をそのまま fit() に渡せること
        ...
```

## ドキュメントの変更詳細

### `docs/product-requirements.md`

#### 入力セクション
- `fit_date()` の3つの引数説明（L59-63）を削除
- `fit_period()` の3つの引数説明（L64-68）を削除
- 代わりに新規セクション「データ準備の補助」を追加し、`split_by_date()` の使い方を簡潔に説明

After のイメージ:
```markdown
- `fit()` に渡す引数（推奨）
  - `df_obs`: ...
  - `df_eval`: ...
  ...
- データ準備の補助
  - `from rfscorer import split_by_date`
  - `split_by_date(df, target_date, observation_days=28, evaluation_days=7)` で観測ログ・評価ログに分割可能
  - 日付・整数いずれも可
```

#### 機能テーブル
変更不要（emp/er/ef 等の機能行が中心であり、fit_date/fit_period は機能行に登場しない）。

#### コード例
変更不要（既に `df_obs = df[df["date"] <= ...]` 方式で書かれている）。

### `docs/functional-design.md`

#### メソッド仕様
- `##### fit_date(...)` セクション全体を削除
- `##### fit_period(...)` セクション全体を削除
- `##### transform_date(...)` セクション全体を削除
- 代わりに `##### split_by_date(df, target_date, observation_days=28, evaluation_days=7, time_col="datetime")` セクションを追加
  - ただし split_by_date は `RecencyFrequencyScorer` のメソッドではないため、配置場所は「クラス仕様」内ではなく独立した「ユーティリティ」セクションへ
- クラス仕様の冒頭に「データ準備のユーティリティは `rfscorer.utils.split_by_date()` を参照」のような誘導を追加

#### データフロー図
- 「`fit_date(df, target_date)` または `fit_period(df, ...)`」の枝を削除
- 代わりに「`split_by_date(df, target_date)` → `fit(df_obs, df_eval)`」のフローを追加

#### 入出力例
- `scorer.fit_date(df, target_date="2015-07-06")` の行を削除
- `scorer.transform_date(df, target_date="2015-07-06")` の行を削除
- `scorer.fit_period(...)` の行を削除
- 代わりに `df_obs, df_eval = split_by_date(df, "2015-07-06")` + `scorer.fit(df_obs, df_eval)` の例を追加

### `docs/glossary.md`

API テーブル（L43-48 周辺）から以下を削除：
- `fit_date(...)` エントリ
- `fit_period(...)` エントリ
- `transform_date(...)` エントリ

代わりに以下を追加：
- `split_by_date(df, target_date, observation_days=28, evaluation_days=7, time_col="datetime")` エントリ

主要 fit メソッドのエントリは「fit()」のみとなる。

各属性エントリ内の「`fit_date()` または `fit_period()` 後」のような言及を「`fit()` 後」に変更（メソッドが削除されるため）。

### `docs/development-guidelines.md` / `docs/architecture.md` / `docs/repository-structure.md`

`fit_date` / `fit_period` / `transform_date` への明示的言及があれば削除。`split_by_date` を追加する必要はおそらく無い（これらは設計ドキュメント全般）。

### `examples/basic_usage.ipynb`

確認: 現状の notebook は `fit()` を直接使っている。`fit_date` / `fit_period` / `transform_date` の使用なし。
→ 変更不要の見込み。grep で再確認する。

### `README.md`

`fit_date` / `fit_period` / `transform_date` の使用箇所がないか grep で確認。

### `CHANGELOG.md`

過去リリース記述は維持（履歴記録）。新規エントリは将来のリリース時に追加。

## 実装手順

### Phase 1: 時刻正規化モジュールの抽出
1. `src/rfscorer/_time_utils.py` を新規作成（`normalize_ref` / `normalize_sequence_col` / `_ORDINAL_ORIGIN`）
2. `src/rfscorer/scorer.py` の import を変更し、`self._normalize_*` / `self._ORDINAL_ORIGIN` を全て新関数呼び出しに置換
3. `RecencyFrequencyScorer` クラスから `_normalize_ref` / `_normalize_sequence_col` メソッド・`_ORDINAL_ORIGIN` 定数を削除
4. 中間チェック: `uv run pytest -k "not (FitDate or FitPeriod or TransformDate)"` でテストが通ること

### Phase 2: `split_by_date` の追加と公開
5. `src/rfscorer/utils.py` を新規作成し、`split_by_date()` を実装
6. `src/rfscorer/__init__.py` に `from .utils import split_by_date` を追加、`__all__` を更新

### Phase 3: fit_date / fit_period / transform_date の削除
7. `RecencyFrequencyScorer` から3メソッドを削除
8. 関連 docstring 言及を整理

### Phase 4: テストの更新
9. `tests/test_scorer.py` から `TestFitDateValidation` / `TestFitDateResult` / `TestFitPeriodValidation` / `TestFitPeriodResult` / `TestTransformDate` を削除
10. `scorer_fitted` 等のフィクスチャを `fit()` + manual split に書き換え
11. `TestNormalizeRef` / `TestNormalizeSequenceCol` を `tests/test_utils.py` に移動
12. `tests/test_utils.py` に `TestSplitByDate` を新規追加
13. `TestIntegerTimeCol` / `TestUnit` 等に残った fit_date/fit_period 利用を `fit()` ベースに書き換え

### Phase 5: ドキュメントの更新
14. `docs/product-requirements.md`
15. `docs/functional-design.md`
16. `docs/glossary.md`
17. `examples/basic_usage.ipynb` / `README.md` の grep 確認

### Phase 6: 品質チェック
18. `uv run ruff check .`
19. `uv run ruff format .`
20. `uv run pytest`

## リスクと対策

| リスク | 対策 |
|--------|------|
| フィクスチャ書き換え時の挙動差異 | 文字列日付の lex 比較は ISO 形式なら正しい。data の境界値テストで担保 |
| `_normalize_ref` / `_normalize_sequence_col` の API 変化に伴うテスト破綻 | モジュール関数として再公開し、テストの import を変更するだけで対応可能 |
| ドキュメント間の表記揺れ | Phase 5 で `grep -r "fit_date\|fit_period\|transform_date" docs/` で漏れ検出 |
| `transform_date` 削除に伴う既存ユーザーへの影響 | ユーザー数が少ない段階。次期メジャーリリースの release note で告知 |

## 完了確認用コマンド

```bash
# 削除対象が残っていないこと
grep -rn "fit_date\|fit_period\|transform_date" src/ tests/ docs/ examples/ README.md \
  | grep -v "^\.steering\|^CHANGELOG"
# → ZERO matches 期待

# 公開 API の確認
uv run python -c "from rfscorer import split_by_date; print(split_by_date.__doc__[:80])"

# kind alias 等の保護確認（前回タスクの assertion）
uv run pytest tests/test_scorer.py::TestKindAliases -v

# 全テスト
uv run pytest
```
