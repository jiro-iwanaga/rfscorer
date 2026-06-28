# Design: view_recency の追加

## 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
|----------|----------|------|
| `src/rfscorer/_recency.py` | 追加 | `build_day_rf()` / `build_view_rf()`（recency/frequency ビルダーの純粋関数。scorer 非依存） |
| `src/rfscorer/_time_utils.py` | 追加 | `normalize_view_key()`（時刻を保持する高解像度キー生成） |
| `src/rfscorer/scorer.py` | 修正 | `__init__`, `_to_internal`, `transform`, `_build_ui_rf_df`（ディスパッチャ化＋docstring）, `show`, `save_zip`, `fit`/`transform` docstring |
| `tests/test_scorer.py` | 修正 | view recency のテスト追加 |
| `tests/test_recency.py` | 追加 | `_recency.py` ビルダーの単体テスト |
| `docs/functional-design.md` | 修正 | `recency_mode` パラメータの記述追加 |

> **設計の核**: recency/frequency の計算ロジックを `_recency.py` の純粋関数（`build_day_rf` /
> `build_view_rf`）に**外出し**し、`_build_ui_rf_df` は `recency_mode` で振り分ける薄い
> ディスパッチャにする。データ供給は高解像度キー（K）方式で、`fit_rolling` の日窓フィルタと
> 両立する。day/view とも独立した小関数なので単体テスト・保守が容易で、将来の
> `recency_mode="session"` / `frequency_mode` も同じ場所への追加で済む。

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
- `_VALID_RECENCY_MODES` はクラス変数として定義し、`__init__` のバリデーションで使う。`_build_ui_rf_df`（ディスパッチャ）の `else: raise` は防御的ガード（`__init__` で弾くため通常到達しない）。
- `recency_mode` は `__init__` で常に設定される。本変更**以前**に保存した pickle には属性が無いため、旧モデルを新コードで `show()`/`transform()` 等に通すと `AttributeError` になりうる（pickle は版間非互換であり、既存の追加属性 `fit_method_` 等と同じ既知の制約。本タスクのスコープ外）。

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
- この列は行フィルタ（`fit_rolling` の窓抽出）でも保持され、`build_view_rf` まで到達する。
- `build_view_rf` が `[user, item, recency, frequency]` を明示選択して返すため、`_VIEW_KEY_COL` は最終出力に漏れない。

### 3. recency/frequency ビルダーの外出し（`_recency.py`）と `_build_ui_rf_df` のディスパッチャ化

現行の `_build_ui_rf_df` はイベント行ごとに recency を計算してから groupby min を取っている。
これを **`_recency.py` の純粋関数に外出し**し、`_build_ui_rf_df` は `recency_mode` で
振り分けるだけの薄いディスパッチャにする。day/view が独立した小関数になり、単体テストと
将来拡張（`session` / `frequency_mode`）が容易になる。

#### `_recency.py`（新規・純粋関数）

```python
"""(user, item) ごとの recency / frequency を構築するビルダー群（scorer 非依存）。

day モードは日単位序数 seq_col を、view モードは高解像度キー key_col を recency 源に使う。
frequency は両モードとも seq_col の件数（view freq）。いずれも列 user, item, recency,
frequency を返す。
"""

def build_day_rf(df, user_col, item_col, seq_col, ref_int, unit):
    ui = (
        df.groupby([user_col, item_col], sort=False)[seq_col]
        .agg(last_ts="max", frequency="count")
        .reset_index()
    )
    ui["recency"] = (ref_int - ui["last_ts"]) // unit + 1
    return ui[[user_col, item_col, "recency", "frequency"]]


def build_view_rf(df, user_col, item_col, key_col, seq_col):
    df = df.reset_index(drop=True)
    ui = (
        df.assign(_row_idx=df.index)
        .groupby([user_col, item_col], sort=False)
        .agg(
            last_key=(key_col, "max"),       # 最新閲覧（高解像度）
            frequency=(seq_col, "count"),    # view freq（件数）
            first_idx=("_row_idx", "min"),   # 同値タイブレーク用（初出行）
        )
        .reset_index()
    )
    # user 内で「最新閲覧 降順, 初出行 昇順」に並べ、1 起算ランクを付与
    ui = ui.sort_values([user_col, "last_key", "first_idx"], ascending=[True, False, True])
    ui["recency"] = ui.groupby(user_col, sort=False).cumcount() + 1
    return ui[[user_col, item_col, "recency", "frequency"]]
```

#### ディスパッチャ（`scorer._build_ui_rf_df`）

```python
def _build_ui_rf_df(self, df, ref_int):
    if self.recency_mode == "view":
        return build_view_rf(
            df, self._USER_COL, self._ITEM_COL, self._VIEW_KEY_COL, self._SEQUENCE_COL
        )
    if self.recency_mode == "day":
        return build_day_rf(
            df, self._USER_COL, self._ITEM_COL, self._SEQUENCE_COL, ref_int, self.unit
        )
    raise ValueError(f"recency_mode must be 'day' or 'view', got {self.recency_mode!r}.")
```

#### 等価性（day モード）

現行は `min((ref - ts) // unit + 1)` を groupby で計算、`build_day_rf` は
`(ref - max(ts)) // unit + 1` を直接計算する。`(ref - ts) // unit` は ts が大きいほど小さく
なるため `min = max(ts)` 対応の値となり、数学的に等価。frequency も両者とも行数カウントで一致。
よって day モードの数値・既存テストは完全保存。

#### view モードの行順について

`build_view_rf` の戻り値は user 昇順になるが、下流（`_aggregate_empirical` の recency/frequency
集約、`transform()` の再ソート）は行順に依存しないため影響しない。

#### `fit_rolling` との両立（最頻メソッド優先）

`fit_rolling` の観測窓・正解窓フィルタは**日単位序数 `_SEQUENCE_COL` のまま**（`observation_days`
は日単位なので当然）。窓フィルタはブールマスクなので `_VIEW_KEY_COL` 列も保持され、各窓の
`build_view_rf` にフル解像度キーが届く。view recency は**窓ごとに 1 から振り直し**（ローリングの
正しい意味論）。`_to_internal` のキー生成は O(n) のベクトル演算1回のみで、全件ソートは発生しない。

#### 将来 `frequency_mode` を追加するときのパターン（本タスクでは実装しない）

frequency は `recency_mode` および高解像度キーと**直交**する。day freq は
**日序数 `seq_col` の `nunique`（閲覧日数）**であり、日序数は両モードで常に存在するため
高解像度キーは不要。ビルダー内の frequency 集約エントリを切り替えるだけでよく、**groupby を
増やさず単一集約のまま**追加できる。

```python
# build_*_rf 内の frequency 集約をモードで切り替えるイメージ
#   frequency_mode="view": ("<count対象列>", "count")   # 現行
#   frequency_mode="day" : (seq_col, "nunique")          # 閲覧日数
```

> **将来メモ（本タスクでは実装しない）**: ビン幅を軸ごとに指定できるよう、`unit` を将来
> `recency_unit` / `frequency_unit` に分割する案。frequency のビン化はビルダー末尾の純粋後処理
> `frequency = (frequency - 1) // frequency_unit + 1` で足せ、`recency_mode` / `frequency_mode`
> と直交する。今回は構造（frequency を後処理を挟める形で返す）だけ確保し、パラメータは追加しない。

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

**`_build_ui_rf_df()`**: 現行 docstring は day 固定（`Recency is (ref_int - value) // unit + 1`）。`recency_mode` に応じて `_recency.py` の `build_day_rf` / `build_view_rf` に委譲する旨へ書き換える（各モードの定義はビルダー側の docstring に記述）。

**`fit()`**: `ref` パラメータの説明に `recency_mode="view"` 時は無視される旨を追記。

**`transform()`**: `ref` パラメータの説明に同様の注記を追記。

## 影響範囲まとめ

| 対象 | 変更内容 | 変更種別 |
|------|----------|----------|
| `_recency.py`（新規） | `build_day_rf()` / `build_view_rf()` ビルダーを外出し | 機能追加 |
| `_time_utils.normalize_view_key()` | 高解像度時刻キー生成関数を追加 | 機能追加 |
| `__init__` | `recency_mode` パラメータ追加・バリデーション・`_VIEW_KEY_COL` 定数 | 機能追加 |
| `_to_internal()` | view モード時に `_VIEW_KEY_COL` 列を併走生成 | 機能追加 |
| `transform()` | view モード時に `_VIEW_KEY_COL` 列を併走生成 ＋ docstring | 機能追加 |
| `_build_ui_rf_df` | `_recency.py` ビルダーへのディスパッチャ化 ＋ docstring | リファクタリング |
| `fit()` | docstring のみ | ドキュメント |
| `show()` | `recency_mode` 表示追加 | 機能追加 |
| `save_zip()` | metadata.json に `recency_mode` 追加 | 機能追加 |
| `save()` / `load()` / `load_zip()` | 変更なし（pickle 復元で自動対応） | — |
| テスト | `_recency.py` の単体テスト＋ view recency のテスト追加 | テスト追加 |
| `docs/functional-design.md` | `recency_mode` の記述追加 | ドキュメント |
