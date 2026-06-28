# Design: コンストラクタの軸対称化（`unit` → `recency_unit`）

## 変更ファイル一覧

| ファイル | 変更種別 | 内容 |
|----------|----------|------|
| `src/rfscorer/scorer.py` | 修正 | `__init__` 署名・バリデーション・`self.recency_unit`・docstring、`fit`/`_build_ui_rf_df` docstring、metadata、ディスパッチャ |
| `src/rfscorer/_recency.py` | 修正 | `build_day_rf` の引数 `unit`→`recency_unit`・docstring |
| `tests/test_scorer.py` | 修正 | `TestUnit` の `unit=`→`recency_unit=`、エラーメッセージ `match` |
| `tests/test_recency.py` | 修正 | `build_day_rf` 呼び出しのコメントのみ（位置引数なので実コード変更不要） |
| `docs/functional-design.md` | 修正 | コンストラクタ表・`recency_mode` 補足・metadata・式中の `unit` |
| `docs/glossary.md` | 修正 | day recency 式・API 表 |
| `docs/product-requirements.md` | 修正 | コンストラクタ入力 |

> **方針**: `unit` を `recency_unit` にハード改名し、`recency_mode` を `recency_unit` の前に並べ替える。
> 非推奨エイリアスは設けない。既定値（`recency_mode="day"`, `recency_unit=1`）で現行と数値完全一致。

## 実装アプローチ

### 1. `__init__` 署名・バリデーション・属性

**変更前:**
```python
def __init__(self, user_col="user", item_col="item", time_col="datetime", unit=1, recency_mode="day"):
    ...
    if unit <= 0:
        raise ValueError(f"unit must be a positive integer, got {unit}.")
    if recency_mode not in self._VALID_RECENCY_MODES:
        raise ValueError(f"recency_mode must be 'day' or 'view', got {recency_mode!r}.")
    ...
    self.unit = unit
```

**変更後:**
```python
def __init__(self, user_col="user", item_col="item", time_col="datetime", recency_mode="day", recency_unit=1):
    ...
    if recency_mode not in self._VALID_RECENCY_MODES:
        raise ValueError(f"recency_mode must be 'day' or 'view', got {recency_mode!r}.")
    if recency_unit <= 0:
        raise ValueError(f"recency_unit must be a positive integer, got {recency_unit}.")
    ...
    self.recency_unit = recency_unit
```

- 引数順: `... time_col, recency_mode, recency_unit`（`recency_mode` を前に）。
- バリデーション順も `recency_mode` → `recency_unit` に揃える（任意だが署名順と一致させ可読性向上）。
- `self.unit` は廃止し `self.recency_unit` のみとする（エイリアス属性も設けない）。

### 2. `__init__` docstring

`unit` パラメータ説明を `recency_unit` に置換し、`recency_mode` を先に記述する（署名順と一致）。

```
recency_mode : str, default "day"
    How recency is computed for each (user, item) pair.
    "day"  : elapsed-days bin relative to ref: ``(ref - last_view) // recency_unit + 1``.
    "view" : 1-indexed rank within each user ordered by most-recent view
             timestamp (1 = latest). Full timestamp resolution is used
             (sub-day order is preserved). ``ref`` and ``recency_unit`` are ignored.
recency_unit : int, default 1
    Number of days (or integer steps) per recency bin (recency-axis bin width).
    Recency is ``(ref - last_view) // recency_unit + 1``. Must be a positive
    integer. Use ``recency_unit=7`` for weekly, ``recency_unit=30`` for approximate
    monthly granularity. Ignored when ``recency_mode="view"``.
```

> **将来メモ（docstring 末尾の Notes か補足に1行）**: frequency 軸も同様に
> `frequency_mode`（既定 `"view"` = 閲覧イベント数）/ `frequency_unit` を将来追加予定。
> 既定が非対称（recency=`"day"` / frequency=`"view"`）なのは、古典的 RF で recency は時間ベース・
> frequency は回数ベースであることに由来する。

### 3. `_recency.build_day_rf` の引数改名

**変更前:**
```python
def build_day_rf(df, user_col, item_col, seq_col, ref_int, unit):
    ...
    ui["recency"] = (ref_int - ui["last_ts"]) // unit + 1
```

**変更後:**
```python
def build_day_rf(df, user_col, item_col, seq_col, ref_int, recency_unit):
    ...
    ui["recency"] = (ref_int - ui["last_ts"]) // recency_unit + 1
```

- モジュール docstring（`_recency.py` 冒頭）の `// unit + 1` も `// recency_unit + 1` に統一。
- `build_view_rf` は `recency_unit` を使わないため変更なし。

### 4. ディスパッチャ `_build_ui_rf_df`

**変更前:**
```python
return build_day_rf(df, self._USER_COL, self._ITEM_COL, self._SEQUENCE_COL, ref_int, self.unit)
```

**変更後:**
```python
return build_day_rf(df, self._USER_COL, self._ITEM_COL, self._SEQUENCE_COL, ref_int, self.recency_unit)
```

- `_build_ui_rf_df` docstring の `// unit + 1` 記述も `recency_unit` に統一。

### 5. `save_zip()` metadata

**変更前:**
```python
"unit": _to_python(self.unit),
```

**変更後:**
```python
"recency_unit": _to_python(self.recency_unit),
```

- metadata は人間可読 JSON のみ。pickle 本体は `self` ごと保存するため `load`/`load_zip` は自動対応。
- metadata スキーマの破壊的変更だが、pre-1.0 かつ本タスクの方針（ハード改名）に沿う。

### 6. `fit()` docstring

`ref` 説明中の `(ref - value) // unit + 1` を `(ref - value) // recency_unit + 1` に置換。

### 7. テスト更新（`tests/test_scorer.py`）

`TestUnit` クラス:
- `RecencyFrequencyScorer(unit=unit)` → `RecencyFrequencyScorer(recency_unit=unit)`
- `RecencyFrequencyScorer(unit=0)` / `unit=-1` → `recency_unit=0` / `recency_unit=-1`
- `pytest.raises(ValueError, match="unit must be a positive integer")` →
  `match="recency_unit must be a positive integer"`
- クラス名・ヘルパー名（`TestUnit` / `_make_scorer_with_unit`）はリネーム任意。
  本タスクでは**実害がないため最小変更**とし、内部の `unit` 変数名・コメントのみ調整（クラス名は `TestRecencyUnit`、ヘルパーは `_make_scorer_with_recency_unit` に改名して意図を明確化することを推奨）。

`tests/test_recency.py`:
- `build_day_rf(df, USER, ITEM, SEQ, 110, 7)` 等は**位置引数**で呼んでおり、引数名改名の影響を受けない。
  コメント（`# unit=7: ...`）を `recency_unit` に整える程度。

### 8. ドキュメント更新

- `docs/functional-design.md`:
  - コンストラクタ署名行・パラメータ表（`unit` 行 → `recency_unit` 行、`recency_mode` を前に）
  - `recency_mode` 補足・式 `(ref - last_view) // unit + 1` → `recency_unit`
  - metadata 一覧の `unit` → `recency_unit`
  - `ref` 行・将来メモ（`frequency_unit` 既定方針）
- `docs/glossary.md`: day recency 式 `(ref - last_view) // unit + 1` → `recency_unit`、API 表の「粒度 `unit`」→「`recency_unit`」
- `docs/product-requirements.md`: コンストラクタ入力の `unit` 箇条 → `recency_unit`

## 影響範囲まとめ

| 対象 | 変更内容 | 種別 |
|------|----------|------|
| `__init__` | `unit`→`recency_unit` 改名・並べ替え・バリデーション・docstring | 破壊的変更 |
| `self.unit` 参照 | `self.recency_unit` に置換（init / dispatcher） | リファクタ |
| `build_day_rf` | 引数 `unit`→`recency_unit` | リファクタ |
| `save_zip` metadata | キー `"unit"`→`"recency_unit"` | 破壊的変更（metadata スキーマ） |
| `fit`/`_build_ui_rf_df`/`_recency` docstring | `unit`→`recency_unit` | ドキュメント |
| `save`/`load`/`load_zip` | 変更なし（pickle 自動対応） | — |
| テスト | `unit=`→`recency_unit=`・`match` 文言 | テスト更新 |
| docs 3ファイル | `unit`→`recency_unit`・将来メモ | ドキュメント |

## 非対象（将来タスク）

- `frequency_mode` / `frequency_unit` の追加（既定 `frequency_mode="view"`）
- `recency_unit` の view ランクへの適用
- 新パラメータのキーワード専用化
- 旧 pickle/metadata（`unit` 名）との後方互換（pre-1.0・ハード改名のため非対応）
