# Tasklist: view_recency の追加

## 実装タスク

### Phase 1: パラメータと内部定数の追加（プラミングのみ、挙動変更なし）

> 目標: `recency_mode` を受け取れるようにする。デフォルト `"day"` で、この時点では未使用。完了時点で既存テストは全パスのまま。

- [ ] **T01** クラス変数を追加する
  ```python
  _VALID_RECENCY_MODES = frozenset({"day", "view"})
  _VIEW_KEY_COL = "_view_key"  # view モードの高解像度時刻キー（内部列名）
  ```

- [ ] **T02** `__init__()` に `recency_mode: str = "day"` を追加する（`scorer.py` 50 行付近）
  - シグネチャ: `def __init__(self, ..., unit=1, recency_mode="day")`
  - バリデーション（`unit` チェックの近くに配置）:
    ```python
    if recency_mode not in self._VALID_RECENCY_MODES:
        raise ValueError(f"recency_mode must be 'day' or 'view', got {recency_mode!r}.")
    self.recency_mode = recency_mode
    ```

- [ ] **T03** Phase 1 の回帰確認: `uv run pytest -q`
  - 既存テストが全パスすること（新パラメータは未使用のため挙動不変）

### Phase 2: 高解像度時刻キーの追加（view モードのみ、design §2）

> 目標: 日単位 `_SEQUENCE_COL` を維持したまま、時刻を保持する `_VIEW_KEY_COL` を view モード時のみ併走させる。day モードでは一切生成しない。

- [ ] **T04** `_time_utils.py` に `normalize_view_key()` を追加する（design §2）
  - `normalize_sequence_col` と同じ origin / dtype 分岐。解像度のみ日→ns:
    datetime / 文字列 → `(series - _ORDINAL_ORIGIN) // pd.Timedelta("1ns")`
  - 整数・浮動小数 → `astype("int64")`（ユーザー指定粒度をそのまま）
  - 非対応 dtype → `ValueError`
  - 注: `datetime64.astype("int64")` は pandas 2.x で `TypeError` になる版があるため使わない

- [ ] **T05** `_to_internal()` に view モード時の `_VIEW_KEY_COL` 生成を追加する（`scorer.py` 265 行付近）
  - `normalize_sequence_col` による上書きの**直前**に、元の値から生成:
    ```python
    if self.recency_mode == "view":
        result[self._VIEW_KEY_COL] = normalize_view_key(result[self._SEQUENCE_COL])
    result[self._SEQUENCE_COL] = normalize_sequence_col(result[self._SEQUENCE_COL])
    ```

- [ ] **T06** `transform()` に同様の `_VIEW_KEY_COL` 生成を追加する（`scorer.py` 944 行付近）
  - `df_log[self._SEQUENCE_COL]` 上書きの直前に同パターンで挿入

- [ ] **T07** Phase 2 の回帰確認: `uv run pytest -q`
  - 既存テストが全パスすること（day モードでは列を追加しないため挙動不変）

### Phase 3: `_build_ui_rf_df` のリファクタリングと view recency の実装

> 目標: 「先に (user, item) 単位で集約 → `recency_mode` で分岐」の構造に変える。day モードの結果は現行と数値完全一致。

- [ ] **T08** `_build_ui_rf_df` を design §3「リファクタリング後」の最終形に置き換える（`scorer.py` 1596 行付近）
  - `df = df.reset_index(drop=True)` 後、`ts_col = self._VIEW_KEY_COL if view else self._SEQUENCE_COL`
  - `df.assign(_row_idx=df.index)` で `agg(last_ts=(ts_col,"max"), frequency=(SEQ,"count"), first_idx=("_row_idx","min"))` の**単一 agg**
  - `if day:` → `ui["recency"] = (ref_int - ui["last_ts"]) // self.unit + 1`
  - `elif view:` → `sort_values([USER,"last_ts","first_idx"], ascending=[True,False,True])` 後
    `groupby(USER, sort=False).cumcount() + 1`（`transform()` の order 計算と同パターン）
  - `else: raise ValueError(...)` の防御的ガード
  - 戻り値は現行と同じ `[USER, ITEM, recency, frequency]`（`last_ts`/`first_idx` は列選択で除外）

- [ ] **T09** `_build_ui_rf_df` の docstring を mode 対応に更新する
  - 現行 docstring は day 固定（`Recency is (ref_int - value) // unit + 1`）。
    day / view 両モードの recency 定義を記述し直す

- [ ] **T10** Phase 3 の回帰確認: `uv run pytest -q`
  - 既存テストが全パスすること（`recency_mode="day"` のリファクタが挙動を変えていないことの担保）

### Phase 4: 周辺箇所の更新

- [ ] **T11** `show()` の Model セクションに `recency_mode` を追記する（1506 行付近）
  ```python
  print(f"  recency_mode     : {self.recency_mode}")
  print(f"  recency_limit    : {self.recency_limit}")
  ```

- [ ] **T12** `save_zip()` の `metadata` dict に `recency_mode` を追加する（1317 行付近）
  ```python
  "unit": _to_python(self.unit),
  "recency_mode": self.recency_mode,   # 追加
  "recency_limit": _to_python(self.recency_limit),
  ```

- [ ] **T13** `__init__()` の docstring に `recency_mode` パラメータ説明を追加する
  ```
  recency_mode : str, default "day"
      How recency is computed for each (user, item) pair.
      "day"  : elapsed-days bin relative to ref: ``(ref - last_view) // unit + 1``.
      "view" : 1-indexed rank within each user ordered by most-recent view
               timestamp (1 = latest). Full timestamp resolution is used
               (sub-day order is preserved). ``ref`` and ``unit`` are ignored.
  ```

- [ ] **T14** `fit()` の `ref` パラメータ説明に注記を追加する
  - `recency_mode="view"` 時は `ref` が無視される旨を1行追記

- [ ] **T15** `transform()` の `ref` パラメータ説明に注記を追加する
  - `fit()` と同じ方針で1行追記

### Phase 5: テストの追加

- [ ] **T16** `tests/test_scorer.py` に `TestViewRecency` クラスを追加する

  - **test_example1_basic** — テスト例1（重複閲覧・順位付け）
    - 1ユーザー・3商品（A を2回閲覧）のログで `transform()` を実行
    - A=1, B=2, C=3 を確認

  - **test_example2_tiebreak** — テスト例2（timestamp 完全一致のタイブレーク）
    - A と B が**全く同一の timestamp**、C は別時刻
    - 入力データで先に出現した A=1, B=2, C=3 を確認

  - **test_example3_multi_user_intraday** — テスト例3（複数ユーザー＋同一日内の時刻順）
    - U1: P@17:00, Q@16:00（同日） / U2: P@16:00, Q@17:00（同日）
    - U1: P=1, Q=2 / **U2: Q=1, P=2**（時刻が新しい方が recency 小）を確認
    - **日単位切り捨てバグの回帰ガード**: 内部表現が日単位だと U2 が P=1,Q=2 になり失敗する

  - **test_intraday_resolution_datetime** — 同一日・異時刻が時刻順で並ぶこと（datetime 経路）
    - 1ユーザーで同日内に複数時刻 → timestamp 降順で recency が付与される

  - **test_integer_time_col_view** — 整数 time_col（連番）で view recency が値順に並ぶこと

  - **test_day_mode_unchanged** — `recency_mode="day"`（デフォルト）で既存挙動が変わらないこと
    - 同一データ・同一 `ref` で `recency_mode` 指定あり／なしの `transform()` 結果が一致

  - **test_fit_predict_transform_work** — `recency_mode="view"` で `fit()` / `predict()` / `transform()` が動作すること

  - **test_invalid_recency_mode_raises** — 未知の `recency_mode` で `ValueError` が送出されること

- [ ] **T17** テスト実行: `uv run pytest tests/test_scorer.py::TestViewRecency -v`
  - 全ケースパス

### Phase 6: ドキュメントの更新

- [ ] **T18** `docs/functional-design.md` を更新する
  - `recency_mode` パラメータの説明と対応表（day / view）を追記
  - view モードが timestamp の完全解像度を用いること（高解像度キー併走）を注記
  - 将来の拡張予定（`recency_mode="session"`, `frequency_mode`）を注記

### Phase 7: 品質チェック

- [ ] **T19** リントを実行する: `uv run ruff check .`

- [ ] **T20** フォーマットを確認する: `uv run ruff format --check .`

- [ ] **T21** 全テストを実行する: `uv run pytest`
  - 既存テスト＋ `TestViewRecency` がすべて green

## 完了条件

すべてのタスクが完了し、以下が成立していること:

1. `uv run ruff check .` が `All checks passed!`
2. `uv run ruff format --check .` が `all files already formatted`
3. `uv run pytest` がすべて green（既存テスト＋ `TestViewRecency`）
4. `recency_mode="view"` で view recency が正しく計算される（テスト例 1〜3 一致）
5. 同一日内の異なる時刻が timestamp 順で区別される（高解像度キーが効いている）
6. `recency_mode="day"`（デフォルト）で既存挙動が変わらない
7. `fit()` / `predict()` / `transform()` が `recency_mode="view"` でも動作する
8. 未知の `recency_mode` で `ValueError` が送出される
9. `show()` に `recency_mode` が表示される
10. `save_zip()` の `metadata.json` に `recency_mode` が含まれる
