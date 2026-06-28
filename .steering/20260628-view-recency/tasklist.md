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

### Phase 3: recency/frequency ビルダーの外出しと `_build_ui_rf_df` のディスパッチャ化

> 目標: recency/frequency の計算を `_recency.py` の純粋関数に外出しし、`_build_ui_rf_df` を薄いディスパッチャにする。day モードの結果は現行と数値完全一致。

- [ ] **T08** `src/rfscorer/_recency.py` を新規作成する（design §3）
  - `build_day_rf(df, user_col, item_col, seq_col, ref_int, unit)`:
    `groupby[seq_col].agg(last_ts="max", frequency="count")` 後 `recency = (ref_int - last_ts)//unit + 1`
  - `build_view_rf(df, user_col, item_col, key_col, seq_col)`:
    `df.reset_index(drop=True)` → `assign(_row_idx)` →
    `agg(last_key=(key_col,"max"), frequency=(seq_col,"count"), first_idx=("_row_idx","min"))` →
    `sort_values([user,"last_key","first_idx"], ascending=[True,False,True])` →
    `groupby(user, sort=False).cumcount() + 1`
  - 両者とも戻り値は `[user_col, item_col, "recency", "frequency"]`
  - frequency 集約は将来 `frequency_mode` で差し替えやすいよう1エントリに局所化（コメント明記）

- [ ] **T09** `_build_ui_rf_df` をディスパッチャに置き換える（`scorer.py` 1596 行付近）
  - `from ._recency import build_day_rf, build_view_rf`
  - view → `build_view_rf(df, USER, ITEM, _VIEW_KEY_COL, _SEQUENCE_COL)`
  - day → `build_day_rf(df, USER, ITEM, _SEQUENCE_COL, ref_int, unit)`
  - それ以外 → `ValueError`
  - docstring を「`recency_mode` に応じて `_recency.py` のビルダーへ委譲する」旨に更新

- [ ] **T10** Phase 3 の回帰確認: `uv run pytest -q`
  - 既存テストが全パスすること（`recency_mode="day"` のディスパッチャ化が挙動を変えていないことの担保）

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

- [ ] **T16** `tests/test_recency.py` を新規作成し、`build_view_rf` / `build_day_rf` の単体テストを追加する
  （純粋関数なので scorer の準備不要で例1〜3を直接検証する）

  - **test_view_example1_basic** — テスト例1（重複閲覧・順位付け）: A=1, B=2, C=3
  - **test_view_example2_tiebreak** — テスト例2（高解像度キー完全一致のタイブレーク）:
    A と B が同一キー、初出順で A=1, B=2, C=3
  - **test_view_example3_multi_user_intraday** — テスト例3（複数ユーザー＋同一日内の時刻順）:
    U1: P=1,Q=2 / **U2: Q=1,P=2**（新しい時刻が recency 小）。
    **日単位切り捨てバグの回帰ガード**（key_col に ns 解像度を与えて検証）
  - **test_view_integer_key** — 整数キーで値順に並ぶこと
  - **test_day_builder_matches_legacy** — `build_day_rf` が現行 `_build_ui_rf_df` の式と一致

- [ ] **T17** `tests/test_scorer.py` に `TestViewRecency` クラスを追加する（統合レベル）

  - **test_intraday_resolution_datetime** — 同日・異時刻が timestamp 降順で並ぶこと（`transform()` 経由、datetime 経路で高解像度キー併走が効いている）
  - **test_integer_time_col_view** — 整数 time_col で view recency が値順に並ぶこと
  - **test_day_mode_unchanged** — `recency_mode="day"`（デフォルト）で `transform()` 結果が `recency_mode` 指定なしと一致
  - **test_fit_predict_transform_work** — `recency_mode="view"` で `fit()` / `predict()` / `transform()` が動作
  - **test_fit_rolling_view** — `recency_mode="view"` で `fit_rolling()` が動作し、各窓で view recency が計算される
  - **test_invalid_recency_mode_raises** — 未知の `recency_mode` で `ValueError`
  - **test_show_and_save_zip_recency_mode** — `show()` 出力と `save_zip()` の metadata に `recency_mode` が含まれる

- [ ] **T18** テスト実行: `uv run pytest tests/test_recency.py tests/test_scorer.py::TestViewRecency -v`
  - 全ケースパス

### Phase 6: ドキュメントの更新

- [ ] **T19** `docs/functional-design.md` を更新する
  - `recency_mode` パラメータの説明と対応表（day / view）を追記
  - view モードが timestamp の完全解像度を用いること（高解像度キー併走）を注記
  - recency/frequency ビルダーを `_recency.py` に外出しした構成を反映
  - 将来の拡張予定（`recency_mode="session"`, `frequency_mode="day"` = 日序数の nunique・直交）を注記

### Phase 7: 品質チェック

- [ ] **T20** リントを実行する: `uv run ruff check .`

- [ ] **T21** フォーマットを確認する: `uv run ruff format --check .`

- [ ] **T22** 全テストを実行する: `uv run pytest`
  - 既存テスト＋ `test_recency.py` ＋ `TestViewRecency` がすべて green

## 完了条件

すべてのタスクが完了し、以下が成立していること:

1. `uv run ruff check .` が `All checks passed!`
2. `uv run ruff format --check .` が `all files already formatted`
3. `uv run pytest` がすべて green（既存テスト＋ `test_recency.py` ＋ `TestViewRecency`）
4. recency/frequency 計算が `_recency.py`（`build_day_rf` / `build_view_rf`）に外出しされ、`_build_ui_rf_df` はディスパッチャになっている
5. `recency_mode="view"` で view recency が正しく計算される（テスト例 1〜3 一致）
6. 同一日内の異なる時刻が timestamp 順で区別される（高解像度キーが効いている）
7. `recency_mode="day"`（デフォルト）で既存挙動が変わらない（数値完全一致）
8. `fit()` / `predict()` / `transform()` / `fit_rolling()` が `recency_mode="view"` でも動作する
9. 未知の `recency_mode` で `ValueError` が送出される
10. `show()` に `recency_mode` が表示され、`save_zip()` の `metadata.json` に `recency_mode` が含まれる
