# Tasklist: `fit_rolling()` ローリング集計

## 実装タスク

### Phase 1: `_fit_impl` のリファクタ（集計部の共有化）

> 目標: `fit()` の挙動を一切変えずに、Part A / Part B をヘルパーに抽出する。完了時点で既存テストは全パスのまま。

- [x] **T01** `_to_internal()` に `time_col` 任意引数を追加する（`scorer.py` 241–250 行）
  - シグネチャを `_to_internal(self, df, time_col=None)` に変更
  - `tc = time_col if time_col is not None else self.time_col` を用いて列抽出
  - 既存呼び出し（`fit()` 内 218 行）は引数省略のまま動作（非破壊）

- [x] **T02** `_build_ui_rf_cv(self, obs_log, gt_log, ref_int)` を新規追加する
  - 現行 `_fit_impl` の 257–264 行（`UIcv` 構築・`_build_ui_rf_df`・cv 付与）を移設
  - 戻り値: `df_ui2frc`（列: `user`, `item`, `recency`, `frequency`, `cv`）

- [x] **T03** `_aggregate_empirical(self, df_ui2frc, recency_limit, frequency_limit)` を新規追加する
  - 現行 `_fit_impl` の 265–416 行（`record_num_target_org` 〜 相関スライス）を移設
  - `obs_log` / `gt_log` / `ref_int` に非依存であることを確認しながら移設

- [x] **T04** `_fit_impl()` を改修後の形に置き換える
  ```python
  def _fit_impl(self, obs_log, gt_log, ref_int, recency_limit, frequency_limit):
      self.record_num_obs = len(obs_log)
      self.record_num_gt = len(gt_log)
      df_ui2frc = self._build_ui_rf_cv(obs_log, gt_log, ref_int)
      self._aggregate_empirical(df_ui2frc, recency_limit, frequency_limit)
  ```

- [x] **T05** Phase 1 の回帰確認: `uv run pytest -q`
  - 既存テストが全パスすること（リファクタが挙動を変えていないことの担保）

### Phase 2: `fit_rolling()` の実装

- [x] **T06** `fit_rolling()` メソッドを追加する（`fit()` / `_fit_impl()` の後、`transform()` の前）
  - シグネチャ: `fit_rolling(self, df_obs, df_gt, observation_days, gt_days, roll_days=1, end_date=None, recency_limit=None, frequency_limit=None, time_col=None)`
  - docstring（引数表・anchor 算出・バリデーション・戻り値 `self`）

- [x] **T07** 入力検証を実装する（design 4.2）
  - `df_obs` / `df_gt` が `pd.DataFrame`（でなければ `TypeError`）
  - `tc = time_col if time_col is not None else self.time_col`
  - `df_obs` に `[user_col, item_col, tc]`、`df_gt` に `[user_col, item_col, tc]` 存在（欠落で `ValueError`）。**df_gt の tc 欠落も検出**
  - `observation_days >= 1`, `gt_days >= 1`, `roll_days >= 1`（でなければ `ValueError`）

- [x] **T08** 内部化と境界量算出を実装する（design 4.3）
  - `obs_internal = self._to_internal(df_obs, time_col=tc)` / `gt_internal = self._to_internal(df_gt, time_col=tc)`
  - `obs_min = int(obs_seq.min())` / `gt_max = int(gt_seq.max())`
  - `end_int = gt_max if end_date is None else normalize_ref(end_date)`
  - `anchor = end_int - gt_days`

- [x] **T09** 集計開始前バリデーション（fail-fast）を実装する（design 4.4 / AC-5・AC-6）
  - AC-6: `end_date is not None and end_int > gt_max` → `ValueError`
  - AC-5: `oldest_obs_start = anchor - (roll_days-1) - observation_days + 1 < obs_min` → `ValueError`
    - `max_roll_days = anchor - obs_min - observation_days + 2`
    - `max_roll_days < 1` の場合は専用メッセージ（observation_days/gt_days 過大）
    - それ以外は最大 `roll_days` を含むメッセージ
  - ウィンドウフィルタ・集計の前に実施

- [x] **T10** ローリング集計ループを実装する（design 4.5）
  - `for k in range(roll_days): td = anchor - k`
  - `obs_w` = `obs_internal` を `[td-observation_days+1, td]` でフィルタ
  - `gt_w` = `gt_internal` を `[td+1, td+gt_days]` でフィルタ
  - `total_obs_rows` / `total_gt_rows` を加算
  - `frames.append(self._build_ui_rf_cv(obs_w, gt_w, td)[["recency", "frequency", "cv"]])` ← 3 列に絞る（design 4.5 / H4 列スリム化）
  - `combined = pd.concat(frames, ignore_index=True)`

- [x] **T11** stats 設定と集計部呼び出しを実装する（design 4.6）
  - `record_num_obs` / `record_num_gt` / `record_num` = 全ロール合算
  - `observation_end_ = anchor`、`observation_start_ = max(obs_min, oldest_obs_start)`
  - `self._aggregate_empirical(combined, recency_limit, frequency_limit)`
  - `return self`

- [x] **T12** 動作確認: `uv run python -c "..."` で `fit_rolling` を最小データで呼び、`emp_probability_` が生成されることを確認

### Phase 3: テストの追加

- [x] **T13** `tests/test_scorer.py` に `TestFitRolling` を追加する（design 8）
  - `test_roll_days_1_equiv_manual_fit` — `roll_days=1` が手動フィルタ+`fit` と `emp_probability_` 一致（AC-9）
  - `test_rolling_accumulates_N` — `roll_days` 増で `N` 合計増加（AC-4）
  - `test_gt_distinct_event` — `df_gt` が別 user/item 集合でも cv 判定が正しい
  - `test_end_date_default_uses_gt_max` — `end_date=None` で `observation_end_ == gt_max - gt_days`（AC-2）
  - `test_end_date_explicit` — `end_date` 明示が `anchor` に反映（AC-2）
  - `test_roll_days_too_large_raises` — 境界突破で `ValueError`、メッセージに最大 `roll_days`（AC-5）
  - `test_end_date_beyond_gt_max_raises` — `end_date > gt_max` で `ValueError`（AC-6）
  - `test_integer_time_col` — 整数 time_col で動作
  - `test_datetime_time_col` — 日付 time_col で動作
  - `test_time_col_override` — `time_col` 引数で列名上書き（AC-1）
  - `test_same_log_df_df` — `fit_rolling(df, df, ...)`（再閲覧）で動作
  - `test_downstream_after_rolling` — 後続 `predict`/`transform`/`optimize`/`show` 動作（AC-7）
  - `test_df_gt_missing_time_col_raises` — `df_gt` に time_col 無で `ValueError`

- [x] **T14** テスト実行: `uv run pytest tests/test_scorer.py::TestFitRolling -v`
  - 全ケースパス

### Phase 4: ドキュメントの更新

- [x] **T15** `docs/functional-design.md` を更新する（AC-11）
  - 「クラス仕様 > メソッド」に `fit_rolling(...)` 節（引数表・挙動・anchor 算出・バリデーション）を追加
  - データフロー図に `df_obs`/`df_gt` →（ローリング窓フィルタ）→ `fit_rolling` → 経験的確率 の経路を追記

- [x] **T16** `docs/glossary.md` を更新する（AC-11）
  - API 簡潔版テーブルに `fit_rolling(df_obs, df_gt, observation_days, gt_days, roll_days, end_date)` 行を追加
  - 「ローリング集計」用語を「アルゴリズム」節に追加（任意）

- [x] **T17** `README.md` / `examples/` を確認する
  - `fit` の言及周辺に `fit_rolling` の簡単な紹介を追記（該当あれば）

### Phase 5: 品質チェック

- [x] **T18** リントを実行する（`uv run ruff check .`）

- [x] **T19** フォーマットを実行する（`uv run ruff format .`）

- [x] **T20** 全テストを実行する（`uv run pytest`）

### Phase 6: 統計属性の仕様確定（design 12）

- [x] **T21** `__init__` に新属性を `None` 初期化で追加する
  - 物理: `n_obs_rows_`, `n_gt_events_`, `n_users_`, `n_items_`
  - 構成: `fit_method_`, `roll_days_`, `observation_days_`, `gt_days_`

- [x] **T22** `_set_dataset_stats(self, obs_df, gt_df)` ヘルパーを追加する（design 12.5）

- [x] **T23** `_fit_impl()`（fit 経路）で構成・物理件数を設定する
  - `fit_method_="fit"`, `roll_days_=1`, `observation_days_=None`, `gt_days_=None`
  - `self._set_dataset_stats(obs_log, gt_log)`

- [x] **T24** `fit_rolling()` 本体で構成・物理件数を設定する
  - `fit_method_="fit_rolling"`, `roll_days_`/`observation_days_`/`gt_days_`
  - 観測和集合 `[observation_start_, anchor]`・正解和集合 `[anchor-roll_days+2, end_int]` でフィルタし `_set_dataset_stats(...)`

- [x] **T25** `show()` の `Data` セクションを物理／延べに分離し `fit_method_` で分岐する（design 12.6）

- [x] **T26** `save_zip` の `metadata.json` に新キーを追加する（design 12.7）

- [x] **T27** `tests/test_scorer.py` に統計属性テストを追加する（design 12.8）

- [x] **T28** `docs/functional-design.md` の属性表・metadata 記述を更新する（design 12.9）

- [x] **T29** 品質チェック: `uv run ruff check . && uv run ruff format . && uv run pytest`

## 完了条件

すべてのタスクが完了し、以下が成立していること：

1. `uv run ruff check .` が `All checks passed!`
2. `uv run ruff format --check .` が `all files already formatted`
3. `uv run pytest` がすべて green（既存テスト＋ `TestFitRolling`）
4. `RecencyFrequencyScorer.fit_rolling()` が design 4.1 のシグネチャで利用可能
5. `roll_days=1` で手動フィルタ + `fit` と `emp_probability_` 一致（AC-9）
6. 境界バリデーションが集計開始前に fail-fast で動作（AC-5 / AC-6）
7. `fit_rolling()` 後に `predict` / `transform` / `optimize` / `show` が動作
8. `docs/functional-design.md` / `docs/glossary.md` が `fit_rolling()` に整合
9. 既存 `fit()` の挙動・既存テストが完全保存（非破壊）

## 将来の高速化（本タスク対象外・design 11 参照）

初手は可読性優先。以下は `_build_ui_rf_cv()` / `_aggregate_empirical()` の集計シーム内で完結し、`fit_rolling()` 本体・公開 API・テスト契約を変えずに差し替え可能。効果が大きい順に将来タスク化する。

- [ ] **F01（最優先）** H1: `_aggregate_empirical` のカウントを `groupby(["recency","frequency"]).agg(...)` + reindex(fill 0) でベクトル化（`fit()` も同時高速化、数値完全一致）
- [ ] **F02** H4: ロールごとに (r,f)→(N,cv) を事前集約しカウント表を加算（pairs を実体化せずメモリ削減）
- [ ] **F03** H3: 時刻ソート + `searchsorted` でウィンドウを連続区間スライス（ブールマスク全行スキャンの排除）
- [ ] **F04** H2: `UIcv` 集合構築を merge indicator / `isin(MultiIndex)` でベクトル化

> F01 は可読性も損なわないため、本タスクに前倒しする選択肢もある。その場合は Phase 1（純粋移設・既存テスト緑）を先に確定し、その後 F01 をシーム内で適用して再度テスト緑を確認する2段で実施する。

## 進め方の留意

- **Phase 1 は純リファクタ**。完了時点で既存テストが全パスすること（T05）が次フェーズへの前提。挙動が変わっていれば抽出を見直す。
- **Phase 2 完了時点で `fit_rolling` が動作**するが、テストは Phase 3 で追加。
- `_aggregate_empirical` は `obs_log`/`gt_log`/`ref_int` に依存しないことが共有の前提。移設時に依存が残っていないか確認する。
- 集計の重い処理は `_build_ui_rf_cv()` / `_aggregate_empirical()` の2ヘルパーに局所化し、`fit_rolling()` 本体は薄く保つ（将来の高速化シーム / design 11）。
- 各 Phase 完了時に `uv run pytest -q` で回帰確認することを推奨。
