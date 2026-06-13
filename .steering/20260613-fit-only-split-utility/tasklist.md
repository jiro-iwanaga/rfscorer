# Tasklist: `fit()` のみ公開 + `split_by_date()` ユーティリティ提供

## 実装タスク

### Phase 1: 時刻正規化モジュールの抽出

- [ ] **T01** `src/rfscorer/_time_utils.py` を新規作成する
  - `_ORDINAL_ORIGIN = pd.Timestamp("0001-01-01")` モジュール定数
  - `normalize_ref(value) -> int` 関数（既存の `_normalize_ref` ロジックをコピー）
  - `normalize_sequence_col(series) -> pd.Series` 関数（既存の `_normalize_sequence_col` ロジックをコピー）
  - import: `datetime`, `numpy as np`, `pandas as pd`, `pandas.api.types` の dtype チェック関数

- [ ] **T02** `src/rfscorer/scorer.py` の import を更新する
  - `from ._time_utils import normalize_ref, normalize_sequence_col` を追加
  - 既存の `import datetime` / `import numpy as np` / `pandas.api.types` の import は scorer.py 内での利用箇所をチェックし、不要なら削除

- [ ] **T03** `RecencyFrequencyScorer` 内の `_normalize_*` / `_ORDINAL_ORIGIN` を全て新関数呼び出しに置換する
  - `self._normalize_ref(value)` → `normalize_ref(value)`
  - `self._normalize_sequence_col(series)` → `normalize_sequence_col(series)`
  - `self._ORDINAL_ORIGIN` → 参照箇所が消えるはず（消えなければ `_time_utils._ORDINAL_ORIGIN` 経由）

- [ ] **T04** `RecencyFrequencyScorer` クラスから `_normalize_ref` / `_normalize_sequence_col` メソッド・`_ORDINAL_ORIGIN` クラス定数を削除する

### Phase 2: `split_by_date` の追加と公開

- [ ] **T05** `src/rfscorer/utils.py` を新規作成する
  - `from ._time_utils import normalize_ref, normalize_sequence_col` を import
  - `split_by_date(df, target_date, observation_days=28, evaluation_days=7, time_col="datetime")` 関数を実装
  - 入力バリデーション（df 型、time_col 存在）
  - 戻り値 `tuple[pd.DataFrame, pd.DataFrame]`
  - 元の df を mutate しない（boolean mask によるサブセット）
  - target_date は観測期間に含む（`<= target_date`）
  - docstring（design.md の例に準拠）

- [ ] **T06** `src/rfscorer/__init__.py` を更新する
  - `from .utils import split_by_date` を追加
  - `__all__` を更新（`["RecencyFrequencyScorer", "split_by_date"]`）

- [ ] **T07** 動作確認: `uv run python -c "from rfscorer import split_by_date; print(split_by_date.__doc__[:80])"`
  - import 成功と docstring 取得を確認

### Phase 3: `fit_date` / `fit_period` / `transform_date` の削除

- [ ] **T08** `RecencyFrequencyScorer.fit_date()` メソッドを削除する
  - メソッド定義と docstring 全体

- [ ] **T09** `RecencyFrequencyScorer.fit_period()` メソッドを削除する
  - メソッド定義と docstring 全体

- [ ] **T10** `RecencyFrequencyScorer.transform_date()` メソッドを削除する
  - メソッド定義と docstring 全体

- [ ] **T11** 残った docstring 内の言及を整理する
  - `fit()` の docstring の Notes セクションで「`fit_date()` / `fit_period()` 後にも利用可能」となっている表現を「`fit()` 後に利用可能」へ修正
  - `transform()` の docstring 内で「use transform_date() to apply automatic filtering」を「pre-filter df manually before calling transform()」へ修正
  - `predict()` / `plot_*()` / `export_probability_csv()` の docstring 内で `fit()`・`fit_date()`・`fit_period()` の3並列言及を `fit()` のみに修正

### Phase 4: テストの更新

- [ ] **T12** `tests/test_scorer.py` から削除対象テストクラスを削除する
  - `TestFitDateValidation`
  - `TestFitDateResult`
  - `TestFitPeriodValidation`
  - `TestFitPeriodResult`
  - `TestTransformDate`

- [ ] **T13** `tests/test_scorer.py` の共有フィクスチャを `fit()` ベースに書き換える
  - ヘルパー関数 `_split_by_period(df, obs_period, eval_period)` を追加（テスト内 private）
  - `scorer_fitted`, `scorer_optimized_mono`, `scorer_optimized_mr`, `scorer_optimized_mf`, `scorer_optimized_mrc`, `scorer_optimized_mfc`, `scorer_optimized_mcc`, `scorer_all_optimized` の各フィクスチャで `fit_period` → `fit` に変更
  - `df_rec` フィクスチャを `transform_date` → 手動フィルタ + `transform(ref=...)` に変更

- [ ] **T14** `TestNormalizeRef` / `TestNormalizeSequenceCol` を `tests/test_utils.py` に移動する
  - `tests/test_scorer.py` から削除
  - `tests/test_utils.py` 内で `from rfscorer._time_utils import normalize_ref, normalize_sequence_col` を import
  - テストメソッド内の `scorer._normalize_ref(...)` → `normalize_ref(...)` へ書き換え
  - 同様に `_normalize_sequence_col` も
  - フィクスチャ `scorer` は不要になるため削除

- [ ] **T15** `TestIntegerTimeCol` / `TestUnit` 等で残存する `fit_date` / `fit_period` 利用を `fit()` ベースに書き換える
  - 該当箇所を grep して特定
  - 観測・評価期間の手動分割に変更

- [ ] **T16** `tests/test_utils.py` に `TestSplitByDate` を新規追加する
  - `test_basic_split_with_string_date` — 日付文字列入力での基本分割
  - `test_basic_split_with_int_target` — 整数 target_date での基本分割
  - `test_observation_days_none_uses_df_start` — `None` 指定でデータ先頭まで
  - `test_evaluation_days_none_uses_df_end` — `None` 指定でデータ末尾まで
  - `test_observation_days_caps_at_df_start` — observation_days が df 先頭を超える場合のクランプ
  - `test_evaluation_days_caps_at_df_end` — evaluation_days が df 末尾を超える場合のクランプ
  - `test_target_date_inclusive_in_obs` — target_date 当日が df_obs に含まれる
  - `test_target_date_exclusive_in_eval` — target_date 当日が df_eval に含まれない
  - `test_returns_tuple_of_dataframes` — 戻り値の型
  - `test_does_not_mutate_input_df` — 元の df が変更されない
  - `test_preserves_original_columns` — カラム構成を保持
  - `test_preserves_original_time_col_type` — time_col の dtype を保持
  - `test_not_dataframe_raises` — 不正な df で TypeError
  - `test_missing_time_col_raises` — time_col 欠落で ValueError
  - `test_custom_time_col` — `time_col="date"` 等のカスタム名
  - `test_integer_time_col` — 整数列入力での分割
  - `test_invalid_target_date_raises` — 不正な target_date で ValueError
  - `test_chained_with_fit` — `split_by_date` の戻り値を `fit()` にそのまま渡せる

### Phase 5: ドキュメントの更新

- [ ] **T17** `docs/product-requirements.md` を更新する
  - 入力セクションから `fit_date()` / `fit_period()` の引数説明（L59-68 周辺）を削除
  - 「データ準備の補助」サブセクションを追加し、`split_by_date` の使い方を簡潔に紹介

- [ ] **T18** `docs/functional-design.md` を更新する
  - メソッド仕様セクションから `fit_date()` / `fit_period()` / `transform_date()` のサブセクションを削除
  - 「ユーティリティ」セクションを新設し、`split_by_date()` の仕様を記載（または既存セクション末尾に追加）
  - データフロー図の `fit_date(df, target_date)` / `fit_period(df, ...)` 枝を削除
  - 入出力例から `scorer.fit_date(...)`・`scorer.fit_period(...)`・`scorer.transform_date(...)` を削除
  - 入出力例に `df_obs, df_eval = split_by_date(df, "2015-07-06")` + `scorer.fit(df_obs, df_eval)` の例を追加

- [ ] **T19** `docs/glossary.md` を更新する
  - API テーブルから `fit_date(...)` / `fit_period(...)` / `transform_date(...)` エントリを削除
  - `split_by_date(...)` エントリを追加
  - 各属性エントリ内の「`fit()`・`fit_date()` または `fit_period()` 後」のような言及を「`fit()` 後」に修正

- [ ] **T20** `docs/development-guidelines.md` / `docs/architecture.md` / `docs/repository-structure.md` を確認・更新する
  - `fit_date` / `fit_period` / `transform_date` への明示的言及を grep で検出し、該当があれば修正

- [ ] **T21** `examples/basic_usage.ipynb` / `README.md` を確認する
  - `fit_date` / `fit_period` / `transform_date` の利用箇所を grep で確認
  - 該当があれば修正

### Phase 6: 品質チェック

- [ ] **T22** 削除対象の残存ゼロを確認する
  ```bash
  grep -rn "fit_date\|fit_period\|transform_date" src/ tests/ docs/ examples/ README.md \
    | grep -v "^\.steering\|^CHANGELOG"
  ```
  → ZERO matches 期待

- [ ] **T23** 公開 API の動作確認
  ```bash
  uv run python -c "from rfscorer import split_by_date; print(split_by_date.__doc__[:80])"
  ```
  → 成功 + docstring 出力期待

- [ ] **T24** kind alias 保護確認（前回タスクの guarantee 継続）
  - `grep -n '"empirical":' src/rfscorer/scorer.py` → 1件残存
  - `uv run pytest tests/test_scorer.py::TestKindAliases -v` → 全パス

- [ ] **T25** リントを実行する（`uv run ruff check .`）

- [ ] **T26** フォーマットを実行する（`uv run ruff format .`）

- [ ] **T27** テストを実行する（`uv run pytest`）

## 完了条件

すべてのタスクが完了し、以下が成立していること：

1. `uv run ruff check .` が `All checks passed!`
2. `uv run ruff format --check .` が `all files already formatted`
3. `uv run pytest` がすべて green
4. `grep -rn "fit_date\|fit_period\|transform_date" src/ tests/ docs/ examples/ README.md | grep -v "^\.steering\|^CHANGELOG"` が 0 件
5. `from rfscorer import split_by_date` が成功する
6. `tests/test_utils.py` が新規作成され、`TestSplitByDate` の全 case パス
7. `RecencyFrequencyScorer` の `dir()` から `fit_date` / `fit_period` / `transform_date` が消えている
8. `tests/test_utils.py` に `TestNormalizeRef` / `TestNormalizeSequenceCol` が移動済み
9. `tests/test_scorer.py` から `TestNormalizeRef` / `TestNormalizeSequenceCol` が消えている

## 進め方の留意

- **Phase 1 完了時点で scorer.py 単体は self-consistent** だが、テスト側 `TestNormalizeRef` は壊れる（Phase 4 で修正）
- **Phase 3 完了時点でテストは大きく壊れる**（多数のフィクスチャが `fit_period` を使うため）
- **Phase 4 完了時点で全テストが通る状態に戻る** ことが目標
- 各 Phase 完了時に `uv run pytest -q` を走らせて進捗確認することを推奨
