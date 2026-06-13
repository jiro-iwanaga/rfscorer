# Tasklist: time_col / unit サポート

## 実装タスク

### scorer.py の変更

- [ ] **T01** インポートを更新する
  - `is_integer_dtype`, `is_float_dtype` を `pandas.api.types` から追加インポート

- [ ] **T02** 内部定数 `_DATETIME_COL` を `_SEQUENCE_COL` にリネームする
  - クラス変数の定義箇所と、ファイル内すべての `self._DATETIME_COL` 参照を一括置換

- [ ] **T03** `__init__` シグネチャを変更する
  - `datetime_col="datetime"` → `time_col="datetime"`
  - `unit=1` パラメータを追加
  - `unit <= 0` のとき `ValueError` を raise する
  - `self.datetime_col` → `self.time_col`、`self.unit` を追加
  - `*_date_` 属性を `*_` にリネーム（`observation_start_date_` → `observation_start_` など 4 属性）
  - docstring を更新

- [ ] **T04** `_normalize_ref(value) -> int` プライベートメソッドを追加する
  - `pd.Timestamp` / `datetime` → `.toordinal()`
  - `str` → `pd.to_datetime()` 経由で `.toordinal()`
  - `int` / `float` / numpy 数値型 → `int()` キャスト
  - それ以外 → `ValueError`

- [ ] **T05** `_normalize_sequence_col(series) -> pd.Series` プライベートメソッドを追加する
  - `datetime64` → `.map(lambda x: x.toordinal())`
  - 文字列型 → `pd.to_datetime()` 経由で `.map(lambda x: x.toordinal())`
  - `int` / `float` 型 → `.astype(int)`
  - それ以外 → `ValueError`

- [ ] **T06** `_to_internal()` を更新する
  - `self.datetime_col` → `self.time_col`
  - `is_datetime64_any_dtype` / `pd.to_datetime` による変換を削除
  - `_normalize_sequence_col()` の呼び出しに置き換え

- [ ] **T07** `fit()` を更新する
  - パラメータ `ref_date=None` → `ref=None`
  - `pd.to_datetime(ref_date)` を `self._normalize_ref(ref)` に置き換え
  - `obs_log[self._DATETIME_COL].max()` → `obs_log[self._SEQUENCE_COL].max()`
  - `self.observation_start_date_` などを `self.observation_start_` などにリネーム
  - docstring を更新

- [ ] **T08** `fit_date()` を更新する
  - `datetime_col` の存在チェックを `time_col` に変更
  - `pd.to_datetime(target_date)` を `self._normalize_ref(target_date)` に置き換え
  - `pd.to_datetime(df[self.datetime_col])` による min/max 取得を削除し、`_to_internal()` 後の整数列から取得
  - `pd.Timedelta(days=N)` を整数加減算（`target_int ± N`）に置き換え
  - `self.observation_start_date_` などを `self.observation_start_` などにリネーム
  - docstring を更新

- [ ] **T09** `fit_period()` を更新する
  - `datetime_col` の存在チェックを `time_col` に変更
  - `pd.to_datetime(observation_period)` / `pd.to_datetime(evaluation_period)` を `_normalize_ref()` に置き換え
  - `self.observation_start_date_` などを `self.observation_start_` などにリネーム
  - docstring を更新

- [ ] **T10** `_build_ui_rf_df()` を更新する
  - 引数名 `ref_date` → `ref_int`
  - `(ref_date - df[self._DATETIME_COL]).dt.days + 1` → `(ref_int - df[self._SEQUENCE_COL]) // self.unit + 1`
  - docstring を更新

- [ ] **T11** `transform()` を更新する
  - パラメータ `datetime_col=None` → `time_col=None`
  - パラメータ `ref_date=None` → `ref=None`
  - インライン正規化（`is_datetime64_any_dtype` / `pd.to_datetime` による変換）を `_normalize_sequence_col()` に置き換え
  - `ref_date` の `pd.to_datetime()` を `_normalize_ref()` に置き換え
  - docstring を更新

- [ ] **T12** `transform_date()` を更新する
  - パラメータ `datetime_col=None` → `time_col=None`
  - `pd.to_datetime(target_date)` を `self._normalize_ref(target_date)` に置き換え
  - `pd.to_datetime(df[datetime_col_name]) <= target_date` のフィルタを、`_to_internal()` 後の整数列との比較に変更
  - `transform()` への `ref_date=` → `ref=`、`datetime_col=` → `time_col=` に変更
  - docstring を更新

### テストの更新（tests/test_scorer.py）

- [ ] **T13** 既存テストのパラメータ名を一括置換する
  - `datetime_col=` → `time_col=`
  - `ref_date=` → `ref=`
  - `observation_start_date_` / `observation_end_date_` / `evaluation_start_date_` / `evaluation_end_date_` → サフィックス `date_` を削除

- [ ] **T14** `_normalize_ref()` の単体テストを追加する
  - 日付文字列 → ordinal 整数に変換されること
  - `pd.Timestamp` → ordinal 整数に変換されること
  - `int` / `float` → `int` にキャストされること
  - 不正な型 → `ValueError`

- [ ] **T15** `_normalize_sequence_col()` の単体テストを追加する
  - `datetime64` 列 → ordinal 整数列に変換されること
  - 文字列日付列 → ordinal 整数列に変換されること
  - `int` / `float` 列 → `int` 列に変換されること
  - 不正な型 → `ValueError`

- [ ] **T16** 整数入力での `fit()` / `fit_date()` / `fit_period()` のテストを追加する
  - 整数列を `time_col` として渡したとき正常にスコアが計算されること
  - 整数入力と日付入力で、同一の RF 分布が得られること（同じデータを ordinal 変換して比較）

- [ ] **T17** `unit` パラメータのテストを追加する
  - `unit=1`（デフォルト）と `unit=7` で Recency 値が `// 7` の関係になること
  - `unit=0` / `unit=-1` で `ValueError` が raise されること

### ドキュメントの更新

- [ ] **T18** `README.md` を更新する
  - 破壊的変更（`datetime_col` → `time_col`、`ref_date` → `ref`、`*_date_` 属性）を明記
  - `observation_start_` などの属性が日付入力でも ordinal 整数を返すようになることを明記
  - 整数入力の使用例を追加
  - `unit` パラメータの使用例を追加

- [ ] **T19** `CHANGELOG.md` を更新する
  - Breaking Changes セクションに `datetime_col` / `ref_date` / `*_date_` 属性の削除を記載
  - 新機能として整数入力サポートと `unit` パラメータを記載

### 品質チェック

- [ ] **T20** リントを実行する（`uv run ruff check .`）
- [ ] **T21** フォーマットを実行する（`uv run ruff format .`）
- [ ] **T22** テストを実行する（`uv run pytest`）

## 完了条件

すべてのタスクが完了し、T20〜T22 がエラーなく通過していること。
