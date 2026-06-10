# Design: fit/transform API Redesign

## 新しい公開 API

### `fit(df_obs, df_eval, ref_date=None, recency_limit=None, frequency_limit=None)`

```python
scorer.fit(df_obs, df_eval)
scorer.fit(df_obs, df_eval, ref_date="2024-01-07")
```

- `df_obs`: 観測期間のインタラクションログ（ユーザーが事前にフィルタ済み）
- `df_eval`: 評価期間のインタラクションログ（必須）
- `ref_date`: 最新度（recency）計算の基準日。省略時は `df_obs[datetime_col].max()`

内部では日付属性を実データから設定する：

| 属性 | 設定値 |
|---|---|
| `observation_start_date_` | `df_obs[datetime_col].min()` |
| `observation_end_date_` | `ref_date`（または `df_obs.max()`） |
| `evaluation_start_date_` | `df_eval[datetime_col].min()` |
| `evaluation_end_date_` | `df_eval[datetime_col].max()` |

### `transform(df, ref_date=None, kind="emp", user_col=None, item_col=None, datetime_col=None)`

```python
df_obs = df[df["datetime"] <= "2024-01-07"]  # ユーザーが事前フィルタ
scorer.transform(df_obs)
scorer.transform(df_obs, ref_date="2024-01-07")
```

- `df`: 観測期間のインタラクションログ（ユーザーが事前にフィルタ済み）
- `ref_date`: 最新度計算の基準日。省略時は `df[datetime_col].max()`
- 内部の `target_date` フィルタ行を削除

## 後方互換メソッド

### `fit_date(df, target_date, observation_days=28, evaluation_days=7, recency_limit=None, frequency_limit=None)`

旧 `fit()` のロジックをそのまま移動。内部では日付分割後に `fit(df_obs, df_eval, ref_date=obs_end)` を呼ぶ。

### `transform_date(df, target_date, kind="emp", user_col=None, item_col=None, datetime_col=None)`

旧 `transform()` のロジックをそのまま移動。内部で `target_date` フィルタ後に `transform(df_filtered, ref_date=target_date)` を呼ぶ。

### `fit_period(df, observation_period, evaluation_period, recency_limit=None, frequency_limit=None)`

現行のまま保持。内部で日付フィルタ後に `fit(df_obs, df_eval, ref_date=obs_end)` を呼ぶよう統一する。

## 内部アーキテクチャ

重複排除のため、コアロジックを新 `fit()` に集約し、各ラッパーはデータ整形のみ担当する。

```
fit(df_obs, df_eval, ref_date)       ← コアロジック（新）
  ↑
fit_date(df, target_date, ...)       ← 日付分割 → fit() 呼び出し
fit_period(df, obs_period, eval_period, ...) ← 日付フィルタ → fit() 呼び出し

transform(df, ref_date)              ← コアロジック（新）
  ↑
transform_date(df, target_date, ...) ← target_date フィルタ → transform() 呼び出し
```

## `ref_date` のデフォルト挙動と注意点

`ref_date=None` のとき `df_obs[datetime_col].max()` を使用する。
これは「観測データ内で最も新しいインタラクションの日付が recency=1」になることを意味する。

ユーザーが観測期間の終端日（例: `2024-01-07`）を基準にしたい場合は `ref_date` を明示する。
実用上、観測データを正しく end_date までフィルタしていれば両者は一致する。

## テスト変更方針

| 旧テストクラス | 変更後 |
|---|---|
| `TestFitValidation` | 旧テスト → `TestFitDateValidation`、新 `fit(df_obs, df_eval)` のバリデーションテストを `TestFitValidation` として新設 |
| `TestFitResult` | 旧テスト → `TestFitDateResult`、新 `fit(df_obs, df_eval)` の正常系テストを `TestFitResult` として新設 |
| `TestTransform` | `transform(df, target_date)` の呼び出しを `transform_date` に変更、新 `transform(df)` テストを追加 |
| `df_rec` fixture | `transform_date` を使うよう更新 |
